"""cnn-lobotomy server: neuron explorer + ablation lab for ResNet18 / CIFAR-100. Stdlib HTTP + JSON; no web framework needed.

Ablation targets are (layer, channel, positions) where positions is None (whole channel)
or a list of [y, x] cells in that layer's spatial grid.
"""
import base64
import io
import math
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from build_index import load_cifar_test
from model import LAYERS, MEAN, backward_rule, device, load_model, normalize

IDX = dict(np.load("cache/index.npz"))
X, FINE, COARSE, FINE_NAMES, COARSE_NAMES = load_cifar_test()
X_T = torch.tensor(X).float().div(255)
DEV = device()
MODEL = load_model()  # CPU copy for single-image requests
MODEL_DEV = load_model().to(DEV)  # accelerator copy for full-test-set evals
LOCK = threading.Lock()
MEANMAP = {l: torch.tensor(IDX[f"{l}_meanmap"]) for l in LAYERS}
SHAPES = {l: tuple(IDX[f"{l}_meanmap"].shape) for l in LAYERS}  # (C, H, W)
COARSE_OF = {int(f): int(c) for f, c in zip(FINE, COARSE)}
# Per-channel reference scale (99th pct of dataset spatial-max) so activations are comparable across channels.
REF = {l: np.percentile(IDX[f"{l}_act"].astype(np.float32), 99, axis=0) + 1e-6 for l in LAYERS}


def similar_layout(layer):
    """Arrange channels on a dense grid so channels with similar class profiles sit together:
    sort by PC1 into rows, then by PC2 within each row."""
    cm = IDX[f"{layer}_classmean"].T  # [C, 100]
    z = (cm - cm.mean(1, keepdims=True)) / (cm.std(1, keepdims=True) + 1e-6)
    z = z - z.mean(0)
    u, sv, _ = np.linalg.svd(z, full_matrices=False)
    pc = u[:, :2] * sv[:2]
    C = len(cm)
    side = int(np.ceil(np.sqrt(C)))
    pos = np.zeros((C, 2), int)
    order = np.argsort(pc[:, 0])
    for r in range(side):
        row = order[r * side:(r + 1) * side]
        for k, c in enumerate(row[np.argsort(pc[row, 1])]):
            pos[c] = (k, r)
    return pos.tolist()


def map_data():
    out = []
    for l in LAYERS:
        C = SHAPES[l][0]
        side = int(np.ceil(np.sqrt(C)))
        sel_f, sel_c, top = IDX[f"{l}_sel_fine"], IDX[f"{l}_sel_coarse"], IDX[f"{l}_top"]
        units = []
        for c in range(C):
            f3 = np.argsort(-sel_f[:, c])[:3]
            cc = int(sel_c[:, c].argmax())
            units.append({"coarse": cc, "csel": round(float(sel_c[cc, c]), 3),
                          "fine": [FINE_NAMES[f] for f in f3], "fsel": round(float(sel_f[f3[0], c]), 3),
                          "top": top[c][:4].tolist()})
        out.append({"name": l, "C": C, "H": SHAPES[l][1], "side": side,
                    "grid": [[c % side, c // side] for c in range(C)], "similar": similar_layout(l), "units": units})
    return {"layers": out, "coarse": list(COARSE_NAMES)}


MAP = map_data()


@torch.no_grad()
def image_acts(idx):
    _, f = run(MODEL, X_T[idx:idx + 1], capture=LAYERS)
    return {l: (f[l][0].amax((1, 2)).numpy() / REF[l]).round(3).tolist() for l in LAYERS}


def ablation_hooks(model, ablations, mode):
    """Register forward hooks that overwrite selected channel cells with 0 or their dataset mean."""
    by_layer = {}
    for a in ablations:
        by_layer.setdefault(a["layer"], []).append(a)
    handles = []
    for layer, items in by_layer.items():
        C, H, W = SHAPES[layer]
        mask = torch.zeros(C, H, W, dtype=torch.bool)
        for a in items:
            if a.get("positions"):
                for y, x in a["positions"]:
                    mask[a["channel"], y, x] = True
            else:
                mask[a["channel"]] = True
        fill = MEANMAP[layer] if mode == "mean" else torch.zeros(C, H, W)

        def hook(m, i, o, mask=mask, fill=fill):
            mk, fl = mask.to(o.device), fill.to(o.device)
            return torch.where(mk, fl, o)

        handles.append(model.get_submodule(layer).register_forward_hook(hook))
    return handles


def run(model, x, ablations=(), mode="zero", capture=()):
    feats = {}
    # Ablation hooks first so captured activations (same layer or downstream) reflect the intervention.
    hs = ablation_hooks(model, ablations, mode)
    hs += [model.get_submodule(l).register_forward_hook(lambda m, i, o, l=l: feats.__setitem__(l, o)) for l in capture]
    try:
        logits = model(normalize(x.to(next(model.parameters()).device)))
    finally:
        for h in hs:
            h.remove()
    return logits, feats


def topk(logits, k=5):
    p = logits.softmax(-1)[0]
    v, i = p.topk(k)
    return [{"cls": int(c), "name": FINE_NAMES[c], "p": float(q)} for q, c in zip(v, i)]


def neuron_list(layer, sort, concept):
    act = IDX[f"{layer}_act"].astype(np.float32)
    sel_f, sel_c = IDX[f"{layer}_sel_fine"], IDX[f"{layer}_sel_coarse"]
    if concept:
        kind, k = concept.split(":")
        score = (sel_f if kind == "fine" else sel_c)[int(k)]
    elif sort == "mean":
        score = act.mean(0)
    else:
        score = sel_f.max(0)
    order = np.argsort(-score)
    out = []
    for c in order:
        f, cc = int(sel_f[:, c].argmax()), int(sel_c[:, c].argmax())
        out.append({"channel": int(c), "score": float(score[c]), "fine": FINE_NAMES[f], "fine_sel": float(sel_f[f, c]),
                    "coarse": COARSE_NAMES[cc], "coarse_sel": float(sel_c[cc, c])})
    return out


@torch.no_grad()
def neuron_detail(layer, c, k):
    top = IDX[f"{layer}_top"][c][:k].tolist()
    _, feats = run(MODEL, X_T[top], capture=[layer])
    maps = feats[layer][:, c].numpy()
    act = IDX[f"{layer}_act"][:, c].astype(np.float32)
    sel_f, sel_c, cm = IDX[f"{layer}_sel_fine"][:, c], IDX[f"{layer}_sel_coarse"][:, c], IDX[f"{layer}_classmean"][:, c]
    fo, co = np.argsort(-sel_f)[:10], np.argsort(-sel_c)[:5]
    hist, edges = np.histogram(act, bins=40)
    return {
        "layer": layer, "channel": c, "shape": SHAPES[layer][1:],
        "top": [{"idx": i, "label": FINE_NAMES[FINE[i]], "act": float(act[i]), "map": maps[j].round(3).tolist(),
                 "vmax": float(maps.max())} for j, i in enumerate(top)],
        "fine": [{"name": FINE_NAMES[f], "sel": float(sel_f[f]), "mean": float(cm[f])} for f in fo],
        "coarse": [{"name": COARSE_NAMES[x], "sel": float(sel_c[x])} for x in co],
        "classmean": cm.round(4).tolist(),
        "hist": {"counts": hist.tolist(), "lo": float(edges[0]), "hi": float(edges[-1])},
        # stem kernels act directly on RGB, so their 3x3x3 weights are viewable as tiny images
        "kernel": MODEL.conv1.weight[c].detach().permute(1, 2, 0).numpy().round(4).tolist() if layer == "stem" else None,
    }


def run_image(body):
    idx, abl, mode = int(body["idx"]), body.get("ablations", []), body.get("mode", "zero")
    layer = body.get("layer", LAYERS[-1])
    x = X_T[idx:idx + 1]
    with torch.no_grad():
        base, feats = run(MODEL, x, capture=[layer])
        abl_logits, abl_feats = run(MODEL, x, abl, mode, capture=[layer])
    # Attribution for the clean prediction: act * d(logit)/d(act), summed over space, at the chosen layer.
    cls = int(base.argmax())
    xg = x.clone()
    holder = {}

    def keep(m, i, o):
        o.retain_grad()
        holder["f"] = o

    h = MODEL.get_submodule(layer).register_forward_hook(keep)
    out = MODEL(normalize(xg))
    h.remove()
    out[0, cls].backward()
    f = holder["f"]
    attr = (f * f.grad)[0]
    MODEL.zero_grad(set_to_none=True)
    per_ch = attr.sum((1, 2)).detach()
    order = per_ch.argsort(descending=True)[:12]
    fmap = feats[layer][0]
    return {
        "idx": idx, "label": FINE_NAMES[FINE[idx]], "base": topk(base), "ablated": topk(abl_logits) if abl else None,
        "layer": layer, "shape": SHAPES[layer][1:],
        "attribution": [{"channel": int(c), "attr": float(per_ch[c]), "map": attr[c].detach().numpy().round(4).tolist()}
                        for c in order],
        "pred": FINE_NAMES[cls],
        "feat_max": fmap.amax((1, 2)).numpy().round(4).tolist(),
    }


@torch.no_grad()
def probe(body):
    """Which channels fire at a specific cell of a specific layer, clean vs ablated."""
    idx, layer, y, x = int(body["idx"]), body["layer"], int(body["y"]), int(body["x"])
    _, clean = run(MODEL, X_T[idx:idx + 1], capture=[layer])
    v = clean[layer][0, :, y, x]
    order = v.argsort(descending=True)[:20]
    return {"cells": [{"channel": int(c), "act": float(v[c]),
                       "map": clean[layer][0, c].numpy().round(3).tolist()} for c in order]}


@torch.no_grad()
def channel_map(body):
    idx, layer, c = int(body["idx"]), body["layer"], int(body["channel"])
    _, f = run(MODEL, X_T[idx:idx + 1], capture=[layer])
    _, fa = run(MODEL, X_T[idx:idx + 1], body.get("ablations", []), body.get("mode", "zero"), capture=[layer])
    return {"map": f[layer][0, c].numpy().round(3).tolist(), "ablated": fa[layer][0, c].numpy().round(3).tolist()}


def _center_rf(layer):
    """Input-pixel box (y0, y1, x0, x1) that influences the center cell of `layer`."""
    feats = {}
    h = MODEL.get_submodule(layer).register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
    x = torch.full((1, 3, 32, 32), 0.5, requires_grad=True)
    MODEL(normalize(x))
    h.remove()
    f = feats["f"]
    f[0, :, f.shape[2] // 2, f.shape[3] // 2].sum().backward()
    MODEL.zero_grad(set_to_none=True)
    ys, xs = torch.nonzero(x.grad.abs().sum((0, 1)) > 0, as_tuple=True)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


CENTER_RF = {l: _center_rf(l) for l in LAYERS}


def rf_box(layer, y, x):
    """Receptive field of cell (y, x): the center cell's box shifted by the layer stride, clipped to the image."""
    _, H, W = SHAPES[layer]
    st = 32 // H
    y0, y1, x0, x1 = CENTER_RF[layer]
    dy, dx = (y - H // 2) * st, (x - W // 2) * st
    return max(0, y0 + dy), min(32, y1 + dy), max(0, x0 + dx), min(32, x1 + dx)


def png_url(arr):
    """HxWx3 float array in [0,1] -> data: URL."""
    buf = io.BytesIO()
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def project(idxs, layer, c, rule):
    """Zeiler & Fergus: for each image, keep only the channel's strongest unit and project it back to pixels.
    Returns per image the unit position, its RF box, the projection and the matching image patch."""
    x = X_T[idxs].clone().requires_grad_(True)
    feats = {}
    h = MODEL.get_submodule(layer).register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
    with backward_rule(rule):
        MODEL(normalize(x))
        f = feats["f"][:, c]
        flat = f.flatten(1)
        val, am = flat.max(1)
        sel = torch.zeros_like(flat).scatter(1, am[:, None], val[:, None]).view_as(f)  # zero every other unit
        (f * sel).sum().backward()  # seeding with the unit's own activation, as in the paper
    h.remove()
    MODEL.zero_grad(set_to_none=True)
    g = x.grad.detach()
    W = f.shape[2]
    out = []
    for k, i in enumerate(idxs):
        y, xx = divmod(int(am[k]), W)
        y0, y1, x0, x1 = rf_box(layer, y, xx)
        gk = g[k, :, y0:y1, x0:x1]
        gk = (gk / (gk.abs().flatten().quantile(0.99) + 1e-8)).clamp(-1, 1)  # per-projection contrast; clip hot pixels
        out.append({"idx": int(i), "label": FINE_NAMES[FINE[i]], "act": float(val[k]), "pos": [y, xx], "box": [y0, y1, x0, x1],
                    "proj": png_url((0.5 + 0.5 * gk).permute(1, 2, 0).numpy()),
                    "patch": png_url(X_T[i, :, y0:y1, x0:x1].permute(1, 2, 0).numpy())})
    return out


def zeiler_fergus(layer, c, k, rule):
    top = IDX[f"{layer}_top"][c][:k].tolist()
    rf = CENTER_RF[layer]
    return {"layer": layer, "channel": c, "rule": rule, "rf": [rf[1] - rf[0], rf[3] - rf[2]], "items": project(top, layer, c, rule)}


@torch.no_grad()
def occlusion(body):
    """Slide a mean-gray square over the image; record how much the target score drops at each position.
    Target is a class logit (default: the clean prediction) or a channel's spatial max."""
    idx, size, stride = int(body["idx"]), int(body.get("size", 8)), int(body.get("stride", 2))
    x = X_T[idx:idx + 1]
    pos = list(range(0, 32 - size + 1, stride))
    batch = x.repeat(len(pos) ** 2, 1, 1, 1)
    gray = torch.tensor(MEAN).view(3, 1, 1)
    for n, (a, b) in enumerate((a, b) for a in pos for b in pos):
        batch[n, :, a:a + size, b:b + size] = gray
    t = body.get("target", {})
    if t.get("layer"):
        layer, c = t["layer"], int(t["channel"])
        score = lambda inp: run(MODEL, inp, capture=[layer])[1][layer][:, c].amax((1, 2))
        name = f"{layer} #{c}"
    else:
        cls = t.get("cls")
        cls = int(run(MODEL, x)[0].argmax()) if cls is None else int(cls)
        score = lambda inp: run(MODEL, inp)[0].softmax(-1)[:, cls]
        name = f"p({FINE_NAMES[cls]})"
    base = float(score(x)[0])
    drops = base - torch.cat([score(batch[i:i + 256]) for i in range(0, len(batch), 256)])
    # accumulate each window's drop over the pixels it covered -> per-pixel importance at full resolution
    heat, cover = torch.zeros(32, 32), torch.zeros(32, 32)
    for n, (a, b) in enumerate((a, b) for a in pos for b in pos):
        heat[a:a + size, b:b + size] += drops[n]
        cover[a:a + size, b:b + size] += 1
    heat = heat / cover.clamp(min=1)
    return {"target": name, "base": base, "map": heat.numpy().round(4).tolist(), "size": size}


BASELINE = {}


@torch.no_grad()
def predict_all(abl, mode):
    preds = []
    for i in range(0, len(X_T), 1000):
        logits, _ = run(MODEL_DEV, X_T[i:i + 1000], abl, mode)
        preds.append(logits.argmax(1).cpu())
    return torch.cat(preds).numpy()


def evaluate(body):
    abl, mode = body.get("ablations", []), body.get("mode", "zero")
    if "p" not in BASELINE:
        BASELINE["p"] = predict_all([], mode)
    p0, p1 = BASELINE["p"], predict_all(abl, mode)
    ok0, ok1 = p0 == FINE, p1 == FINE
    cls = [{"cls": k, "name": FINE_NAMES[k], "coarse": COARSE_NAMES[COARSE_OF[k]],
            "before": float(ok0[FINE == k].mean()), "after": float(ok1[FINE == k].mean())} for k in range(100)]
    sup = [{"name": COARSE_NAMES[k], "before": float(ok0[COARSE == k].mean()), "after": float(ok1[COARSE == k].mean())}
           for k in range(20)]
    # where did the ablated images go?
    changed = (p0 != p1)
    flow = {}
    for a, b in zip(p0[changed], p1[changed]):
        flow[(int(a), int(b))] = flow.get((int(a), int(b)), 0) + 1
    flows = sorted(flow.items(), key=lambda kv: -kv[1])[:15]
    return {"overall": [float(ok0.mean()), float(ok1.mean())], "classes": cls, "coarse": sup,
            "flows": [{"from": FINE_NAMES[a], "to": FINE_NAMES[b], "n": n} for (a, b), n in flows]}


SWEEP_NS = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 128, 200]


def mcnemar_p(broke, fixed):
    """One-sided exact McNemar test: P(at least `broke` of the discordant pairs went correct->wrong under H0)."""
    n = broke + fixed
    return sum(math.comb(n, k) for k in range(broke, n + 1)) / 2 ** n if n else 1.0


SWEEP_CACHE = {}


def sweep(body):
    key = (body["layer"], body["concept"], tuple(body.get("ns", SWEEP_NS)))
    if key not in SWEEP_CACHE:  # deterministic for a given model, so repeat requests are free
        SWEEP_CACHE[key] = _sweep(body)
    return SWEEP_CACHE[key]


def _sweep(body):
    """Ablate the top-N channels for a concept (ranked by d') at one layer, for increasing N.
    Compares zero vs mean ablation and a random-N-channel control, on concept vs all other images."""
    layer, concept = body["layer"], body["concept"]
    kind, k = concept.split(":")
    k = int(k)
    target = (FINE if kind == "fine" else COARSE) == k
    score = IDX[f"{layer}_sel_{'fine' if kind == 'fine' else 'coarse'}"][k]
    order = np.argsort(-score)
    rand = np.random.default_rng(0).permutation(len(order))  # fixed control ordering, nested across N
    Ns = [n for n in body.get("ns", SWEEP_NS) if n < len(order)]  # N = all channels would just delete the layer
    if "p" not in BASELINE:
        BASELINE["p"] = predict_all([], "zero")
    ok0 = BASELINE["p"] == FINE
    acc = lambda ok, m: float(ok[m].mean())
    rows = []
    for n in Ns:
        row = {"n": n}
        for key, chans, mode in [("top_zero", order[:n], "zero"), ("top_mean", order[:n], "mean"), ("rand_zero", rand[:n], "zero")]:
            ok = predict_all([{"layer": layer, "channel": int(c), "positions": None} for c in chans], mode) == FINE
            broke, fixed = int((ok0 & ~ok & target).sum()), int((~ok0 & ok & target).sum())
            row[key] = {"target": acc(ok, target), "other": acc(ok, ~target), "p": mcnemar_p(broke, fixed)}
        rows.append(row)
    base_t, base_o = acc(ok0, target), acc(ok0, ~target)
    first = lambda pred: next((r["n"] for r in rows if pred(r)), None)
    return {
        "layer": layer, "concept": concept, "n_target": int(target.sum()), "base": {"target": base_t, "other": base_o},
        "rows": rows,
        "first_significant": first(lambda r: r["top_zero"]["p"] < 0.05),
        "first_halved": first(lambda r: r["top_zero"]["target"] <= base_t / 2),
        "first_off_target": first(lambda r: base_o - r["top_zero"]["other"] >= 0.02),
    }


def image_png(idx, scale):
    img = Image.fromarray(X[idx].transpose(1, 2, 0)).resize((32 * scale, 32 * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def meta():
    acc = float((IDX["preds"] == FINE).mean())
    return {"layers": [{"name": l, "C": SHAPES[l][0], "H": SHAPES[l][1], "W": SHAPES[l][2]} for l in LAYERS],
            "fine": list(FINE_NAMES), "coarse": list(COARSE_NAMES), "acc": acc,
            "fine_to_coarse": [COARSE_OF[k] for k in range(100)],
            "by_class": {k: np.where(FINE == k)[0][:60].tolist() for k in range(100)}}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        if ctype == "application/json":
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if ctype == "image/png" else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = u.path.strip("/").split("/")
        try:
            if u.path in ("/", "/index.html"):
                return self.send(200, open("static/index.html", "rb").read(), "text/html; charset=utf-8")
            if u.path == "/static/map.js":
                return self.send(200, open("static/map.js", "rb").read(), "text/javascript; charset=utf-8")
            if u.path == "/api/meta":
                return self.send(200, meta())
            if u.path == "/api/map":
                return self.send(200, MAP)
            if parts[:2] == ["api", "fv"] and parts[2] in LAYERS:
                prefix = "fv_center_" if q.get("kind") == "center" else "fv_"
                return self.send(200, open(f"cache/{prefix}{parts[2]}.png", "rb").read(), "image/png")
            if parts[:2] == ["api", "acts"]:
                with LOCK:
                    return self.send(200, image_acts(int(parts[2])))
            if u.path == "/api/neurons":
                return self.send(200, neuron_list(q["layer"], q.get("sort", "sel"), q.get("concept")))
            if parts[:2] == ["api", "neuron"]:
                with LOCK:
                    return self.send(200, neuron_detail(parts[2], int(parts[3]), int(q.get("k", 24))))
            if parts[:2] == ["api", "zf"]:
                with LOCK:
                    return self.send(200, zeiler_fergus(parts[2], int(parts[3]), int(q.get("k", 9)), q.get("rule", "guided")))
            if parts[:2] == ["api", "img"]:
                return self.send(200, image_png(int(parts[2]), int(q.get("s", 4))), "image/png")
            self.send(404, {"error": "not found"})
        except Exception as e:  # surface errors in the UI instead of hanging
            self.send(500, {"error": repr(e)})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or "{}")
        routes = {"/api/run": run_image, "/api/probe": probe, "/api/map": channel_map, "/api/eval": evaluate, "/api/occlusion": occlusion, "/api/sweep": sweep}
        fn = routes.get(self.path)
        if not fn:
            return self.send(404, {"error": "not found"})
        try:
            with LOCK:
                return self.send(200, fn(body))
        except Exception as e:
            self.send(500, {"error": repr(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"serving on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
