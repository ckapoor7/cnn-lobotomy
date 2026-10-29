"""Run the CIFAR-100 test set through the model and cache per-channel statistics.

Unit of analysis: each output channel of layer1..layer4 (block outputs, i.e. the residual stream).
Per image we reduce a channel's HxW map to its spatial max (where it fires most strongly) and keep
the argmax location; full maps are recomputed on demand by the server for fine-grained views.
"""
import pickle

import numpy as np
import torch

from model import LAYERS, device, load_model, normalize

TOPK = 32


def load_cifar_test():
    with open("data/cifar-100-python/test", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    with open("data/cifar-100-python/meta", "rb") as f:
        m = pickle.load(f, encoding="latin1")
    x = d["data"].reshape(-1, 3, 32, 32)
    return x, np.array(d["fine_labels"]), np.array(d["coarse_labels"]), m["fine_label_names"], m["coarse_label_names"]


def selectivity(acts, labels, n):
    """d'-style score per (label, channel): (mean_in - mean_out) / pooled std."""
    out = np.zeros((n, acts.shape[1]), np.float32)
    for k in range(n):
        a, b = acts[labels == k], acts[labels != k]
        out[k] = (a.mean(0) - b.mean(0)) / np.sqrt(0.5 * (a.var(0) + b.var(0)) + 1e-6)
    return out


@torch.no_grad()
def main():
    dev = device()
    model = load_model().to(dev)
    x, fine, coarse, fine_names, coarse_names = load_cifar_test()
    acts = {l: [] for l in LAYERS}
    pos = {l: [] for l in LAYERS}
    mean_map = {l: 0 for l in LAYERS}
    feats = {}
    hooks = [model.get_submodule(l).register_forward_hook(lambda m, i, o, l=l: feats.__setitem__(l, o)) for l in LAYERS]
    preds = []
    for i in range(0, len(x), 500):
        xb = torch.tensor(x[i:i + 500]).float().div(255).to(dev)
        preds.append(model(normalize(xb)).argmax(1).cpu())
        for l in LAYERS:
            f = feats[l]
            flat = f.flatten(2)
            mx, am = flat.max(2)
            acts[l].append(mx.cpu())
            pos[l].append(am.to(torch.int16).cpu())
            mean_map[l] = mean_map[l] + f.sum(0).cpu()
    for h in hooks:
        h.remove()

    preds = torch.cat(preds).numpy()
    out = dict(fine=fine, coarse=coarse, preds=preds, fine_names=np.array(fine_names), coarse_names=np.array(coarse_names))
    for l in LAYERS:
        a = torch.cat(acts[l]).numpy()
        out[f"{l}_act"] = a.astype(np.float16)
        out[f"{l}_pos"] = torch.cat(pos[l]).numpy()
        out[f"{l}_meanmap"] = (mean_map[l] / len(x)).numpy()
        out[f"{l}_top"] = np.argsort(-a, axis=0)[:TOPK].T  # [C, TOPK] image indices
        out[f"{l}_classmean"] = np.stack([a[fine == k].mean(0) for k in range(100)])
        out[f"{l}_sel_fine"] = selectivity(a, fine, 100)
        out[f"{l}_sel_coarse"] = selectivity(a, coarse, 20)
        print(l, a.shape)
    np.savez_compressed("cache/index.npz", **out)
    print("test acc", (preds == fine).mean())


if __name__ == "__main__":
    import os
    os.makedirs("cache", exist_ok=True)
    main()
