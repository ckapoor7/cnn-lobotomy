"""Activation maximization for every channel of every unit, in two flavors:

channel: maximize the channel's mean activation over its whole spatial map -> tiling textures.
    Fourier parameterization with 1/f scaling (as in Lucid / Distill), random jitter, small TV penalty.
center:  maximize the single unit at the map's center cell (the center-pixel unit). The input is a
    perturbation of mid-gray kept inside an L2 ball, so a linear layer's optimum is its own kernel
    rather than a saturated color fill. Each result is cropped to the unit's receptive field.

Saved as one sprite sheet per unit and flavor (16 tiles per row, 32px tiles).
"""
import math

import numpy as np
import torch
from PIL import Image

from model import LAYERS, device, load_model, normalize

STEPS, LR, TV, COLS = 256, 0.03, 0.2, 16
CENTER_RMS = 0.25  # center mode: RMS pixel deviation from gray allowed within the receptive field


def tv(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., 1:] - x[..., :-1]).abs().mean()


def fourier_raw(spec, size=40, decay=1.0):
    """spec: [N, 3, size, size//2+1, 2] -> unbounded images with a 1/f^decay spectrum."""
    fy = torch.fft.fftfreq(size, device=spec.device)[:, None]
    fx = torch.fft.rfftfreq(size, device=spec.device)[None, :]
    scale = 1.0 / torch.sqrt(fx ** 2 + fy ** 2).clamp(min=1.0 / size) ** decay
    return torch.fft.irfft2(torch.view_as_complex(spec) * scale, s=(size, size), norm="ortho")


def fourier_image(spec, size=40, decay=1.0):
    return torch.sigmoid(fourier_raw(spec, size, decay))


def save_sheet(imgs, path):
    n, rows = len(imgs), math.ceil(len(imgs) / COLS)
    sheet = torch.zeros(3, rows * 32, COLS * 32)
    for k, im in enumerate(imgs):
        r, c = divmod(k, COLS)
        sheet[:, r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = im
    Image.fromarray((sheet.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(path)


def receptive_field(model, feats, dev):
    """Bounding box (y0, y1, x0, x1) of input pixels that affect the center cell of the hooked unit."""
    x = torch.full((1, 3, 32, 32), 0.5, device=dev, requires_grad=True)
    model(normalize(x))
    f = feats["f"]
    f[0, :, f.shape[2] // 2, f.shape[3] // 2].sum().backward()
    ys, xs = torch.nonzero(x.grad.abs().sum((0, 1)) > 0, as_tuple=True)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def pre_relu_hook(model, layer, feats):
    """Capture the unit's value before its final ReLU, so units that start out negative still get gradient."""
    if layer == "stem":
        return model.bn1.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
    block = model.get_submodule(layer)

    def hook(m, i, o):
        x = i[0]
        feats["f"] = m.bn2(m.conv2(torch.relu(m.bn1(m.conv1(x))))) + m.shortcut(x)

    return block.register_forward_hook(hook)


def visualize_center(model, layer, dev, chunk=256):
    feats = {}
    h = pre_relu_hook(model, layer, feats)
    y0, y1, x0, x1 = receptive_field(model, feats, dev)
    radius = CENTER_RMS * math.sqrt(3 * (y1 - y0) * (x1 - x0))
    decay = 0.5 if y1 - y0 <= 11 else 1.0  # small RFs need high frequencies; large ones need the smoother prior
    n, out = feats["f"].shape[1], []
    for c0 in range(0, n, chunk):
        idx = torch.arange(c0, min(n, c0 + chunk), device=dev)
        p = (torch.randn(len(idx), 3, 32, 17, 2, device=dev) * 0.01).requires_grad_()
        opt = torch.optim.Adam([p], lr=LR)

        def image():
            d = fourier_raw(p, 32, decay=decay)
            d = d * (radius / d.flatten(1).norm(dim=1).clamp(min=radius)).view(-1, 1, 1, 1)  # project into L2 ball
            return (0.5 + d).clamp(0, 1)

        for _ in range(STEPS):
            model(normalize(image()))
            f = feats["f"]
            act = f[torch.arange(len(idx)), idx, f.shape[2] // 2, f.shape[3] // 2]
            opt.zero_grad()
            (-act.sum()).backward()
            opt.step()
        with torch.no_grad():
            crop = image()[:, :, y0:y1, x0:x1]
            out.append(torch.nn.functional.interpolate(crop, size=32, mode="nearest").cpu())
    h.remove()
    save_sheet(torch.cat(out), f"cache/fv_center_{layer}.png")
    print("center", layer, n, f"rf={y1 - y0}x{x1 - x0}", flush=True)


def visualize(model, layer, dev, chunk=256):
    C = model.get_submodule(layer)
    feats = {}
    h = C.register_forward_hook(lambda m, i, o: feats.__setitem__("f", o))
    n = None
    out = []
    with torch.no_grad():
        model(normalize(torch.zeros(1, 3, 32, 32, device=dev)))
        n = feats["f"].shape[1]
    for c0 in range(0, n, chunk):
        idx = torch.arange(c0, min(n, c0 + chunk), device=dev)
        p = (torch.randn(len(idx), 3, 40, 21, 2, device=dev) * 0.01).requires_grad_()
        opt = torch.optim.Adam([p], lr=LR)
        for _ in range(STEPS):
            i, j = np.random.randint(0, 9, 2)
            x = fourier_image(p)[:, :, i:i + 32, j:j + 32]
            model(normalize(x))
            act = feats["f"][torch.arange(len(idx)), idx].mean((1, 2))
            loss = -act.sum() + TV * len(idx) * tv(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out.append(fourier_image(p)[:, :, 4:36, 4:36].cpu())
    h.remove()
    save_sheet(torch.cat(out), f"cache/fv_{layer}.png")
    print("channel", layer, n, flush=True)


def main():
    dev = device()
    model = load_model().to(dev)
    for prm in model.parameters():
        prm.requires_grad_(False)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("layers", nargs="*", default=LAYERS)
    ap.add_argument("--mode", choices=["channel", "center", "both"], default="both")
    a = ap.parse_args()
    for layer in a.layers:
        if a.mode in ("channel", "both"):
            visualize(model, layer, dev)
        if a.mode in ("center", "both"):
            visualize_center(model, layer, dev)


if __name__ == "__main__":
    main()
