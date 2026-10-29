"""Train ResNet18 on CIFAR-100. Data lives on-device and is augmented there (no DataLoader overhead)."""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from model import ResNet18, device, normalize


def to_tensor(ds, dev):
    x = torch.tensor(np.asarray(ds.data)).permute(0, 3, 1, 2).float().div(255).to(dev)
    y = torch.tensor(ds.targets).to(dev)
    return x, y


def augment(x):
    n = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    ar = torch.arange(32, device=x.device)
    rows = (torch.randint(0, 9, (n, 1), device=x.device) + ar)[:, None, :, None]  # per-sample random crop
    cols = (torch.randint(0, 9, (n, 1), device=x.device) + ar)[:, None, None, :]
    x = x[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = sum((model(normalize(x[i:i + bs])).argmax(1) == y[i:i + bs]).sum().item() for i in range(0, len(x), bs))
    return correct / len(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    a = p.parse_args()

    dev = device()
    tr = torchvision.datasets.CIFAR100("data", train=True, download=True)
    te = torchvision.datasets.CIFAR100("data", train=False, download=True)
    xtr, ytr = to_tensor(tr, dev)
    xte, yte = to_tensor(te, dev)

    model = ResNet18().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps = a.epochs * (len(xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.15)
    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(a.epochs):
        model.train()
        t0, perm, tot = time.time(), torch.randperm(len(xtr), device=dev), 0.0
        for k in range(len(xtr) // a.bs):
            idx = perm[k * a.bs:(k + 1) * a.bs]
            loss = F.cross_entropy(model(normalize(augment(xtr[idx]))), ytr[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() if k % 50 == 0 else 0
        acc = evaluate(model, xte, yte)
        print(f"epoch {ep + 1}/{a.epochs}  test_acc {acc:.4f}  {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), "checkpoints/resnet18_cifar100.pt")


if __name__ == "__main__":
    main()
