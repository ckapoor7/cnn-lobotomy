"""CIFAR-style ResNet18 (3x3 stem, no maxpool) so late layers keep spatial resolution."""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hookable units: stem output and every residual block output (the residual stream after each block).
LAYERS = ["stem"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)]
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


class _BackwardReLU(torch.autograd.Function):
    """ReLU whose backward pass follows a visualization rule instead of the true gradient.
    deconv (Zeiler & Fergus 2013): pass only positive signal, ignoring the forward mask.
    guided (Springenberg et al. 2015): positive signal AND forward mask."""

    @staticmethod
    def forward(ctx, x, guided):
        ctx.save_for_backward(x)
        ctx.guided = guided
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        g = g.clamp(min=0)
        return (g * (x > 0) if ctx.guided else g), None


_RULE = None  # None = ordinary ReLU


def relu(x):
    return F.relu(x) if _RULE is None else _BackwardReLU.apply(x, _RULE == "guided")


@contextlib.contextmanager
def backward_rule(rule):
    """Within this context every ReLU in the model uses the given backward rule ("deconv" or "guided")."""
    global _RULE
    prev, _RULE = _RULE, rule
    try:
        yield
    finally:
        _RULE = prev


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.stem = nn.Identity()  # hook point for the post-ReLU stem output
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize(x):
    """x: uint8-range float tensor scaled to [0,1], NCHW."""
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def load_model(path="checkpoints/resnet18_cifar100.pt"):
    m = ResNet18()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()
