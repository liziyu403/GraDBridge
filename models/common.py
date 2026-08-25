import torch
import torch.nn as nn

from models.component._sanitize import _sanitize
from models.component.GNSBOperatorBiasing import GNSBOperatorFusion


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        x = self.tr(p + self.linear(p))
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        return x.reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                                    x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return _sanitize(torch.cat([_sanitize(t) for t in x], self.d))


class Add(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg

    def forward(self, x):
        return _sanitize(_sanitize(x[0]) + _sanitize(x[1]))


class Add2(nn.Module):
    def __init__(self, c1: int, index: int):
        super().__init__()
        self.index = int(index)
        self.target_c = int(c1)

    @staticmethod
    def _align_channels(t: torch.Tensor, target_c: int) -> torch.Tensor:
        t = _sanitize(t)
        B, C, H, W = t.shape
        if C == target_c:
            return t
        if C > target_c:
            return _sanitize(t[:, :target_c])
        pad = t.new_zeros(B, target_c - C, H, W)
        return _sanitize(torch.cat([t, pad], dim=1))

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2, \
            f"Add2 expects [base, addend], got {type(x)} with len={len(x) if isinstance(x, (list, tuple)) else 'NA'}"
        base = _sanitize(x[0])
        addend = x[1][self.index] if isinstance(x[1], (list, tuple)) else x[1]
        return _sanitize(base + self._align_channels(addend, base.shape[1]))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x = _sanitize(x)
        return _sanitize(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)))


class C3TR(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


__all__ = [
    "Conv",
    "TransformerLayer",
    "TransformerBlock",
    "Bottleneck",
    "SPP",
    "Focus",
    "Concat",
    "Add",
    "Add2",
    "C3",
    "C3TR",
    "GNSBOperatorFusion",
]
