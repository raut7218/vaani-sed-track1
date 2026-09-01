"""Frequency-Dynamic CRNN backbone.

Plain 2-D convolution is translation-equivariant along frequency, which is wrong
for sound events: a kernel that detects a horn at 800 Hz should not be the same
kernel applied at 4 kHz. Frequency-dynamic convolution (Nam et al., 2022) fixes
this by making the kernel a frequency-dependent mixture of K basis kernels.

`FDYConv2d` degrades to an ordinary conv when `n_basis == 1`, so the same
backbone code runs the ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqAttention(nn.Module):
    """Per-frequency attention over the K basis kernels."""

    def __init__(self, in_ch: int, n_basis: int, temperature: float = 31.0,
                 hidden_ratio: int = 4):
        super().__init__()
        hidden = max(in_ch // hidden_ratio, 8)
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, n_basis, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, F, T) -> (B, K, F)"""
        z = x.mean(dim=3)                      # pool time -> (B, C, F)
        a = self.net(z)                        # (B, K, F)
        return torch.softmax(a / self.temperature, dim=1)


class FDYConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, n_basis: int = 4, temperature: float = 31.0,
                 bias: bool = False):
        super().__init__()
        self.in_ch, self.out_ch, self.n_basis = in_ch, out_ch, n_basis
        self.k, self.stride, self.padding = kernel_size, stride, padding
        self.weight = nn.Parameter(
            torch.empty(n_basis * out_ch, in_ch, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(n_basis * out_ch)) if bias else None
        self.att = FreqAttention(in_ch, n_basis, temperature) if n_basis > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, F, T)"""
        B = x.size(0)
        y = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        if self.att is None:
            return y
        Fo, To = y.shape[-2], y.shape[-1]
        y = y.view(B, self.n_basis, self.out_ch, Fo, To)
        a = self.att(x)                                    # (B, K, F_in)
        if a.size(-1) != Fo:                               # stride changed F
            a = F.interpolate(a, size=Fo, mode="linear", align_corners=False)
        y = (y * a[:, :, None, :, None]).sum(dim=1)
        return y


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int], n_basis: int,
                 dropout: float = 0.0, temperature: float = 31.0):
        super().__init__()
        self.conv = FDYConv2d(in_ch, out_ch, 3, 1, 1, n_basis=n_basis, temperature=temperature)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.pool = nn.AvgPool2d(pool) if pool != (1, 1) else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class FDYCNN(nn.Module):
    """(B, 1, F, T) -> (B, T', D) with D = channels[-1] * F'."""

    def __init__(self, n_mels: int = 128, channels=(32, 64, 128, 256, 256, 256, 256),
                 # (freq_pool, time_pool) per block; time pooling totals 4 -> 25 fps
                 pools=((2, 2), (2, 2), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)),
                 n_basis: int = 4, dropout: float = 0.1, temperature: float = 31.0,
                 fdy_from_block: int = 1):
        super().__init__()
        blocks, in_ch, f = [], 1, n_mels
        for i, (c, p) in enumerate(zip(channels, pools)):
            # Frequency-dynamic conv only from block `fdy_from_block` on: the first
            # block sees raw mel bins where the basis mixture just wastes compute.
            k = n_basis if i >= fdy_from_block else 1
            blocks.append(ConvBlock(in_ch, c, p, k, dropout, temperature))
            in_ch = c
            f = max(1, f // p[0])
        self.blocks = nn.Sequential(*blocks)
        self.out_freq = f
        self.out_dim = channels[-1] * f
        self.time_pool = 1
        for p in pools:
            self.time_pool *= p[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)                     # (B, C, F', T')
        B, C, Fp, Tp = x.shape
        return x.permute(0, 3, 1, 2).reshape(B, Tp, C * Fp)


class AttentionPool(nn.Module):
    """Frame logits -> clip logits.

    This is the bridge that lets bronze clips (tag only, no timestamps) train the
    frame-level representation: the clip loss backpropagates through the softmax
    attention into every frame.
    """

    def __init__(self, in_dim: int, n_class: int):
        super().__init__()
        self.strong = nn.Linear(in_dim, n_class)
        self.att = nn.Linear(in_dim, n_class)

    def forward(self, h: torch.Tensor, frame_valid: torch.Tensor | None = None):
        strong_logit = self.strong(h)                       # (B, T, C)
        a = self.att(h)
        if frame_valid is not None:
            a = a.masked_fill(frame_valid[..., None] < 0.5, torch.finfo(a.dtype).min)
        a = torch.softmax(a, dim=1)
        strong = torch.sigmoid(strong_logit)
        clip = (strong * a).sum(dim=1).clamp(1e-7, 1 - 1e-7)
        return strong_logit, clip
