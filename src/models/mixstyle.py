"""Frequency MixStyle.

Mixes *frequency-wise* feature statistics between clips in a batch, which
simulates the channel / device / room variation that separates Vaani's states,
languages and recording devices. Statistics are pooled over channel and time so
each frequency bin keeps its own mean/std, and only those get mixed.

The permutation is restricted to samples sharing a tier: mixing a bronze clip's
frequency response into a gold clip would blur the very supervision split the
masked loss depends on.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FreqMixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, tier: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, C, F, T)"""
        if not self.training or self.p <= 0 or torch.rand(1).item() > self.p or x.size(0) < 2:
            return x

        B = x.size(0)
        # One statistic per frequency bin, pooled over channel and time.
        mu = x.mean(dim=[1, 3], keepdim=True)
        var = x.var(dim=[1, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        perm = self._tier_perm(tier, B, x.device)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sig_mix = sig * lam + sig[perm] * (1 - lam)
        return x_norm * sig_mix + mu_mix

    @staticmethod
    def _tier_perm(tier: torch.Tensor | None, B: int, device) -> torch.Tensor:
        """Permutation that only pairs samples within the same tier."""
        if tier is None:
            return torch.randperm(B, device=device)
        perm = torch.arange(B, device=device)
        for t in tier.unique():
            idx = (tier == t).nonzero(as_tuple=True)[0]
            if idx.numel() > 1:
                perm[idx] = idx[torch.randperm(idx.numel(), device=device)]
        return perm
