"""The single SED model: mel + frozen BEATs -> FDY-CRNN -> frame & clip heads.

One model serves all three tiers. Tier handling lives entirely in the loss mask,
never in the architecture - training three per-tier models would throw away the
shared representation that makes the weak tiers useful in the first place.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from src.models.beats_encoder import align_time
from src.models.fdy_crnn import AttentionPool, FDYCNN
from src.models.mixstyle import FreqMixStyle


class LogMel(nn.Module):
    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop: int = 160,
                 n_mels: int = 128, fmin: int = 0, fmax: int = 8000):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            f_min=fmin, f_max=fmax, n_mels=n_mels, power=2.0, center=True)

    def forward(self, wav: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        # The whole front-end runs in float32 with autocast disabled.
        #
        # Under fp16 autocast this produced NaN from the first step: clips are
        # zero-padded to the window length, so the padded region gives mel power
        # of exactly 0, and the 1e-10 clamp floor is itself below the smallest
        # fp16 subnormal (6e-8) - it rounds to 0.0 and stops guarding log().
        # log(0) = -inf, and the per-clip mean/std then turn the whole batch NaN.
        # (bf16 hides this: it has fp32's exponent range.)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            m = self.mel(wav.float())                  # (B, n_mels, T)
            m = torch.log(m.clamp(min=1e-10))
            return self._normalise(m, valid)

    def _normalise(self, m: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        # Per-clip normalisation: recording level varies hugely across Vaani
        # districts and devices, and absolute loudness is not the signal.
        #
        # Statistics are computed over *valid* frames only. Vaani clips are far
        # shorter than the window (corpus mean ~6 s vs a 10 s window), so the
        # padding fraction varies from clip to clip; normalising over the padded
        # region would make every clip's scaling depend on its own length and
        # leak that length into the features.
        if valid is None:
            mu = m.mean(dim=(1, 2), keepdim=True)
            sd = m.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        else:
            # `valid` arrives on the output frame grid; stretch it to mel frames.
            w = torch.nn.functional.interpolate(
                valid[:, None, :].to(m.dtype), size=m.size(-1), mode="nearest")
            n = (w.sum(dim=(1, 2), keepdim=True) * m.size(1)).clamp(min=1.0)
            mu = (m * w).sum(dim=(1, 2), keepdim=True) / n
            var = ((m - mu) ** 2 * w).sum(dim=(1, 2), keepdim=True) / n
            sd = var.sqrt().clamp(min=1e-5)
        return ((m - mu) / sd).unsqueeze(1)            # (B, 1, F, T)


class SpecAugment(nn.Module):
    def __init__(self, n_freq_mask: int = 2, freq_width: int = 16,
                 n_time_mask: int = 2, time_width: int = 40):
        super().__init__()
        self.nf, self.fw, self.nt, self.tw = n_freq_mask, freq_width, n_time_mask, time_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, _, F, T = x.shape
        for _ in range(self.nf):
            w = int(torch.randint(0, self.fw + 1, (1,)).item())
            if w:
                f0 = int(torch.randint(0, max(1, F - w), (1,)).item())
                x[:, :, f0:f0 + w, :] = 0
        for _ in range(self.nt):
            w = int(torch.randint(0, self.tw + 1, (1,)).item())
            if w:
                t0 = int(torch.randint(0, max(1, T - w), (1,)).item())
                x[:, :, :, t0:t0 + w] = 0
        return x


class VaaniSEDModel(nn.Module):
    def __init__(self, n_class: int, n_frames: int, beats=None, n_mels: int = 128,
                 sr: int = 16000, hop: int = 160, rnn_dim: int = 256, rnn_layers: int = 2,
                 dropout: float = 0.2, n_basis: int = 4, mixstyle_p: float = 0.5,
                 mixstyle_alpha: float = 0.3, use_specaug: bool = True,
                 cnn_channels=(32, 64, 128, 256, 256, 256, 256)):
        super().__init__()
        self.n_class, self.n_frames = n_class, n_frames
        self.logmel = LogMel(sr=sr, hop=hop, n_mels=n_mels)
        self.specaug = SpecAugment() if use_specaug else nn.Identity()
        self.mixstyle = FreqMixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.cnn = FDYCNN(n_mels=n_mels, channels=cnn_channels, n_basis=n_basis,
                          dropout=dropout * 0.5)

        self.beats = beats
        feat_dim = self.cnn.out_dim
        if beats is not None:
            # Project BEATs down before fusion so the 768-d encoder does not
            # swamp the CNN branch in the concatenated representation.
            self.beats_proj = nn.Sequential(
                nn.Linear(beats.out_dim, 256), nn.LayerNorm(256), nn.GELU())
            feat_dim += 256

        self.pre_rnn = nn.Sequential(nn.Linear(feat_dim, rnn_dim), nn.LayerNorm(rnn_dim),
                                     nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(rnn_dim, rnn_dim, num_layers=rnn_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if rnn_layers > 1 else 0.0)
        self.head_drop = nn.Dropout(dropout)
        self.head = AttentionPool(rnn_dim * 2, n_class)

    def forward(self, wav: torch.Tensor, tier: torch.Tensor | None = None,
                frame_valid: torch.Tensor | None = None):
        x = self.logmel(wav, frame_valid)           # (B, 1, F, T)
        x = self.mixstyle(x, tier)
        x = self.specaug(x)
        h = self.cnn(x)                            # (B, T', D)

        if self.beats is not None:
            with torch.no_grad():
                b = self.beats(wav)                # (B, Tb, 768)
                # Frozen encoder: a non-finite value here can only poison the
                # trainable branch, so neutralise it rather than propagate it.
                b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
            b = self.beats_proj(b.to(h.dtype))
            h = torch.cat([h, align_time(b, h.size(1))], dim=-1)

        h = self.pre_rnn(h)
        h, _ = self.rnn(h)
        h = self.head_drop(h)
        h = align_time(h, self.n_frames)           # fix output rate to the target grid
        return self.head(h, frame_valid)           # (frame_logits, clip_probs)
