"""Frozen BEATs encoder wrapper.

BEATs is the single largest lever in the SED literature (0.353 -> 0.497 PSDS1 in
a controlled DCASE test). We keep it frozen: fine-tuning a 90M-parameter encoder
on ~20 h of verified data overfits, and it costs GPU time we do not have on a
Colab session.

Deliberately *not* wav2vec2/HuBERT/WavLM: those are trained to discard non-speech
content, and benchmark at or below a no-pretraining baseline on SED.

The wrapper degrades gracefully - if the checkpoint is absent, `build_beats`
returns None and the model runs mel-only so the pipeline still trains.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Self-supervised + AS2M checkpoint, the one DCASE Task 4 systems use.
BEATS_REPO = "lpepino/beats_ckpts"
BEATS_FILE = "BEATs_iter3_plus_AS2M.pt"


def download_beats(dest: str | Path = "checkpoints") -> Path | None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    local = dest / BEATS_FILE
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=BEATS_REPO, filename=BEATS_FILE, repo_type="model")
        import shutil
        shutil.copy(p, local)
        return local
    except Exception as e:  # noqa: BLE001
        print("[beats] download failed: %s" % e)
        return None


class FrozenBEATs(nn.Module):
    """Wraps BEATs and exposes (B, T', D) time-major features."""

    def __init__(self, ckpt_path: str | Path):
        super().__init__()
        from third_party.beats.BEATs import BEATs, BEATsConfig

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print("[beats] missing keys: %d (first: %s)" % (len(missing), missing[:3]))
        # The predictor head would collapse the sequence to AudioSet logits; we
        # want the hidden states, so drop it and let extract_features return them.
        model.predictor = None
        self.beats = model
        self.out_dim = cfg.encoder_embed_dim
        self.n_freq_patches = max(1, 128 // cfg.input_patch_size)
        for p in self.beats.parameters():
            p.requires_grad = False
        self.beats.eval()

    def train(self, mode: bool = True):  # keep frozen encoder in eval forever
        super().train(mode)
        self.beats.eval()
        return self

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, L) at 16 kHz -> (B, T', D)"""
        out = self.beats.extract_features(wav)
        x = out[0] if isinstance(out, (tuple, list)) else out   # (B, N, D)
        B, N, D = x.shape
        nf = self.n_freq_patches
        if N % nf == 0:
            # Tokens are (time_patches x freq_patches); average the freq axis to
            # get one embedding per time step.
            x = x.view(B, N // nf, nf, D).mean(dim=2)
        return x


def build_beats(ckpt_path: str | Path | None, enabled: bool = True) -> FrozenBEATs | None:
    if not enabled:
        return None
    if ckpt_path is None:
        return None
    p = Path(ckpt_path)
    if not p.exists():
        print("[beats] checkpoint not found at %s - running mel-only" % p)
        return None
    try:
        m = FrozenBEATs(p)
        print("[beats] loaded %s (dim=%d)" % (p.name, m.out_dim))
        return m
    except Exception as e:  # noqa: BLE001
        print("[beats] failed to load (%s) - running mel-only" % e)
        return None


def align_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """(B, T, D) -> (B, target_len, D) by linear interpolation along time."""
    if x.size(1) == target_len:
        return x
    return F.interpolate(x.transpose(1, 2), size=target_len, mode="linear",
                         align_corners=False).transpose(1, 2)
