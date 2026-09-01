"""Per-tier masked loss - the core trick for Track 1.

A clip contributes only the loss terms its annotation quality can support:

    gold    frame BCE (full weight) + clip BCE
    silver  frame BCE (down-weighted) + clip BCE
    bronze  clip BCE only, backpropagated through attention pooling

Training only on the ~22 h of verified data would discard ~85% of the corpus.
Training on all of it as if it were verified would teach the model that silver's
looser boundaries and bronze's absent ones are ground truth. The mask is what
lets one model use every clip for exactly what it is worth.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

TIER_ORDER = ["gold", "silver", "bronze"]


def masked_frame_bce(logits: torch.Tensor, target: torch.Tensor,
                     frame_valid: torch.Tensor, sample_w: torch.Tensor) -> torch.Tensor:
    """logits/target: (B, T, C); frame_valid: (B, T); sample_w: (B,)"""
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # (B,T,C)
    m = (frame_valid[..., None] * sample_w[:, None, None]).expand_as(per)
    return (per * m).sum() / m.sum().clamp(min=1.0)


def masked_clip_bce(probs: torch.Tensor, target: torch.Tensor,
                    sample_w: torch.Tensor) -> torch.Tensor:
    """probs: (B, C) already in (0,1) from attention pooling.

    BCE is written out by hand in float32 rather than via
    `F.binary_cross_entropy`, which torch refuses to run under autocast. We
    cannot use the `_with_logits` form here either: attention pooling produces a
    weighted *average of probabilities*, which has no corresponding logit before
    the pooling step.
    """
    p = probs.float().clamp(1e-7, 1.0 - 1e-7)
    t = target.float()
    per = -(t * torch.log(p) + (1.0 - t) * torch.log1p(-p))                     # (B,C)
    w = sample_w.float()[:, None].expand_as(per)
    return (per * w).sum() / w.sum().clamp(min=1.0)


def tier_weights(tier: torch.Tensor, cfg: dict, key: str) -> torch.Tensor:
    """Look up a per-sample weight from the tier id tensor."""
    w = torch.zeros_like(tier, dtype=torch.float32)
    for i, name in enumerate(TIER_ORDER):
        w = torch.where(tier == i, torch.full_like(w, float(cfg[key].get(name, 0.0))), w)
    return w


def consistency_loss(student: torch.Tensor, teacher: torch.Tensor,
                     frame_valid: torch.Tensor | None = None) -> torch.Tensor:
    """MSE between student and (detached) teacher predictions."""
    t = teacher.detach()
    per = (student - t) ** 2
    if frame_valid is not None and per.dim() == 3:
        m = frame_valid[..., None].expand_as(per)
        return (per * m).sum() / m.sum().clamp(min=1.0)
    return per.mean()


def sigmoid_rampup(step: int, length: int) -> float:
    """Standard mean-teacher ramp-up: consistency must not dominate before the
    teacher is worth listening to."""
    if length <= 0:
        return 1.0
    x = float(max(0.0, min(1.0, step / length)))
    return math.exp(-5.0 * (1.0 - x) ** 2)


def compute_total_loss(student_out, teacher_out, batch, cfg, step: int) -> tuple:
    """Returns (loss, logs).

    `logs` holds *detached GPU tensors*, not Python floats. Calling `float()` on
    a live CUDA tensor forces a device synchronisation, and doing that three or
    four times inside every training step drains the CUDA queue and serialises
    the host against the device. The training loop accumulates these on-device
    and reads them once per epoch instead.
    """
    s_frame_logit, s_clip = student_out
    tier = batch["tier"]
    frame_t, clip_t = batch["frame_target"], batch["clip_target"]
    frame_valid = batch["frame_valid"]

    w_strong = tier_weights(tier, cfg, "strong_weight") * batch["strong_mask"]
    w_weak = tier_weights(tier, cfg, "weak_weight")

    l_strong = masked_frame_bce(s_frame_logit, frame_t, frame_valid, w_strong)
    l_weak = masked_clip_bce(s_clip, clip_t, w_weak)
    loss = cfg["lambda_strong"] * l_strong + cfg["lambda_weak"] * l_weak
    logs = {"loss_strong": l_strong.detach(), "loss_weak": l_weak.detach()}

    if teacher_out is not None and cfg.get("lambda_cons", 0.0) > 0:
        t_frame_logit, t_clip = teacher_out
        ramp = sigmoid_rampup(step, cfg.get("cons_rampup_steps", 3000))
        c_frame = consistency_loss(torch.sigmoid(s_frame_logit),
                                   torch.sigmoid(t_frame_logit), frame_valid)
        c_clip = consistency_loss(s_clip, t_clip)
        l_cons = c_frame + c_clip
        loss = loss + cfg["lambda_cons"] * ramp * l_cons
        logs.update({"loss_cons": l_cons.detach(), "cons_ramp": ramp})

    logs["loss"] = loss.detach()
    return loss, logs
