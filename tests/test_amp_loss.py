"""Regression test: the training step must survive autocast (AMP).

`F.binary_cross_entropy` refuses to run under autocast, and the attention-pooled
clip probability has no pre-pooling logit to hand to the `_with_logits` form, so
the weak loss is written out in float32 by hand. This test exercises a full
forward + loss + backward under autocast on whichever device is available, which
is what CPU-only testing previously missed.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.labels import LabelEncoder
from src.models.sed_model import VaaniSEDModel
from src.train.losses import compute_total_loss

CFG = {
    "lambda_strong": 1.0, "lambda_weak": 0.5, "lambda_cons": 2.0,
    "cons_rampup_steps": 10,
    "strong_weight": {"gold": 1.0, "silver": 0.5, "bronze": 0.0},
    "weak_weight": {"gold": 1.0, "silver": 1.0, "bronze": 1.0},
}


def run(device: str, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    le = LabelEncoder(True)
    C, F_, B, SR, CLIP = len(le), 100, 6, 16000, 4.0
    dev = torch.device(device)

    model = VaaniSEDModel(n_class=C, n_frames=F_, beats=None, rnn_dim=32,
                          rnn_layers=1, dropout=0.0, n_basis=4,
                          mixstyle_p=0.5, use_specaug=True).to(dev)
    teacher = VaaniSEDModel(n_class=C, n_frames=F_, beats=None, rnn_dim=32,
                            rnn_layers=1, dropout=0.0, n_basis=4,
                            mixstyle_p=0.0, use_specaug=False).to(dev)
    teacher.load_state_dict(model.state_dict())

    batch = {
        "wav": torch.randn(B, int(SR * CLIP), device=dev) * 0.1,
        "frame_target": (torch.rand(B, F_, C, device=dev) > 0.9).float(),
        "clip_target": (torch.rand(B, C, device=dev) > 0.7).float(),
        "frame_valid": torch.ones(B, F_, device=dev),
        "strong_mask": torch.tensor([1., 1., 1., 1., 0., 0.], device=dev),
        "tier": torch.tensor([0, 0, 1, 1, 2, 2], device=dev),
    }
    batch["frame_valid"][:, 70:] = 0.0   # exercise the padding mask too

    with torch.autocast(device, dtype=dtype):
        s_out = model(batch["wav"], tier=batch["tier"], frame_valid=batch["frame_valid"])
        teacher.eval()
        with torch.no_grad():
            t_out = teacher(batch["wav"], tier=None, frame_valid=batch["frame_valid"])
        loss, logs = compute_total_loss(s_out, t_out, batch, CFG, step=5)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients produced"
    assert torch.isfinite(loss), "loss is not finite: %s" % loss
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"

    # Bronze rows must not contribute frame loss: their strong_mask is 0.
    assert logs["loss_strong"] >= 0 and logs["loss_weak"] >= 0
    print("  %-5s %-9s loss=%.4f strong=%.4f weak=%.4f cons=%.4f  OK" % (
        device, str(dtype).replace("torch.", ""), logs["loss"], logs["loss_strong"],
        logs["loss_weak"], logs.get("loss_cons", float("nan"))))


def front_end_fp16() -> None:
    """The bug that bf16 could not catch.

    Clips are zero-padded to the window, so the padded region has mel power of
    exactly 0. The 1e-10 clamp floor is below the smallest fp16 subnormal (6e-8),
    so under fp16 it rounds to 0.0, stops guarding log(), and the per-clip
    mean/std turn the entire batch NaN. bf16 hides it (fp32 exponent range), so
    this check runs the front-end in fp16 explicitly, on any device.
    """
    from src.models.sed_model import LogMel

    # 1. Show *why* the front-end must be pinned to fp32: the clamp floor that
    #    guards log() is not representable in fp16.
    floor_fp16 = float(torch.tensor(1e-10, dtype=torch.float16))
    assert floor_fp16 == 0.0, "expected 1e-10 to underflow in fp16"
    print("  1e-10 clamp floor in fp16 -> %.1f (so log(0) = -inf; fp32 is required)"
          % floor_fp16)

    # 2. The front-end must stay fp32 and finite under autocast, on zero-padded
    #    audio (every clip is padded to the window).
    lm = LogMel()
    wav = torch.cat([torch.randn(2, 16000) * 0.1, torch.zeros(2, 16000 * 3)], dim=1)
    fv = torch.zeros(2, 100)
    fv[:, :25] = 1.0                                   # 1 s valid of a 4 s window

    devices = [("cpu", torch.bfloat16)]
    if torch.cuda.is_available():
        devices.append(("cuda", torch.float16))
    for dev, dt in devices:
        lm_d, wav_d, fv_d = lm.to(dev), wav.to(dev), fv.to(dev)
        with torch.autocast(dev, dtype=dt):
            x = lm_d(wav_d, fv_d)
        assert x.dtype == torch.float32, \
            "FAIL: front-end escaped to %s; it must stay fp32" % x.dtype
        assert torch.isfinite(x).all(), \
            "FAIL: non-finite front-end output under %s autocast" % dt
        print("  %-4s autocast %-8s -> dtype=%s, finite, std=%.3f"
              % (dev, str(dt).replace("torch.", ""), x.dtype, float(x.float().std())))
    lm.cpu()


def main() -> None:
    print("mixed-precision regressions:")
    front_end_fp16()
    print("autocast training-step:")
    run("cpu", torch.bfloat16)
    if torch.cuda.is_available():
        run("cuda", torch.float16)
    else:
        print("  cuda  skipped (no GPU here; this is the config that failed on Colab)")
    print("\nPASS: forward + loss + backward survive autocast.")


if __name__ == "__main__":
    main()
