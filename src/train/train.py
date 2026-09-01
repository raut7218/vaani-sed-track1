"""Training: one model, three tiers, mean-teacher consistency.

    python -m src.train.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import TierBatchSampler, VaaniSED, collate, read_manifest, split_manifest
from src.data.labels import LabelEncoder
from src.evaluation.metrics import evaluate
from src.models.beats_encoder import build_beats, download_beats
from src.models.sed_model import VaaniSEDModel
from src.postproc.csebbs import decode_clip, default_params_for, union_events
from src.train.losses import compute_total_loss


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def update_ema(student: torch.nn.Module, teacher: torch.nn.Module, decay: float) -> None:
    """EMA over trainable params only - the frozen BEATs module is shared between
    student and teacher, so averaging it would be pure wasted bandwidth."""
    sp = dict(student.named_parameters())
    for name, tp in teacher.named_parameters():
        if not tp.requires_grad and "beats.beats" in name:
            continue
        p = sp.get(name)
        if p is None:
            continue
        tp.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    sb = dict(student.named_buffers())
    for name, tb in teacher.named_buffers():
        b = sb.get(name)
        if b is not None and tb.dtype == b.dtype:
            tb.copy_(b)


@torch.no_grad()
def infer_scores(model, loader, device, amp: bool):
    """Returns {uid: (T, C) float32 scores}, {uid: n_valid_frames}."""
    model.eval()
    scores, valid = {}, {}
    for batch in loader:
        wav = batch["wav"].to(device, non_blocking=True)
        fv = batch["frame_valid"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            frame_logit, _ = model(wav, tier=None, frame_valid=fv)
        p = torch.sigmoid(frame_logit.float()).cpu().numpy()
        nv = batch["frame_valid"].sum(dim=1).long().cpu().numpy()
        for i, uid in enumerate(batch["uid"]):
            scores[uid] = p[i]
            valid[uid] = int(nv[i])
    return scores, valid


def refs_from_records(records, union_gap: float = 0.0):
    """Ground-truth class-agnostic event spans, matching the submission target."""
    out = {}
    for r in records:
        spans = sorted((float(e["start"]), float(e["end"])) for e in r.get("events", []))
        merged = []
        for a, b in spans:
            if merged and a - merged[-1][1] <= union_gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out[r["uid"]] = [(a, b) for a, b in merged]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default=None, help="override data.root")
    ap.add_argument("--out", default=None, help="override output dir")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-beats", action="store_true")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision (use if you see non-finite loss)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.data:
        cfg["data"]["root"] = args.data
    if args.out:
        cfg["output_dir"] = args.out
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.no_beats:
        cfg["model"]["use_beats"] = False
    if args.no_amp:
        cfg["train"]["amp"] = False

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[train] device=%s" % device)

    root = Path(cfg["data"]["root"])
    recs = read_manifest(root / "manifest.jsonl")
    tr_recs, va_recs = split_manifest(recs, val_frac=cfg["data"]["val_frac"],
                                      seed=cfg.get("seed", 42))
    print("[train] %d train / %d val clips" % (len(tr_recs), len(va_recs)))
    from collections import Counter
    print("[train] train tiers: %s" % dict(Counter(r["tier"] for r in tr_recs)))

    le = LabelEncoder(cfg["data"]["expand_vehicle"])
    fps = float(cfg["data"]["fps"])
    clip_len = float(cfg["data"]["clip_len"])

    ds_tr = VaaniSED(tr_recs, root, le, clip_len, cfg["data"]["sr"], fps, train=True)
    ds_va = VaaniSED(va_recs, root, le, clip_len, cfg["data"]["sr"], fps,
                     train=False, augment=False)

    bs = int(cfg["train"]["batch_size"])
    sampler = TierBatchSampler(tr_recs, bs, cfg["train"]["tier_quotas"], seed=cfg.get("seed", 42))
    nw = int(cfg["train"].get("num_workers", 2))
    dl_tr = DataLoader(ds_tr, batch_sampler=sampler, collate_fn=collate, num_workers=nw,
                       pin_memory=True, persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, collate_fn=collate,
                       num_workers=nw, pin_memory=True) if va_recs else None

    beats = None
    if cfg["model"].get("use_beats", True):
        ck = cfg["model"].get("beats_ckpt") or ""
        if not ck or not Path(ck).exists():
            got = download_beats(cfg["model"].get("beats_dir", "checkpoints"))
            ck = str(got) if got else ""
        beats = build_beats(ck if ck else None, True)

    n_frames = int(round(clip_len * fps))
    mk = lambda: VaaniSEDModel(  # noqa: E731
        n_class=len(le), n_frames=n_frames, beats=beats,
        n_mels=cfg["data"]["n_mels"], sr=cfg["data"]["sr"], hop=cfg["data"]["hop"],
        rnn_dim=cfg["model"]["rnn_dim"], rnn_layers=cfg["model"]["rnn_layers"],
        dropout=cfg["model"]["dropout"], n_basis=cfg["model"]["n_basis"],
        mixstyle_p=cfg["model"]["mixstyle_p"], mixstyle_alpha=cfg["model"]["mixstyle_alpha"],
        use_specaug=cfg["model"]["specaug"])

    student = mk().to(device)
    # Build the teacher through the same factory rather than deepcopy: `mk`
    # reuses the one frozen BEATs instance, so we never hold two 90M copies.
    teacher = mk().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print("[train] trainable params: %.2fM" % (n_train / 1e6))

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = max(1, len(sampler))
    total_steps = epochs * steps_per_epoch
    warmup = int(cfg["train"].get("warmup_steps", min(500, total_steps // 10)))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    loss_cfg = cfg["loss"]
    ema_decay = float(cfg["train"]["ema_decay"])
    params_pp = default_params_for(le.classes)
    va_refs = refs_from_records(va_recs) if va_recs else {}

    best = -1.0
    history = []
    gstep = 0
    n_nonfinite = 0
    nonfinite_limit = int(cfg['train'].get('nonfinite_limit', 50))

    for ep in range(1, epochs + 1):
        student.train()
        teacher.train()
        t0, agg, nb = time.time(), {}, 0
        for batch in dl_tr:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", enabled=amp):
                s_out = student(batch["wav"], tier=batch["tier"],
                                frame_valid=batch["frame_valid"])
                t_out = None
                if loss_cfg.get("lambda_cons", 0) > 0:
                    # Teacher sees the same audio without MixStyle/SpecAugment:
                    # a stable target is the whole point of the EMA branch.
                    teacher.eval()
                    with torch.no_grad():
                        t_out = teacher(batch["wav"], tier=None,
                                        frame_valid=batch["frame_valid"])
                    teacher.train()
                loss, logs = compute_total_loss(s_out, t_out, batch, loss_cfg, gstep)

            # Fail fast on a persistently non-finite loss. GradScaler silently
            # skips such steps, so without this the run burns every epoch
            # updating nothing and reports NaN the whole way down.
            if not torch.isfinite(loss):
                n_nonfinite += 1
                if n_nonfinite >= nonfinite_limit:
                    raise RuntimeError(
                        "loss has been non-finite for %d consecutive steps.\n"
                        "Most likely a mixed-precision issue: rerun with --no-amp "
                        "(or train.amp: false) to confirm.\n"
                        "Last components: %s" % (n_nonfinite, logs))
            else:
                n_nonfinite = 0

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad],
                float(cfg["train"].get("grad_clip", 5.0)))
            scaler.step(opt)
            scaler.update()
            sched.step()
            update_ema(student, teacher, ema_decay)
            gstep += 1
            nb += 1
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v

        msg = " ".join("%s=%.4f" % (k, v / max(1, nb)) for k, v in sorted(agg.items()))
        line = "[ep %d/%d] %s lr=%.2e %.0fs" % (
            ep, epochs, msg, sched.get_last_lr()[0], time.time() - t0)

        if dl_va is not None and (ep % int(cfg["train"].get("eval_every", 1)) == 0):
            for name, m in (("student", student), ("teacher", teacher)):
                sc, vl = infer_scores(m, dl_va, device, amp)
                preds = {u: union_events(
                    decode_clip(sc[u], le.classes, fps, params_pp, n_valid_frames=vl[u]))
                    for u in sc}
                res = evaluate(preds, va_refs, collar_mode=cfg["eval"]["collar_mode"])
                line += "  %s: F1=%.4f dice=%.4f score=%.4f" % (
                    name, res["event_f1"], res["segment_dice"], res["score"])
                if res["score"] > best:
                    best = res["score"]
                    torch.save({"model": m.state_dict(), "cfg": cfg,
                                "classes": le.classes, "which": name, "epoch": ep,
                                "score": best}, out_dir / "best.pt")
                    np.savez_compressed(out_dir / "val_scores.npz",
                                        **{u: sc[u] for u in sc})
                    (out_dir / "val_meta.json").write_text(json.dumps(
                        {"valid": vl, "refs": {u: va_refs[u] for u in sc},
                         "classes": le.classes, "fps": fps}), encoding="utf-8")
                history.append({"epoch": ep, "which": name, **res})
        print(line)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({"model": student.state_dict(), "teacher": teacher.state_dict(),
                    "cfg": cfg, "classes": le.classes, "epoch": ep}, out_dir / "last.pt")

    if best < 0:
        # No timestamped clips to validate against (e.g. a bronze-only batch of
        # the corpus). Still emit best.pt so inference has something to load.
        torch.save({"model": student.state_dict(), "cfg": cfg, "classes": le.classes,
                    "which": "student", "epoch": epochs, "score": None},
                   out_dir / "best.pt")
        print("[train] no validation set - saved final student as best.pt")
    else:
        print("[train] best val score: %.4f  ->  %s" % (best, out_dir / "best.pt"))


if __name__ == "__main__":
    main()
