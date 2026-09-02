"""Training: one model, three tiers, mean-teacher consistency.

    python -m src.train.train --config configs/default.yaml

Multi-GPU (2x T4 on Kaggle, or any single-node multi-GPU box) via DistributedDataParallel:

    torchrun --standalone --nproc_per_node=2 -m src.train.train --config configs/default.yaml

`torchrun` sets RANK/WORLD_SIZE/LOCAL_RANK before launching each process; a plain
`python -m src.train.train` never sets them, so world_size defaults to 1 and every
distributed branch below is skipped - single-GPU/CPU behaviour is unchanged. `--batch-size`
is the *per-process* batch size (the standard DDP convention), so the global batch is
`batch_size * world_size`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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


def build_ema_pairs(student: torch.nn.Module, teacher: torch.nn.Module):
    """Pair up the tensors the EMA has to touch, once, at startup.

    Resolving names through `named_parameters` on every step meant rebuilding two
    dicts and issuing two kernel launches per tensor - a few hundred launches per
    step, which a Colab host CPU cannot feed fast enough to keep the GPU busy.
    Pairing here lets the update run as two fused `_foreach` calls instead.

    Only trainable tensors are paired. The frozen BEATs module is *the same
    object* in both models, so its parameters and buffers are filtered out by
    identity - averaging a tensor onto itself is pure wasted bandwidth.

    Always call this with the raw (unwrapped) student module, never a
    DistributedDataParallel wrapper - DDP prefixes every name with "module.",
    which would silently break the name lookup against the teacher's names.
    """
    sp = dict(student.named_parameters())
    s_par, t_par = [], []
    for name, tp in teacher.named_parameters():
        p = sp.get(name)
        if p is None or not p.requires_grad or p is tp:
            continue
        s_par.append(p)
        t_par.append(tp)

    sb = dict(student.named_buffers())
    s_buf, t_buf = [], []
    for name, tb in teacher.named_buffers():
        b = sb.get(name)
        if b is None or b is tb or tb.dtype != b.dtype or tb.shape != b.shape:
            continue
        s_buf.append(b)
        t_buf.append(tb)
    return s_par, t_par, s_buf, t_buf


@torch.no_grad()
def update_ema(pairs, decay: float) -> None:
    """teacher = decay * teacher + (1 - decay) * student, plus buffer sync.

    Under DDP every rank calls this with its own local `student` parameters -
    those are already bitwise-identical across ranks because the optimiser step
    that just ran consumed gradients DDP had already all-reduced, so the teacher
    this produces stays in sync across ranks too without any extra communication.
    """
    s_par, t_par, s_buf, t_buf = pairs
    if t_par:
        torch._foreach_mul_(t_par, decay)
        torch._foreach_add_(t_par, s_par, alpha=1.0 - decay)
    # BatchNorm running stats are copied, not averaged: they are already an EMA
    # of the student's batch statistics.
    if t_buf:
        for tb, b in zip(t_buf, s_buf):
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
    ap.add_argument("--batch-size", type=int, default=None,
                    help="per-process batch size; global batch = this * world_size under DDP")
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

    # ---- distributed setup --------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    dist = None
    if is_distributed:
        import torch.distributed as dist
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(*a) -> None:
        # Every rank runs the identical loop; only rank 0 prints and touches
        # disk, so checkpoints/history never get two writers at once.
        if rank == 0:
            print(*a)

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    # Every training step sees the identical tensor shape (the window is fixed),
    # so cuDNN's autotuner pays for itself in the first few steps and then hands
    # back the fastest algorithm for this conv stack for the rest of the run.
    torch.backends.cudnn.benchmark = True
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log("[train] device=%s  world_size=%d" % (device, world_size))

    root = Path(cfg["data"]["root"])
    if rank == 0 and "drive" in str(root.resolve()).lower() and "mydrive" in str(root.resolve()).lower():
        print("[train] WARNING: data.root (%s) looks like a Google Drive mount. Random "
              "small-file reads over Drive's FUSE layer during training are extremely slow - "
              "this is the classic 'nothing happens for 20 minutes' stall before the first "
              "epoch prints anything. Copy the dataset to local disk first, e.g.:\n"
              "    rsync -a '%s/' /content/work/data/\n"
              "and point --data at that local copy instead." % (root, root))
    recs = read_manifest(root / "manifest.jsonl")
    tr_recs, va_recs = split_manifest(recs, val_frac=cfg["data"]["val_frac"],
                                      seed=cfg.get("seed", 42))
    if is_distributed:
        # Shard only the *training* records: each rank then trains on a
        # disjoint slice, so a per-process --batch-size of B gives a global
        # batch of B * world_size, the standard DDP convention. Validation
        # stays whole and only runs on rank 0 - duplicating it on every rank
        # would waste GPU time without changing the number it reports.
        tr_recs = tr_recs[rank::world_size]
    log("[train] %d train / %d val clips (this rank)" % (len(tr_recs), len(va_recs)))
    from collections import Counter
    log("[train] train tiers: %s" % dict(Counter(r["tier"] for r in tr_recs)))

    le = LabelEncoder(cfg["data"]["expand_vehicle"])
    fps = float(cfg["data"]["fps"])
    clip_len = float(cfg["data"]["clip_len"])

    ds_tr = VaaniSED(tr_recs, root, le, clip_len, cfg["data"]["sr"], fps, train=True)
    ds_va = VaaniSED(va_recs, root, le, clip_len, cfg["data"]["sr"], fps,
                     train=False, augment=False) if (rank == 0 and va_recs) else None

    bs = int(cfg["train"]["batch_size"])
    sampler = TierBatchSampler(tr_recs, bs, cfg["train"]["tier_quotas"], seed=cfg.get("seed", 42))
    if is_distributed:
        # An uneven split can leave one rank's per-tier pools one batch short
        # of another's. DDP requires every rank to call backward() the same
        # number of times per epoch - a rank that runs out of batches early
        # would leave the rest stuck waiting on its share of the next
        # all-reduce forever. Agree on the shortest epoch and use that
        # everywhere.
        local_nb = torch.tensor([len(sampler)], dtype=torch.long, device=device)
        dist.all_reduce(local_nb, op=dist.ReduceOp.MIN)
        sampler._nb = max(1, int(local_nb.item()))

    # Clamp to the real core count, split fairly across the ranks sharing this
    # machine: two DDP processes each independently asking for `num_workers`
    # would otherwise oversubscribe the CPU on a single-node multi-GPU box.
    nw = max(0, min(int(cfg["train"].get("num_workers", 2)),
                    (os.cpu_count() or 1) // max(1, world_size)))
    dl_kw = dict(collate_fn=collate, num_workers=nw, pin_memory=True,
                 persistent_workers=nw > 0)
    if nw > 0:
        dl_kw["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 4))
    dl_tr = DataLoader(ds_tr, batch_sampler=sampler, **dl_kw)
    # The val loader keeps its workers alive too - it is re-entered every epoch,
    # and respawning them each time costs more than the evaluation itself on a
    # small validation split. Rank 0 only: see the sharding note above.
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, **dl_kw) if ds_va is not None else None

    beats = None
    if cfg["model"].get("use_beats", True):
        ck = cfg["model"].get("beats_ckpt") or ""
        beats_dir = cfg["model"].get("beats_dir", "checkpoints")
        if not ck or not Path(ck).exists():
            if rank == 0:
                download_beats(beats_dir)
            if is_distributed:
                # Ranks on a single-node multi-GPU box share one local disk:
                # let rank 0 finish the ~360 MB download before anyone else
                # tries to read the same file mid-write.
                dist.barrier()
            got = download_beats(beats_dir)  # already cached everywhere now - instant
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
    # The teacher is only ever run to produce a stable target, which means always
    # in eval mode. Pinning it here removes a train()/eval() round trip - each of
    # which walks every submodule, BEATs' 12 transformer layers included - from
    # every single step.
    teacher.eval()
    ema_pairs = build_ema_pairs(student, teacher)

    # Materialised once: rebuilding this list inside the step just to clip
    # gradients walks every parameter of a 90M-parameter model every iteration.
    trainable = [p for p in student.parameters() if p.requires_grad]
    log("[train] trainable params: %.2fM" % (sum(p.numel() for p in trainable) / 1e6))

    opt = torch.optim.AdamW(
        trainable,
        lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    # `student_fwd` is what the step below actually calls: the DDP wrapper when
    # distributed (so gradients get all-reduced across GPUs every backward()),
    # or `student` itself otherwise. The optimiser, EMA and every checkpoint
    # still go through the raw `student` module - DDP wraps by reference rather
    # than copying, so those are the exact tensors DDP fills with all-reduced
    # gradients, and state_dict() stays free of the "module." prefix DDP would
    # otherwise add (which would break loading into the plain model at
    # inference time).
    student_fwd = student
    if is_distributed:
        ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank) \
            if device.type == "cuda" else {}
        student_fwd = torch.nn.parallel.DistributedDataParallel(student, **ddp_kwargs)

    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = max(1, len(sampler))
    total_steps = epochs * steps_per_epoch
    # Cap warmup against the actual run length. A configured 500-step warmup on a
    # run that only has 30 steps means the LR never leaves the ramp and the model
    # trains at ~6e-5 instead of 1e-3 the whole way.
    warmup = int(cfg["train"].get("warmup_steps", 500))
    warmup_cap = max(1, total_steps // 10)
    if warmup > warmup_cap:
        log("[train] warmup %d steps > 10%% of the %d-step run; capping to %d"
            % (warmup, total_steps, warmup_cap))
        warmup = warmup_cap

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
    va_refs = refs_from_records(va_recs) if (rank == 0 and va_recs) else {}

    best = -1.0
    history = []
    gstep = 0
    nonfinite_limit = int(cfg["train"].get("nonfinite_limit", 50))
    # Consecutive-non-finite counter kept *on the device*. Reading it every step
    # would reintroduce the synchronisation this loop is built to avoid, so it is
    # maintained with device-side arithmetic and only read back periodically.
    nonfinite_run = torch.zeros((), device=device)
    check_every = int(cfg["train"].get("nonfinite_check_every", 50))

    log_every = int(cfg["train"].get("log_every", 50))
    log("[train] %d steps/epoch/rank, %d epochs -> %d steps total  (global batch %d)"
        % (steps_per_epoch, epochs, total_steps, bs * world_size))

    for ep in range(1, epochs + 1):
        student_fwd.train()
        t0, agg, nb = time.time(), {}, 0
        for batch in dl_tr:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast("cuda", enabled=amp):
                # One frozen-encoder pass for both branches. Student and teacher
                # see the identical waveform and share the identical frozen BEATs
                # module, so a second pass would recompute the same tensor for
                # roughly a third of the step's wall clock. Called on the raw
                # `student`, never `student_fwd`: it is a @torch.no_grad()
                # forward through frozen (requires_grad=False) parameters, so
                # it never needs DDP's gradient-sync hooks.
                beats_feat = student.encode_beats(batch["wav"])
                s_out = student_fwd(batch["wav"], tier=batch["tier"],
                                    frame_valid=batch["frame_valid"],
                                    beats_feat=beats_feat)
                t_out = None
                if loss_cfg.get("lambda_cons", 0) > 0:
                    # Teacher sees the same audio without MixStyle/SpecAugment:
                    # a stable target is the whole point of the EMA branch.
                    with torch.no_grad():
                        t_out = teacher(batch["wav"], tier=None,
                                        frame_valid=batch["frame_valid"],
                                        beats_feat=beats_feat)
                loss, logs = compute_total_loss(s_out, t_out, batch, loss_cfg, gstep)

            # Fail fast on a persistently non-finite loss. GradScaler silently
            # skips such steps, so without this the run burns every epoch
            # updating nothing and reports NaN the whole way down. The counter
            # multiplies by `bad`, so any finite step resets it to zero.
            bad = (~torch.isfinite(loss)).to(nonfinite_run.dtype)
            if is_distributed:
                # Each rank's local loss comes from different data and can be
                # finite on one rank while not on another; without agreeing
                # here, only the unlucky rank would raise and exit while the
                # rest hang forever waiting for its share of the next
                # all-reduce.
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            nonfinite_run = (nonfinite_run + bad) * bad
            if gstep % check_every == 0 and float(nonfinite_run) >= nonfinite_limit:
                raise RuntimeError(
                    "loss has been non-finite for %d consecutive steps.\n"
                    "Most likely a mixed-precision issue: rerun with --no-amp "
                    "(or train.amp: false) to confirm."
                    % int(nonfinite_run))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                trainable, float(cfg["train"].get("grad_clip", 5.0)))
            scaler.step(opt)
            scaler.update()
            sched.step()
            # Ramp the EMA decay in. At a fixed 0.999 the teacher needs ~3000
            # steps to forget its random init, so on shorter runs it stays near
            # noise and its predictions are meaningless. This is the standard
            # mean-teacher warmup and is a no-op once the run is long enough.
            update_ema(ema_pairs, min(1.0 - 1.0 / (gstep + 1), ema_decay))
            gstep += 1
            nb += 1
            # Accumulated on-device; read back once, below, after the epoch.
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v

            # A heartbeat. An epoch here is thousands of steps and the summary
            # line only lands after validation, so without this a healthy run is
            # indistinguishable from a hung one for tens of minutes. Reading the
            # loss syncs, hence once every `log_every` steps rather than always.
            if rank == 0 and log_every and nb % log_every == 0:
                el = time.time() - t0
                print("[ep %d/%d] step %d/%d  loss=%.4f  %.2fs/step  eta %.1fm"
                      % (ep, epochs, nb, steps_per_epoch, float(logs["loss"]),
                         el / nb, (steps_per_epoch - nb) * el / nb / 60), flush=True)

        agg = {k: (float(v) / max(1, nb)) for k, v in agg.items()}
        msg = " ".join("%s=%.4f" % (k, v) for k, v in sorted(agg.items()))
        line = "[ep %d/%d] %s lr=%.2e %.0fs" % (
            ep, epochs, msg, sched.get_last_lr()[0], time.time() - t0)

        if rank == 0 and dl_va is not None and (ep % int(cfg["train"].get("eval_every", 1)) == 0):
            for name, m in (("student", student), ("teacher", teacher)):
                sc, vl = infer_scores(m, dl_va, device, amp)
                preds = {u: union_events(
                    decode_clip(sc[u], le.classes, fps, params_pp, n_valid_frames=vl[u]))
                    for u in sc}
                res = evaluate(preds, va_refs)
                line += "  %s: F1=%.4f dice=%.4f score=%.4f" % (
                    name, res["event_f1"], res["segment_dice"], res["score"])
                if res["score"] > best:
                    best = res["score"]
                    torch.save({"model": m.state_dict(), "cfg": cfg,
                                "classes": le.classes, "which": name, "epoch": ep,
                                "score": best}, out_dir / "best.pt")
                    # Uncompressed: these are ~10 MB of float32 and zlib on a
                    # 2-vCPU Colab host costs more than the evaluation did.
                    np.savez(out_dir / "val_scores.npz", **{u: sc[u] for u in sc})
                    (out_dir / "val_meta.json").write_text(json.dumps(
                        {"valid": vl, "refs": {u: va_refs[u] for u in sc},
                         "classes": le.classes, "fps": fps}), encoding="utf-8")
                history.append({"epoch": ep, "which": name, **res})
        if rank == 0:
            print(line)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            torch.save({"model": student.state_dict(), "teacher": teacher.state_dict(),
                        "cfg": cfg, "classes": le.classes, "epoch": ep}, out_dir / "last.pt")

    if rank == 0:
        if best < 0:
            # No timestamped clips to validate against (e.g. a bronze-only batch of
            # the corpus). Still emit best.pt so inference has something to load.
            torch.save({"model": student.state_dict(), "cfg": cfg, "classes": le.classes,
                        "which": "student", "epoch": epochs, "score": None},
                       out_dir / "best.pt")
            print("[train] no validation set - saved final student as best.pt")
        else:
            print("[train] best val score: %.4f  ->  %s" % (best, out_dir / "best.pt"))

    if is_distributed:
        dist.barrier()  # let rank 0 finish writing before anyone tears the group down
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
