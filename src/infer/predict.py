"""Inference -> submission JSON.

    python -m src.infer.predict --ckpt runs/baseline/best.pt \
        --audio-dir data/test --out submission.json

Track 1 wants class-agnostic onset/offset pairs per clip, so per-class detections
are unioned at the end. Long files are processed in overlapping windows and
stitched, since the model is trained on a fixed window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.labels import LabelEncoder
from src.models.beats_encoder import build_beats, download_beats
from src.models.sed_model import VaaniSEDModel
from src.postproc.csebbs import decode_clip, default_params_for, union_events

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def load_model(ckpt_path: str, device: torch.device, use_beats: bool | None = None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    classes = ck["classes"]
    le = LabelEncoder(cfg["data"]["expand_vehicle"])
    le.classes, le.idx = classes, {c: i for i, c in enumerate(classes)}

    want_beats = cfg["model"].get("use_beats", True) if use_beats is None else use_beats
    beats = None
    if want_beats:
        p = cfg["model"].get("beats_ckpt") or ""
        if not p or not Path(p).exists():
            got = download_beats(cfg["model"].get("beats_dir", "checkpoints"))
            p = str(got) if got else ""
        beats = build_beats(p if p else None, True)

    fps = float(cfg["data"]["fps"])
    n_frames = int(round(float(cfg["data"]["clip_len"]) * fps))
    model = VaaniSEDModel(
        n_class=len(classes), n_frames=n_frames, beats=beats,
        n_mels=cfg["data"]["n_mels"], sr=cfg["data"]["sr"], hop=cfg["data"]["hop"],
        rnn_dim=cfg["model"]["rnn_dim"], rnn_layers=cfg["model"]["rnn_layers"],
        dropout=cfg["model"]["dropout"], n_basis=cfg["model"]["n_basis"],
        mixstyle_p=0.0, mixstyle_alpha=cfg["model"]["mixstyle_alpha"],
        use_specaug=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print("[predict] missing keys: %d" % len(missing))
    model.to(device).eval()
    return model, cfg, le, fps


@torch.no_grad()
def score_file(model, wav: np.ndarray, sr: int, clip_len: float, fps: float,
               device: torch.device, hop_frac: float = 0.5, amp: bool = True) -> np.ndarray:
    """Windowed scoring with overlap-add averaging -> (T_total, C)."""
    win = int(round(clip_len * sr))
    total_frames = max(1, int(round(len(wav) / sr * fps)))
    n_cls = model.n_class
    acc = np.zeros((total_frames, n_cls), dtype=np.float64)
    cnt = np.zeros((total_frames, 1), dtype=np.float64)

    step = max(1, int(win * hop_frac))
    starts = list(range(0, max(1, len(wav) - win + 1), step))
    if not starts or starts[-1] + win < len(wav):
        starts.append(max(0, len(wav) - win))

    for s in starts:
        chunk = wav[s:s + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        x = torch.from_numpy(chunk[None]).float().to(device)
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            logit, _ = model(x, tier=None, frame_valid=None)
        p = torch.sigmoid(logit.float())[0].cpu().numpy()      # (n_frames, C)

        f0 = int(round(s / sr * fps))
        n = min(p.shape[0], total_frames - f0)
        if n <= 0:
            continue
        acc[f0:f0 + n] += p[:n]
        cnt[f0:f0 + n] += 1

    cnt[cnt == 0] = 1.0
    return (acc / cnt).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio-dir", default="", help="directory of test audio files")
    ap.add_argument("--manifest", default="", help="alternatively, a manifest.jsonl")
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--params", default="", help="tuned cSEBBs params json")
    ap.add_argument("--method", default="csebbs", choices=["csebbs", "median"])
    ap.add_argument("--union-gap", type=float, default=0.05)
    ap.add_argument("--per-class-out", default="", help="also dump per-class events")
    ap.add_argument("--save-scores", default="", help="npz of raw frame scores")
    args = ap.parse_args()

    import soundfile as sf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, le, fps = load_model(args.ckpt, device)
    sr = int(cfg["data"]["sr"])
    clip_len = float(cfg["data"]["clip_len"])

    params = default_params_for(le.classes)
    if args.params and Path(args.params).exists():
        loaded = json.loads(Path(args.params).read_text(encoding="utf-8"))
        if "params" in loaded:
            args.union_gap = loaded.get("union_gap", args.union_gap)
            loaded = loaded["params"]
        for c, p in loaded.items():
            if c in params:
                params[c].update(p)
        print("[predict] loaded tuned params for %d classes" % len(loaded))

    files = []
    if args.manifest:
        root = Path(args.manifest).parent
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                files.append((r["uid"], root / r["path"]))
    else:
        for p in sorted(Path(args.audio_dir).rglob("*")):
            if p.suffix.lower() in AUDIO_EXT:
                files.append((p.stem, p))
    if not files:
        raise SystemExit("no audio found - pass --audio-dir or --manifest")
    print("[predict] %d files" % len(files))

    submission, per_class_out, raw = {}, {}, {}
    for i, (uid, path) in enumerate(files):
        y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            import librosa
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        scores = score_file(model, y, sr, clip_len, fps, device)
        if args.save_scores:
            raw[uid] = scores
        per_cls = decode_clip(scores, le.classes, fps, params, method=args.method)
        ev = union_events(per_cls, merge_gap_s=args.union_gap)
        submission[uid] = [{"onset": round(a, 3), "offset": round(b, 3)} for a, b in ev]
        per_class_out[uid] = {c: [{"onset": a, "offset": b} for a, b in v]
                              for c, v in per_cls.items() if v}
        if (i + 1) % 200 == 0:
            print("[predict] %d/%d" % (i + 1, len(files)))

    Path(args.out).write_text(json.dumps(submission, indent=2), encoding="utf-8")
    n_ev = sum(len(v) for v in submission.values())
    print("[predict] wrote %s: %d clips, %d events" % (args.out, len(submission), n_ev))
    if args.per_class_out:
        Path(args.per_class_out).write_text(json.dumps(per_class_out, indent=2),
                                            encoding="utf-8")
    if args.save_scores:
        np.savez_compressed(args.save_scores, **raw)


if __name__ == "__main__":
    main()
