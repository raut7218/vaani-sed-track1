"""Download the Vaani dataset straight from Hugging Face and materialise it.

Pulls **every** parquet shard in the repo with `snapshot_download` (so new
batches are picked up automatically as they are published), then decodes each
row to an audio file and appends to `manifest.jsonl`.

    python scripts/download_data.py --out /content/work/data

Both stages are resumable: the HF cache skips shards already fetched, and clips
already in the manifest are skipped. Re-run it whenever new batches land.

Why not `datasets.load_dataset`? This path shows you exactly which files exist on
the server and how big they are, works without the `datasets` library, streams
each shard in row batches so memory stays flat, and never silently falls back to
a cached subset.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import LabelEncoder  # noqa: E402
from src.data.prepare import build_record  # noqa: E402

REPO = "PavanKumarJ-ARTPARK/Vaani_Noise_Event_TimeStamp"

# Announced target from the dataset card, for the coverage report.
TARGET_HOURS = 167.47


def list_remote(repo: str, token: str | None) -> list[dict]:
    """What the server actually holds, before downloading anything."""
    from huggingface_hub import HfApi
    api = HfApi(token=token or None)
    info = api.repo_info(repo_id=repo, repo_type="dataset", files_metadata=True)
    out = []
    for s in info.siblings:
        out.append({"path": s.rfilename,
                    "size": getattr(s, "size", None) or getattr(s, "lfs", None) and s.lfs.size})
    return out


def human(n: int | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f GB" % n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", required=True, help="where to write audio/ + manifest.jsonl")
    ap.add_argument("--cache", default="", help="HF cache dir (default: HF's own)")
    ap.add_argument("--token", default="", help="HF token (raises rate limits)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--format", default="wav", choices=["wav", "flac"],
                    help="flac is lossless and roughly half the size - use it if "
                         "you are writing to Google Drive")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N clips")
    ap.add_argument("--gold-ids", default="")
    ap.add_argument("--default-ts-tier", default="silver", choices=["gold", "silver"])
    ap.add_argument("--expand-vehicle", type=int, default=1)
    ap.add_argument("--list-only", action="store_true",
                    help="show what is on the server and exit, downloading nothing")
    ap.add_argument("--batch-rows", type=int, default=256,
                    help="parquet rows held in memory at once")
    args = ap.parse_args()

    token = args.token or None

    # ---- 1. what is actually published -------------------------------------
    print("[download] repo: %s" % args.repo)
    try:
        remote = list_remote(args.repo, token)
    except Exception as e:  # noqa: BLE001
        print("[download] could not list repo (%s); continuing to snapshot" % e)
        remote = []
    shards = [f for f in remote if f["path"].endswith(".parquet")]
    for f in remote:
        print("   %-45s %s" % (f["path"], human(f["size"])))
    total_bytes = sum(f["size"] or 0 for f in shards)
    print("[download] %d parquet shard(s), %s total" % (len(shards), human(total_bytes)))
    if args.list_only:
        return

    if len(shards) == 1 and "-of-00001" in shards[0]["path"]:
        print("\n[download] NOTE: the shard is named '-of-00001', i.e. the dataset "
              "declares\n            exactly one shard. Everything published is "
              "being downloaded;\n            if that is far short of the ~%.0f h "
              "advertised, the rest has\n            not been uploaded yet.\n" % TARGET_HOURS)

    # ---- 2. fetch every shard ----------------------------------------------
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id=args.repo, repo_type="dataset", token=token,
        cache_dir=args.cache or None,
        allow_patterns=["*.parquet", "*.json", "README.md"],
        max_workers=4,
    )
    local = Path(local)
    files = sorted(local.rglob("*.parquet"))
    print("[download] snapshot at %s (%d parquet files)" % (local, len(files)))
    if not files:
        raise SystemExit("no parquet files found in the snapshot")

    # ---- 3. materialise -----------------------------------------------------
    import pyarrow.parquet as pq
    import soundfile as sf

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.jsonl"

    done = set()
    if man_path.exists():
        with man_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:  # noqa: BLE001
                    pass
        print("[download] resuming: %d clips already materialised" % len(done))

    gold_ids = set()
    if args.gold_ids and Path(args.gold_ids).exists():
        gold_ids = {ln.strip() for ln in Path(args.gold_ids).read_text().splitlines() if ln.strip()}
        print("[download] %d gold ids loaded" % len(gold_ids))

    le = LabelEncoder(bool(args.expand_vehicle))
    ext = args.format
    tiers, cls_count, unknown = Counter(), Counter(), Counter()
    dur_by_tier = defaultdict(float)
    n_written = n_skipped = n_failed = 0

    with man_path.open("a", encoding="utf-8") as fout:
        for fi, pf in enumerate(files):
            print("[download] shard %d/%d: %s" % (fi + 1, len(files), pf.name))
            table = pq.ParquetFile(str(pf))
            row_i = 0
            for batch in table.iter_batches(batch_size=args.batch_rows):
                for row in batch.to_pylist():
                    if args.limit and n_written >= args.limit:
                        break
                    audio = row.get("audio") or {}
                    path = audio.get("path") or ""
                    stem = Path(str(path)).stem
                    uid = stem if stem else "%s_%07d" % (pf.stem, row_i)
                    row_i += 1
                    if uid in done:
                        n_skipped += 1
                        continue

                    wav_path = out / "audio" / ("%s.%s" % (uid, ext))
                    try:
                        if audio.get("bytes"):
                            import librosa
                            y, _ = librosa.load(io.BytesIO(audio["bytes"]), sr=args.sr,
                                                mono=True)
                        elif audio.get("array") is not None:
                            import numpy as np
                            y = np.asarray(audio["array"], dtype="float32")
                        else:
                            n_failed += 1
                            continue
                        sf.write(str(wav_path), y, args.sr)
                    except Exception as e:  # noqa: BLE001
                        print("[download]   failed %s: %s" % (uid, e))
                        n_failed += 1
                        continue

                    duration = float(len(y)) / args.sr
                    rec = build_record(row, uid, duration, gold_ids,
                                       args.default_ts_tier, bool(args.expand_vehicle),
                                       unknown)
                    rec["path"] = "audio/%s.%s" % (uid, ext)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    n_written += 1
                    tiers[rec["tier"]] += 1
                    dur_by_tier[rec["tier"]] += duration
                    for ev in rec["events"]:
                        cls_count[ev["cls"]] += 1
                    if n_written % 500 == 0:
                        fout.flush()
                        print("[download]   %d clips (%.2f h)" % (
                            n_written, sum(dur_by_tier.values()) / 3600))
                if args.limit and n_written >= args.limit:
                    break

    # ---- 4. report ----------------------------------------------------------
    total_h = sum(dur_by_tier.values()) / 3600
    prev_h = 0.0
    if done:  # account for clips written by an earlier run
        with man_path.open(encoding="utf-8") as f:
            prev_h = sum(json.loads(l).get("duration", 0.0) for l in f if l.strip()) / 3600
        total_h = prev_h

    stats = {
        "new_clips": n_written, "skipped_existing": n_skipped, "failed": n_failed,
        "total_clips_in_manifest": len(done) + n_written,
        "hours_total": round(total_h, 3),
        "tiers_this_run": dict(tiers),
        "hours_by_tier_this_run": {k: round(v / 3600, 3) for k, v in dur_by_tier.items()},
        "events_per_class_this_run": dict(cls_count),
        "unknown_category_strings": dict(unknown),
        "classes": le.classes,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n" + json.dumps(stats, indent=2))

    pct = 100.0 * total_h / TARGET_HOURS if TARGET_HOURS else 0.0
    print("\n[download] coverage: %.3f h of the ~%.0f h advertised (%.2f%%)"
          % (total_h, TARGET_HOURS, pct))
    if pct < 50:
        print("[download] The remainder is not on the server yet - this is a dataset\n"
              "           availability issue, not a download problem. Re-run this\n"
              "           script when new batches are published; it resumes.")
    if unknown:
        print("[download] WARNING: unmapped category strings above; add them to "
              "ALIASES in src/data/labels.py")


if __name__ == "__main__":
    main()
