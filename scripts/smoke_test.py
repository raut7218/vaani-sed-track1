"""End-to-end smoke test on synthetic audio - no download, no GPU, ~1 minute.

Builds a fake corpus with all three tiers, trains for 2 epochs, tunes the
post-processor and writes a submission. Run this before burning Colab GPU time:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SR = 16000
TMP = ROOT / "_smoke"


def make_corpus(n: int = 72) -> None:
    import soundfile as sf
    rng = np.random.RandomState(0)
    (TMP / "data" / "audio").mkdir(parents=True, exist_ok=True)
    classes = ["animal_sound", "vehicle_traffic", "human_non_speech", "singing_music"]
    lines = []
    for i in range(n):
        dur = float(rng.uniform(2.0, 8.0))
        y = (rng.randn(int(dur * SR)) * 0.02).astype("float32")
        events = []
        for _ in range(rng.randint(1, 3)):
            s = float(rng.uniform(0, max(0.1, dur - 0.6)))
            e = min(dur, s + float(rng.uniform(0.15, 1.5)))
            cls = classes[rng.randint(len(classes))]
            # Give each class a distinguishable tone so the model has real signal.
            f = {"animal_sound": 900, "vehicle_traffic": 300,
                 "human_non_speech": 2500, "singing_music": 1500}[cls]
            a, b = int(s * SR), int(e * SR)
            t = np.arange(b - a) / SR
            y[a:b] += (0.3 * np.sin(2 * np.pi * f * t)).astype("float32")
            events.append({"cls": cls, "start": round(s, 3), "end": round(e, 3), "tag": ""})
        uid = "smoke_%03d" % i
        sf.write(str(TMP / "data" / "audio" / (uid + ".wav")), y, SR)
        tier = ["gold", "silver", "bronze"][i % 3]
        rec = {
            "uid": uid, "path": "audio/%s.wav" % uid, "duration": round(len(y) / SR, 3),
            "tier": tier, "state": "S%d" % (i % 4), "district": "D", "language": "L",
            "events": [] if tier == "bronze" else events,
            "clip_labels": sorted({e["cls"] for e in events}),
        }
        lines.append(json.dumps(rec))
    (TMP / "data" / "manifest.jsonl").write_text("\n".join(lines), encoding="utf-8")
    print("[smoke] built %d clips" % n)


def write_cfg() -> Path:
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["data"]["root"] = str(TMP / "data")
    cfg["data"]["clip_len"] = 4.0
    cfg["data"]["val_frac"] = 0.25
    cfg["output_dir"] = str(TMP / "run")
    cfg["model"]["use_beats"] = False          # keep the smoke test offline
    cfg["model"]["rnn_dim"] = 64
    cfg["model"]["rnn_layers"] = 1
    cfg["train"].update(epochs=2, batch_size=6, num_workers=0, warmup_steps=5, amp=False)
    cfg["loss"]["cons_rampup_steps"] = 10
    p = TMP / "smoke.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def check_submission(zip_path: Path, n_clips: int) -> None:
    """Validate the archive against the competition's stated format."""
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert names == ["predictions.jsonl"], \
            "archive must hold exactly predictions.jsonl at its root, got %s" % names
        lines = z.read("predictions.jsonl").decode("utf-8").strip().split("\n")

    assert len(lines) == n_clips, "expected %d lines, got %d" % (n_clips, len(lines))
    seen, n_ev = set(), 0
    for line in lines:
        rec = json.loads(line)
        assert set(rec) == {"clip_id", "events"}, "bad record keys: %s" % sorted(rec)
        assert isinstance(rec["clip_id"], str) and rec["clip_id"]
        assert rec["clip_id"] not in seen, "duplicate clip_id %s" % rec["clip_id"]
        seen.add(rec["clip_id"])
        assert isinstance(rec["events"], list)
        for ev in rec["events"]:
            assert set(ev) == {"onset", "offset"}, "bad event keys: %s" % sorted(ev)
            assert isinstance(ev["onset"], float) and isinstance(ev["offset"], float)
            assert 0.0 <= ev["onset"] <= ev["offset"], "bad span %s" % ev
        n_ev += len(rec["events"])
    print("\n[smoke] PASS - %d clips, %d events, submission format valid" % (len(seen), n_ev))


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit("FAILED: %s" % " ".join(cmd))


def main() -> None:
    if TMP.exists():
        shutil.rmtree(TMP)
    make_corpus()
    cfg_path = write_cfg()
    py = sys.executable

    run([py, "-m", "src.train.train", "--config", str(cfg_path)])
    run([py, "scripts/tune_postproc.py", "--run", str(TMP / "run"), "--rounds", "1"])
    run([py, "-m", "src.infer.predict",
         "--ckpt", str(TMP / "run" / "best.pt"),
         "--manifest", str(TMP / "data" / "manifest.jsonl"),
         "--params", str(TMP / "run" / "postproc_params.json"),
         "--out", str(TMP / "submission.zip")])

    check_submission(TMP / "submission.zip", n_clips=72)
    print("[smoke] artefacts in %s" % TMP)


if __name__ == "__main__":
    main()
