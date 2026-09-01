"""Tune cSEBBs parameters on cached validation scores (no retraining).

    python scripts/tune_postproc.py --run runs/baseline

Writes <run>/postproc_params.json, which src/infer/predict.py reads via --params.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate  # noqa: E402
from src.postproc.csebbs import decode_clip, default_params_for, union_events  # noqa: E402
from src.postproc.tune import tune  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", default="csebbs", choices=["csebbs", "median"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--collar-mode", default="pct", choices=["pct", "sed_eval"])
    args = ap.parse_args()

    run = Path(args.run)
    npz = np.load(run / "val_scores.npz")
    meta = json.loads((run / "val_meta.json").read_text(encoding="utf-8"))
    classes, fps = meta["classes"], float(meta["fps"])
    scores = {u: npz[u] for u in npz.files}
    refs = {u: [tuple(x) for x in v] for u, v in meta["refs"].items()}
    valid = {u: int(v) for u, v in meta["valid"].items()}
    print("[tune] %d val clips, %d classes, fps=%s" % (len(scores), len(classes), fps))

    # Baseline for comparison: plain median-filter thresholding, untuned.
    base_params = default_params_for(classes)
    base_preds = {u: union_events(decode_clip(scores[u], classes, fps, base_params,
                                              method="median",
                                              n_valid_frames=valid.get(u)))
                  for u in scores}
    base = evaluate(base_preds, refs, collar_mode=args.collar_mode)
    print("[tune] median-filter baseline: %s" % json.dumps(base))

    params, report = tune(scores, refs, classes, fps, valid=valid, method=args.method,
                          rounds=args.rounds, collar_mode=args.collar_mode)

    out = run / "postproc_params.json"
    out.write_text(json.dumps({"params": params, "union_gap": report.get("union_gap", 0.05),
                               "method": args.method, "report": report,
                               "baseline_median": base}, indent=2), encoding="utf-8")
    print("[tune] wrote %s" % out)
    print("[tune] baseline score %.4f -> tuned %.4f (+%.4f)" % (
        base["score"], report["score"], report["score"] - base["score"]))


if __name__ == "__main__":
    main()
