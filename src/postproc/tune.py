"""Fit cSEBBs parameters against the Track 1 metric by coordinate ascent.

Per-class parameters are tuned one class at a time while the objective stays the
*class-agnostic union* score that Track 1 actually ranks on. Independent
per-class tuning would over-fit classes whose events are already covered by
another class's detections in the union.

Cheap and worth it: this runs on cached validation scores, so no retraining.
"""
from __future__ import annotations

import json
from typing import Dict, Sequence

import numpy as np

from src.evaluation.metrics import evaluate
from src.postproc.csebbs import decode_clip, default_params_for, union_events

# Kept deliberately small - a wider grid mostly buys noise on a small val split.
GRID = {
    "detect_thr": [0.3, 0.4, 0.5, 0.6],
    "step_len_s": [0.08, 0.16, 0.32, 0.48],
    "medfilt_s": [0.0, 0.04, 0.12, 0.24],
    "merge_gap_s": [0.04, 0.12, 0.24],
}


def _decode_all(scores: Dict[str, np.ndarray], valid: Dict[str, int],
                classes: Sequence[str], fps: float, params: Dict[str, dict],
                method: str, union_gap: float) -> Dict[str, list]:
    out = {}
    for uid, s in scores.items():
        per_cls = decode_clip(s, classes, fps, params, method=method,
                              n_valid_frames=valid.get(uid))
        out[uid] = union_events(per_cls, merge_gap_s=union_gap)
    return out


def tune(scores: Dict[str, np.ndarray], refs: Dict[str, list], classes: Sequence[str],
         fps: float, valid: Dict[str, int] | None = None, method: str = "csebbs",
         rounds: int = 2, union_gap: float = 0.05,
         verbose: bool = True) -> tuple[Dict[str, dict], dict]:
    valid = valid or {}
    params = default_params_for(classes)

    def score_of(p: Dict[str, dict]) -> float:
        preds = _decode_all(scores, valid, classes, fps, p, method, union_gap)
        return evaluate(preds, refs)["score"]

    best = score_of(params)
    if verbose:
        print("[tune] start score=%.4f" % best)

    for r in range(rounds):
        improved = False
        for c in classes:
            for key, values in GRID.items():
                cur = params[c].get(key)
                for v in values:
                    if v == cur:
                        continue
                    trial = {k: dict(p) for k, p in params.items()}
                    trial[c][key] = v
                    s = score_of(trial)
                    if s > best + 1e-6:
                        best, params, cur, improved = s, trial, v, True
                        if verbose:
                            print("[tune] round %d %s.%s=%s -> %.4f" % (r + 1, c, key, v, best))
        if not improved:
            break

    # A final sweep on the union merge gap, which is a global knob.
    for g in [0.0, 0.05, 0.1, 0.2]:
        preds = _decode_all(scores, valid, classes, fps, params, method, g)
        s = evaluate(preds, refs)["score"]
        if s > best + 1e-6:
            best, union_gap = s, g
            if verbose:
                print("[tune] union_gap=%s -> %.4f" % (g, best))

    preds = _decode_all(scores, valid, classes, fps, params, method, union_gap)
    report = evaluate(preds, refs)
    report["union_gap"] = union_gap
    if verbose:
        print("[tune] final %s" % json.dumps(report))
    return params, report
