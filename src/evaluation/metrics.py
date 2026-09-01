"""Track 1 metrics: event-based F1 (+/-20% collar) and segment Dice.

The challenge page defines:
  * event-based F1 - a prediction is correct when it aligns with a reference
    event within "+/-20% of event duration";
  * segment Dice   - 2*|P intersect G| / (|P| + |G|) over time;
  * final rank     - the two weighted equally.

The exact collar convention is not fully pinned down on the challenge page, so
both readings are implemented: `collar_mode="pct"` uses a pure 20%-of-duration
tolerance, `collar_mode="sed_eval"` uses max(0.2 s, 20% of duration), which is
the standard DCASE convention. Tune with the stricter one; it is the safer
assumption.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Event = Tuple[float, float]


def _total_len(spans: Sequence[Event]) -> float:
    return float(sum(max(0.0, b - a) for a, b in spans))


def _merge(spans: Sequence[Event]) -> List[Event]:
    if not spans:
        return []
    s = sorted(spans)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _intersection(p: Sequence[Event], g: Sequence[Event]) -> float:
    p, g = _merge(p), _merge(g)
    i = j = 0
    tot = 0.0
    while i < len(p) and j < len(g):
        lo = max(p[i][0], g[j][0])
        hi = min(p[i][1], g[j][1])
        if hi > lo:
            tot += hi - lo
        if p[i][1] < g[j][1]:
            i += 1
        else:
            j += 1
    return tot


def segment_dice(pred: Sequence[Event], ref: Sequence[Event]) -> float:
    """2|P n G| / (|P| + |G|). Defined as 1.0 when both are empty."""
    inter = _intersection(pred, ref)
    denom = _total_len(pred) + _total_len(ref)
    if denom <= 0:
        return 1.0
    return 2.0 * inter / denom


def match_events(pred: Sequence[Event], ref: Sequence[Event], pct: float = 0.2,
                 min_collar: float = 0.0) -> int:
    """Greedy 1-to-1 matching; returns the number of true positives.

    A pair matches when both onset and offset fall within the collar, where the
    collar scales with the *reference* event's duration.
    """
    used_pred = set()
    tp = 0
    # Match the tightest (shortest-tolerance) references first so a generous
    # long-event collar cannot steal a prediction a short event needed.
    order = sorted(range(len(ref)), key=lambda i: ref[i][1] - ref[i][0])
    for gi in order:
        gs, ge = ref[gi]
        collar = max(min_collar, pct * (ge - gs))
        best, best_err = -1, None
        for pi, (ps, pe) in enumerate(pred):
            if pi in used_pred:
                continue
            if abs(ps - gs) <= collar and abs(pe - ge) <= collar:
                err = abs(ps - gs) + abs(pe - ge)
                if best_err is None or err < best_err:
                    best, best_err = pi, err
        if best >= 0:
            used_pred.add(best)
            tp += 1
    return tp


def evaluate(preds: Dict[str, List[Event]], refs: Dict[str, List[Event]],
             collar_mode: str = "pct", pct: float = 0.2) -> dict:
    """Corpus-level metrics over {uid: [(on, off), ...]}."""
    min_collar = 0.2 if collar_mode == "sed_eval" else 0.0

    tp = n_pred = n_ref = 0
    inter = len_p = len_g = 0.0
    dice_per_clip: List[float] = []

    for uid, ref in refs.items():
        pred = preds.get(uid, [])
        ref_m, pred_m = _merge(ref), _merge(pred)
        tp += match_events(pred_m, ref_m, pct=pct, min_collar=min_collar)
        n_pred += len(pred_m)
        n_ref += len(ref_m)
        inter += _intersection(pred_m, ref_m)
        len_p += _total_len(pred_m)
        len_g += _total_len(ref_m)
        dice_per_clip.append(segment_dice(pred_m, ref_m))

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    dice_micro = 2 * inter / (len_p + len_g) if (len_p + len_g) else 0.0
    dice_macro = sum(dice_per_clip) / len(dice_per_clip) if dice_per_clip else 0.0

    return {
        "event_f1": round(f1, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "segment_dice": round(dice_micro, 5),
        "segment_dice_macro": round(dice_macro, 5),
        # Equal weighting, matching the stated ranking rule.
        "score": round(0.5 * f1 + 0.5 * dice_micro, 5),
        "n_pred": n_pred, "n_ref": n_ref, "tp": tp,
    }


def evaluate_per_class(preds: Dict[str, Dict[str, List[Event]]],
                       refs: Dict[str, Dict[str, List[Event]]],
                       classes: Sequence[str], **kw) -> Dict[str, dict]:
    """Same metrics, computed independently per class - use this to find which
    categories are dragging the score down before touching the architecture."""
    out = {}
    for c in classes:
        p = {u: v.get(c, []) for u, v in preds.items()}
        g = {u: v.get(c, []) for u, v in refs.items()}
        if any(g.values()):
            out[c] = evaluate(p, g, **kw)
    return out
