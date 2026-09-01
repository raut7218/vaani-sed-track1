"""Post-processing: change-detection Sound Event Bounding Boxes (cSEBBs).

Frame-wise thresholding decides **whether each frame is active**, which fragments
long events into bursts and merges nearby short ones. cSEBBs instead decides
**where the boundaries are** (via change detection on the score curve) and only
then scores the resulting box. In DCASE 2024 this swap gained ~+4.1 PSDS1 points
averaged over 13 independent systems with zero retraining.

This is a reimplementation from the method description, not the authors' code.
It is parameterised per class and `tune_csebbs` fits those parameters directly
against the Track 1 metric on validation data - which is where the gain actually
comes from, so do not skip the tuning step.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Event = Tuple[float, float]


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(w, dtype=np.float64) / w
    return np.convolve(xp, k, mode="same")[pad:pad + len(x)]


def _median_filter(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i + w])
    return out


def change_curve(scores: np.ndarray, step_len: int) -> np.ndarray:
    """Difference of the trailing and leading moving averages.

    Positive peaks mark onsets (score rising), negative peaks mark offsets.
    """
    step_len = max(1, int(step_len))
    fwd = _moving_average(scores, step_len)
    # shift by step_len to compare the window after vs the window before
    after = np.concatenate([fwd[step_len:], np.repeat(fwd[-1], min(step_len, len(fwd)))])
    after = after[:len(fwd)]
    before = np.concatenate([np.repeat(fwd[0], min(step_len, len(fwd))), fwd[:-step_len]])
    before = before[:len(fwd)]
    return after - before


def _local_extrema(c: np.ndarray, want_max: bool, thr: float,
                   tol: float = 1e-9) -> List[int]:
    """Local extrema of the change curve, resolved to the *centre* of a plateau.

    A step edge in the score curve produces a flat-topped plateau in the change
    curve (its width is set by the filter length). Taking the first or last index
    of that plateau biases every boundary by half the filter length - fatal here,
    because the collar for a 0.5 s event is only 0.1 s. Take the midpoint.
    """
    s = c if want_max else -c
    n = len(s)
    out: List[int] = []
    i = 1
    while i < n - 1:
        if s[i] >= thr and s[i] > s[i - 1] + tol:
            j = i
            while j + 1 < n and abs(s[j + 1] - s[i]) <= tol:
                j += 1
            if j + 1 < n and s[j + 1] < s[i] - tol:
                out.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return out


def csebbs_single(scores: np.ndarray, fps: float, detect_thr: float = 0.5,
                  step_len_s: float = 0.32, change_thr: float = 0.08,
                  medfilt_s: float = 0.0, merge_gap_s: float = 0.12,
                  min_dur_s: float = 0.04, max_dur_s: float = 0.0,
                  boundary_pad_s: float = 0.0) -> List[Event]:
    """One class, one clip. Returns [(onset_s, offset_s), ...]."""
    n = len(scores)
    if n == 0:
        return []
    y = scores.astype(np.float64)
    if medfilt_s > 0:
        y = _median_filter(y, max(1, int(round(medfilt_s * fps))))

    step = max(1, int(round(step_len_s * fps)))
    c = change_curve(y, step)

    onsets = _local_extrema(c, True, change_thr)
    offsets = _local_extrema(c, False, change_thr)

    # Boundaries always include the clip edges so events that start at t=0 or
    # run to the end of the clip are not lost (common in Vaani - many clips are
    # almost entirely covered by one event).
    bounds = sorted(set([0] + onsets + offsets + [n]))

    # Score each inter-boundary segment and keep the active ones.
    segs: List[Tuple[int, int]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        if float(y[a:b].mean()) >= detect_thr:
            segs.append((a, b))

    if not segs:
        return []

    merge_gap = int(round(merge_gap_s * fps))
    merged: List[List[int]] = [list(segs[0])]
    for a, b in segs[1:]:
        if a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    pad = boundary_pad_s
    out: List[Event] = []
    for a, b in merged:
        on, off = a / fps - pad, b / fps + pad
        on = max(0.0, on)
        off = min(n / fps, off)
        dur = off - on
        if dur < min_dur_s:
            continue
        if max_dur_s and dur > max_dur_s:
            off = on + max_dur_s
        out.append((round(float(on), 3), round(float(off), 3)))
    return out


def median_filter_decode(scores: np.ndarray, fps: float, thr: float = 0.5,
                         medfilt_s: float = 0.16, min_dur_s: float = 0.04) -> List[Event]:
    """Classic frame-threshold baseline, kept for the ablation."""
    y = _median_filter(scores.astype(np.float64), max(1, int(round(medfilt_s * fps))))
    act = y >= thr
    out, start = [], None
    for i, a in enumerate(act):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if (i - start) / fps >= min_dur_s:
                out.append((round(start / fps, 3), round(i / fps, 3)))
            start = None
    if start is not None and (len(act) - start) / fps >= min_dur_s:
        out.append((round(start / fps, 3), round(len(act) / fps, 3)))
    return out


# Short, impulsive classes need a much shorter change filter than stationary ones.
# `human_non_speech` (coughs, breaths, lip smacks: 0.05-0.5 s) is the known floor
# class - give it its own narrow window or it gets smeared into its neighbours.
DEFAULT_PARAMS: Dict[str, dict] = {
    "human_non_speech": dict(step_len_s=0.08, medfilt_s=0.04, merge_gap_s=0.04,
                             min_dur_s=0.03),
    "phone_signal_alarm": dict(step_len_s=0.16, medfilt_s=0.08, merge_gap_s=0.08),
    "vehicle_horn": dict(step_len_s=0.16, medfilt_s=0.08, merge_gap_s=0.08),
    "baby_child_noise": dict(step_len_s=0.24, medfilt_s=0.12, merge_gap_s=0.12),
    "animal_sound": dict(step_len_s=0.24, medfilt_s=0.12, merge_gap_s=0.12),
    "vehicle_engine": dict(step_len_s=0.48, medfilt_s=0.24, merge_gap_s=0.24),
    "appliance_machine": dict(step_len_s=0.48, medfilt_s=0.24, merge_gap_s=0.24),
    "singing_music": dict(step_len_s=0.48, medfilt_s=0.24, merge_gap_s=0.24),
}


def default_params_for(classes: Sequence[str]) -> Dict[str, dict]:
    base = dict(detect_thr=0.5, step_len_s=0.32, change_thr=0.08, medfilt_s=0.12,
                merge_gap_s=0.12, min_dur_s=0.04, boundary_pad_s=0.0)
    out = {}
    for c in classes:
        p = dict(base)
        p.update(DEFAULT_PARAMS.get(c, {}))
        out[c] = p
    return out


def decode_clip(scores: np.ndarray, classes: Sequence[str], fps: float,
                params: Dict[str, dict], method: str = "csebbs",
                n_valid_frames: int | None = None) -> Dict[str, List[Event]]:
    """scores: (T, C) -> {class: [(on, off), ...]}"""
    if n_valid_frames is not None:
        scores = scores[:max(1, n_valid_frames)]
    out: Dict[str, List[Event]] = {}
    for ci, c in enumerate(classes):
        p = params.get(c, {})
        if method == "csebbs":
            out[c] = csebbs_single(scores[:, ci], fps, **p)
        else:
            out[c] = median_filter_decode(
                scores[:, ci], fps, thr=p.get("detect_thr", 0.5),
                medfilt_s=p.get("medfilt_s", 0.16), min_dur_s=p.get("min_dur_s", 0.04))
    return out


def union_events(per_class: Dict[str, List[Event]], merge_gap_s: float = 0.05,
                 min_dur_s: float = 0.03) -> List[Event]:
    """Collapse per-class detections into class-agnostic onset/offset pairs.

    Track 1 is scored on noise-event boundaries without class, so the submission
    is the union over classes. Training multi-class and unioning here beats
    training a single 'any noise' head: the class structure is what makes the
    frame representation learnable.
    """
    spans = sorted([e for evs in per_class.values() for e in evs])
    if not spans:
        return []
    merged = [list(spans[0])]
    for on, off in spans[1:]:
        if on - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], off)
        else:
            merged.append([on, off])
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= min_dur_s]
