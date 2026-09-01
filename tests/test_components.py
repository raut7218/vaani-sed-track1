"""Verify csebbs / metrics / tuner against synthetic score curves with known truth."""
import pathlib, sys, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.postproc.csebbs import (csebbs_single, median_filter_decode, union_events,
                                 default_params_for, _median_filter)
from src.evaluation.metrics import evaluate, clip_dice, match_events
from src.postproc.tune import tune

FPS = 25.0
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m) or (0 if c else fails.append(m))
fails = []

# ---------- metrics (must mirror the official Codabench scorer) ----------
# Dice rasterises to 10 ms frames with an *inclusive* offset frame, so exact
# values carry that one-frame widening - these are the scorer's numbers, not
# the continuous-overlap ones.
ok(abs(clip_dice([(0,1)], [(0,1)]) - 1.0) < 1e-9, "dice identical = 1")
ok(abs(clip_dice([(0,1)], [(1,2)]) - 2/202) < 1e-9, "dice disjoint = only the shared edge frame")
ok(abs(clip_dice([(0,2)], [(1,3)]) - 2*101/402) < 1e-9, "dice half overlap")
ok(abs(clip_dice([], []) - 1.0) < 1e-9, "dice empty/empty = 1")
ok(abs(clip_dice([(0,1)], []) - 0.0) < 1e-9, "dice pred-only = 0")

# match_events(ref, pred) -> (tp, fp, fn). Tolerance = max(0.2 * ref_dur, 0.05).
tp = lambda ref, pred: match_events(ref, pred)[0]
# ref 0..1 (dur 1) -> tol 0.2
ok(tp([(0.0,1.0)], [(0.1,1.1)]) == 1, "event within 20% tolerance matches")
ok(tp([(0.0,1.0)], [(0.3,1.3)]) == 0, "event outside tolerance rejected")
# ref 0..0.1 (dur 0.1) -> 20% is 0.02, but the floor lifts it to 0.05
ok(tp([(0.0,0.1)], [(0.05,0.15)]) == 1, "50 ms floor applies to short events")
ok(tp([(0.0,0.1)], [(0.06,0.16)]) == 0, "beyond the 50 ms floor is rejected")
# one prediction cannot satisfy two refs
ok(match_events([(0.0,1.0),(0.0,1.0)], [(0.0,1.0)]) == (1, 0, 1), "1-to-1 matching enforced")
# closest-first: the exact prediction must claim the exact reference
ok(match_events([(0.0,1.0),(0.1,1.1)], [(0.1,1.1),(0.0,1.0)]) == (2, 0, 0),
   "greedy closest-first pairs both events")

r = evaluate({"a": [(0.0,1.0)]}, {"a": [(0.0,1.0)]})
ok(r["event_f1"] == 1.0 and r["segment_dice"] == 1.0 and r["score"] == 2.0,
   "perfect eval scores 2.0 (F1 + Dice, not their mean)")
r = evaluate({"a": []}, {"a": [(0.0,1.0)]})
ok(r["event_f1"] == 0.0 and r["recall"] == 0.0, "empty pred = 0 recall")
# clips predicted but absent from the reference are pure false positives
r = evaluate({"a": [(0.0,1.0)], "ghost": [(0.0,1.0)]}, {"a": [(0.0,1.0)]})
ok(r["fp"] == 1 and r["event_f1"] < 1.0, "extra clip costs precision")

# ---------- median filter: vectorised form must equal the naive loop ----------
def _median_ref(x, w):
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i + w]) for i in range(len(x))])

_mf_rng = np.random.RandomState(3)
_mf_bad = 0
for _n in (1, 2, 7, 50, 251):
    _x = _mf_rng.rand(_n)
    for _w in range(1, 12):
        if not np.allclose(_median_filter(_x, _w), _median_ref(_x, _w)):
            _mf_bad += 1
ok(_mf_bad == 0, "vectorised median filter matches the naive loop (odd and even w)")

# ---------- csebbs decoding ----------
def curve(spans, n=250, hi=0.95, lo=0.05, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    y = np.full(n, lo)
    for a, b in spans:
        y[int(a*FPS):int(b*FPS)] = hi
    if noise:
        y = np.clip(y + rng.randn(n)*noise, 0, 1)
    return y

# single clean event 2.0 - 5.0 s
ev = csebbs_single(curve([(2.0,5.0)]), FPS, detect_thr=0.5, step_len_s=0.32,
                   change_thr=0.08, medfilt_s=0.12, merge_gap_s=0.12)
ok(len(ev) == 1, "clean single event -> 1 box (got %s)" % ev)
if ev:
    ok(abs(ev[0][0]-2.0) < 0.3 and abs(ev[0][1]-5.0) < 0.3, "boundaries near truth %s" % (ev,))

# two well-separated events
ev = csebbs_single(curve([(1.0,2.0),(5.0,6.5)]), FPS, detect_thr=0.5, step_len_s=0.16,
                   change_thr=0.05, medfilt_s=0.08, merge_gap_s=0.08)
ok(len(ev) == 2, "two separated events -> 2 boxes (got %s)" % ev)

# event running to the very end of the clip (common in Vaani)
ev = csebbs_single(curve([(8.0,10.0)]), FPS, detect_thr=0.5, step_len_s=0.32,
                   change_thr=0.08, medfilt_s=0.12)
ok(len(ev) == 1 and ev[0][1] > 9.5, "event at clip end kept (got %s)" % ev)

# event starting at t=0
ev = csebbs_single(curve([(0.0,2.0)]), FPS, detect_thr=0.5, step_len_s=0.32,
                   change_thr=0.08, medfilt_s=0.12)
ok(len(ev) == 1 and ev[0][0] < 0.4, "event at t=0 kept (got %s)" % ev)

# all-silent -> nothing
ok(csebbs_single(curve([]), FPS) == [], "silent curve -> no events")
# all-active -> one box
ev = csebbs_single(curve([(0.0,10.0)]), FPS)
ok(len(ev) == 1, "fully active -> 1 box (got %s)" % ev)

# noisy curve: csebbs should not shatter into many fragments
noisy = curve([(2.0,5.0)], noise=0.22, seed=3)
ev_c = csebbs_single(noisy, FPS, detect_thr=0.5, step_len_s=0.32, change_thr=0.08, medfilt_s=0.12)
ev_m = median_filter_decode(noisy, FPS, thr=0.5, medfilt_s=0.04)
print("   noisy: csebbs=%d boxes %s | median=%d boxes" % (len(ev_c), ev_c, len(ev_m)))
ok(len(ev_c) <= len(ev_m), "csebbs no more fragmented than median filter")

# ---------- union ----------
u = union_events({"a": [(0.0,1.0)], "b": [(0.95,2.0)]}, merge_gap_s=0.05)
ok(u == [(0.0,2.0)], "overlapping classes union (got %s)" % u)
u = union_events({"a": [(0.0,1.0)], "b": [(3.0,4.0)]}, merge_gap_s=0.05)
ok(len(u) == 2, "disjoint classes stay separate")

# ---------- tuner actually improves a mis-set default ----------
classes = ["c0"]
scores, refs = {}, {}
rng = np.random.RandomState(7)
for i in range(30):
    a = rng.uniform(1.0, 6.0); b = a + rng.uniform(0.8, 2.5)
    scores["u%d" % i] = curve([(a, b)], noise=0.15, seed=i)[:, None].astype(np.float32)
    refs["u%d" % i] = [(a, b)]
params, rep = tune(scores, refs, classes, FPS, rounds=2, verbose=False)
base = evaluate({u: union_events({"c0": csebbs_single(scores[u][:,0], FPS,
                 **default_params_for(classes)["c0"])}) for u in scores}, refs)
print("   tuner: baseline score=%.4f -> tuned=%.4f" % (base["score"], rep["score"]))
ok(rep["score"] >= base["score"], "tuning never degrades")
ok(rep["score"] > 1.0, "tuner reaches a sane score on easy data (%.3f/2.0)" % rep["score"])

print("\n%d failures" % len(fails))
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
