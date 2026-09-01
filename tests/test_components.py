"""Verify csebbs / metrics / tuner against synthetic score curves with known truth."""
import pathlib, sys, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.postproc.csebbs import csebbs_single, median_filter_decode, union_events, default_params_for
from src.evaluation.metrics import evaluate, segment_dice, match_events, _intersection
from src.postproc.tune import tune

FPS = 25.0
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m) or (0 if c else fails.append(m))
fails = []

# ---------- metrics ----------
ok(abs(segment_dice([(0,1)], [(0,1)]) - 1.0) < 1e-9, "dice identical = 1")
ok(abs(segment_dice([(0,1)], [(1,2)]) - 0.0) < 1e-9, "dice disjoint = 0")
ok(abs(segment_dice([(0,2)], [(1,3)]) - 0.5) < 1e-9, "dice half overlap = 0.5")
ok(abs(segment_dice([], []) - 1.0) < 1e-9, "dice empty/empty = 1")
ok(abs(_intersection([(0,1),(2,3)], [(0.5,2.5)]) - 1.0) < 1e-9, "intersection multi-span")

# collar: ref 0..1 (dur 1) -> collar 0.2
ok(match_events([(0.1,1.1)], [(0.0,1.0)], pct=0.2) == 1, "event within 20% collar matches")
ok(match_events([(0.3,1.3)], [(0.0,1.0)], pct=0.2) == 0, "event outside collar rejected")
# short event: ref 0..0.1 -> collar 0.02
ok(match_events([(0.05,0.15)], [(0.0,0.1)], pct=0.2) == 0, "short event collar is tight")
ok(match_events([(0.01,0.11)], [(0.0,0.1)], pct=0.2) == 1, "short event tight match ok")
# one prediction cannot satisfy two refs
ok(match_events([(0.0,1.0)], [(0.0,1.0),(0.0,1.0)], pct=0.2) == 1, "1-to-1 matching enforced")

r = evaluate({"a": [(0.0,1.0)]}, {"a": [(0.0,1.0)]})
ok(r["event_f1"] == 1.0 and r["segment_dice"] == 1.0 and r["score"] == 1.0, "perfect eval = 1.0")
r = evaluate({"a": []}, {"a": [(0.0,1.0)]})
ok(r["event_f1"] == 0.0 and r["recall"] == 0.0, "empty pred = 0 recall")

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
ok(rep["score"] > 0.5, "tuner reaches a sane score on easy data (%.3f)" % rep["score"])

print("\n%d failures" % len(fails))
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
