"""Can the model overfit a tiny gold-only set? If not, targets are misaligned."""
import pathlib, sys, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.labels import LabelEncoder
from src.data.dataset import VaaniSED, collate
from src.models.sed_model import VaaniSEDModel
from src.postproc.csebbs import csebbs_single
from src.evaluation.metrics import evaluate

import soundfile as sf, json, pathlib, shutil
SR, CLIP, FPS = 16000, 4.0, 25.0
TMP = pathlib.Path(__file__).resolve().parent / "_tmp_overfit"
if TMP.exists(): shutil.rmtree(TMP)
(TMP/"audio").mkdir(parents=True)

rng = np.random.RandomState(0)
recs, truth = [], {}
for i in range(8):
    y = (rng.randn(int(CLIP*SR))*0.01).astype("float32")
    s = 0.5 + 0.35*i; e = s + 1.0
    a_, b_ = int(s*SR), int(e*SR)
    t = np.arange(b_-a_)/SR
    y[a_:b_] += (0.5*np.sin(2*np.pi*1000*t)).astype("float32")
    uid = "o%d" % i
    sf.write(str(TMP/"audio"/(uid+".wav")), y, SR)
    recs.append({"uid": uid, "path": "audio/%s.wav" % uid, "duration": CLIP, "tier": "gold",
                 "state": "S", "district": "D", "language": "L",
                 "events": [{"cls": "animal_sound", "start": round(s,3), "end": round(e,3), "tag": ""}],
                 "clip_labels": ["animal_sound"]})
    truth[uid] = [(s, e)]

le = LabelEncoder(True)
ds = VaaniSED(recs, TMP, le, CLIP, SR, FPS, train=False, augment=False)
batch = collate([ds[i] for i in range(len(ds))])

# sanity: does the frame target line up with where we injected the tone?
ft = batch["frame_target"].numpy()
ci = le.idx["animal_sound"]
for i in range(3):
    act = np.where(ft[i][:, ci] > 0.5)[0]
    print("clip %d target frames %.2f-%.2fs (injected %.2f-%.2f)" %
          (i, act[0]/FPS, (act[-1]+1)/FPS, 0.5+0.35*i, 0.5+0.35*i+1.0))

model = VaaniSEDModel(n_class=len(le), n_frames=int(CLIP*FPS), beats=None,
                      rnn_dim=64, rnn_layers=1, dropout=0.0, n_basis=4,
                      mixstyle_p=0.0, use_specaug=False)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
model.train()
for step in range(400):
    logit, clip = model(batch["wav"], tier=None, frame_valid=batch["frame_valid"])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, batch["frame_target"])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0 or step == 399:
        print("step %3d  frame_bce=%.4f" % (step, loss.item()))

model.eval()
with torch.no_grad():
    logit, _ = model(batch["wav"], tier=None, frame_valid=batch["frame_valid"])
    p = torch.sigmoid(logit).numpy()

preds = {}
for i, uid in enumerate(batch["uid"]):
    ev = csebbs_single(p[i][:, ci], FPS, detect_thr=0.5, step_len_s=0.16,
                       change_thr=0.05, medfilt_s=0.08, merge_gap_s=0.08)
    preds[uid] = ev
print("\nexample pred", preds[batch["uid"][0]], "truth", truth[batch["uid"][0]])
res = evaluate(preds, truth)
print("overfit eval:", res)
assert loss.item() < 0.05, "FAIL: could not drive frame loss down (%.4f)" % loss.item()
assert res["event_f1"] > 0.85, "FAIL: overfit F1 too low (%.3f) -> target misalignment" % res["event_f1"]
print("\nPASS: model overfits gold set; frame targets are time-aligned.")
