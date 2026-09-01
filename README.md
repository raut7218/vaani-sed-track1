# Vaani Noise Event Detection — IndoML Datathon Track 1

Sound event detection on Indic speech recordings: detect noise events and their
onset/offset timestamps.

**Architecture: frozen BEATs encoder + FDY-CRNN backbone, trained with mean-teacher
semi-supervision on a per-tier masked loss, with frequency MixStyle for domain shift
and cSEBBs for post-processing.**

This is the configuration DCASE 2024 Task 4 converged on, and that challenge posed
nearly the same problem: one model, multiple label qualities, missing labels, real
domain shift.

<p align="center"><a href="notebooks/Vaani_Track1_Colab.ipynb"><b>▶ Open the Colab notebook</b></a></p>

---

## Quick start (Colab)

1. New Colab notebook → `Runtime → Change runtime type → T4 GPU`
2. Open `notebooks/Vaani_Track1_Colab.ipynb` (File → Open notebook → GitHub → this repo)
3. Run all cells.

Or from a terminal:

```bash
git clone https://github.com/raut7218/vaani-sed-track1.git && cd vaani-sed-track1
pip install -r requirements.txt
python -m src.data.prepare --out data/vaani --streaming
python -m src.train.train --config configs/default.yaml --data data/vaani --out runs/baseline
python scripts/tune_postproc.py --run runs/baseline
python -m src.infer.predict --ckpt runs/baseline/best.pt --audio-dir data/test \
    --params runs/baseline/postproc_params.json --out submission.json
```

Verify the whole pipeline first — synthetic audio, no download, no GPU, ~1 minute:

```bash
python scripts/smoke_test.py     # end-to-end: train -> tune -> submission.json
python tests/test_components.py  # 25 checks on metrics, cSEBBs decoding, tuner
python tests/test_overfit.py     # proves frame targets are time-aligned
```

`test_overfit.py` is the one that catches the nastiest class of bug: it drives frame BCE
to ~2e-4 on a tiny gold set and confirms the decoded events land on the injected tones
(F1 = 1.0). If you change the feature front-end, the CNN pooling, or the target builder,
run it — a silent time-offset between audio and targets will not show up in the loss.

---

## Why each stage

| Stage | Why |
|---|---|
| **Frozen BEATs + CNN features** | The single biggest lever in the literature: 0.353 → 0.497 PSDS1 in a controlled test. Frozen — fine-tuning 90M params on ~20 h of verified data overfits and costs GPU time you don't have. |
| **FDY-CRNN backbone** | Frequency-dynamic conv beats a plain CRNN by ~6.3%. Verify it *per category*: it helps horns/barks/coughs and can hurt fans/engines. |
| **Student + EMA teacher** | Mean-teacher (decay 0.999) extracts a training signal from clips with no timestamps by enforcing student/teacher consistency. |
| **Masked per-tier loss** | The actual trick for Track 1 — see below. |
| **Frequency MixStyle** | Vaani's domain shift (states, languages, devices) is worse than anything this technique was validated on. Applied within tier, never across. |
| **cSEBBs** | Cheapest large win available: +4.1 PSDS1 averaged across 13 independent systems, **zero retraining**. Don't skip it to save time. |

**Deliberately not used:** wav2vec2 / HuBERT / WavLM. Benchmarked head-to-head, all three
scored *at or below* a no-pretraining baseline on SED — they are trained to discard exactly
the non-speech sounds this task is about.

### The masked per-tier loss

A clip contributes only the loss terms its annotation quality can support:

| Tier | Data | Frame loss | Clip loss |
|---|---|---|---|
| 🥇 gold | ~21.8 h, verified timestamps | BCE, weight 1.0 | BCE |
| 🥈 silver | ~100.3 h, unverified timestamps | BCE, weight 0.5 | BCE |
| 🥉 bronze | ~32.4 h, tags only | **masked out** | BCE via attention pooling |

Training only on the verified 22 h throws away ~85% of the corpus. Training on all of it
as if it were verified teaches the model that silver's looser boundaries are ground truth.
One model, one loop, three masks — never three separate models.

Implemented in [`src/train/losses.py`](src/train/losses.py); tier quotas per batch in
[`src/data/dataset.py`](src/data/dataset.py).

---

## ⚠️ Tier assignment — the one open question

**The released parquet has no tier column.** The schema is `audio, imageFileName, state,
district, duration, language, isTranscriptionAvailable, transcript, NoiseCategory,
NoiseSubCategoryTimeStamp` — nothing distinguishing verified from unverified.

`prepare.py` therefore resolves tiers like this:

* **bronze** — no `NoiseSubCategoryTimeStamp` entries. This *is* the bronze definition, so
  it is exact.
* **gold vs silver** — not derivable from the current schema. Timestamped clips default to
  **silver**, the conservative choice (silver carries the lower strong-loss weight, so the
  model doesn't over-trust boundaries that may be unverified).

Three ways to fix this the moment better information exists, no code changes needed:

```bash
# 1. organisers publish gold ids
python -m src.data.prepare --out data/vaani --gold-ids gold_ids.txt

# 2. you decide timestamped == gold
python -m src.data.prepare --out data/vaani --default-ts-tier gold

# 3. a tier column appears in the dataset -> picked up automatically
#    (checks tier, quality, annotation_quality, verified, num_annotators, ...)
```

Since gold and silver differ *only* by `loss.strong_weight` in the config, you can also
sweep that weight directly instead of guessing the split.

---

## Build order

Designed so you are always shippable.

| # | Step | Status |
|---|---|---|
| 1 | FDY-CRNN + frozen BEATs + mean teacher + masked loss | ✅ implemented |
| 2 | cSEBBs post-processing, tuned per category | ✅ `scripts/tune_postproc.py` |
| 3 | Frequency MixStyle (per tier) | ✅ `model.mixstyle_p` |
| 4 | Self-train on the unverified tier | 🔜 see below |
| 5 | Seed ensembling, then **re-tune cSEBBs on the ensemble** | 🔜 |

**Self-training (step 4)** unlocks the bronze hours you haven't used for frame supervision:
run `predict.py --manifest` over bronze clips with `--save-scores`, keep predictions above a
confidence threshold *that are consistent with the clip's known tags*, write them into a new
manifest as silver, retrain. The tag constraint is what makes this safe — you already know
which classes are present, you're only inferring where.

---

## Metrics

Track 1 ranks on two metrics, equally weighted:

* **Event-based F1** — correct when a prediction aligns within ±20% of event duration
* **Segment Dice** — `2|P ∩ G| / (|P| + |G|)`

Both are in [`src/evaluation/metrics.py`](src/evaluation/metrics.py), with
`score = 0.5·F1 + 0.5·Dice` as the tuning objective.

The exact collar convention isn't fully pinned down on the challenge page, so both readings
are implemented and switchable via `eval.collar_mode`:

* `pct` (default) — strict 20% of event duration
* `sed_eval` — `max(0.2 s, 20%)`, the standard DCASE convention

Tune with `pct`; it's the stricter, safer assumption.

---

## Categories

Seven top-level categories. **The strings in the parquet differ from the dataset card**
(`animal_sound` not `animal`, `baby_child_noise` not `baby_child`), so
[`src/data/labels.py`](src/data/labels.py) is alias-driven and `prepare.py` reports any
category string it could not map instead of silently dropping it.

`vehicle_traffic` is split into **`vehicle_horn`** and **`vehicle_engine`** before training
(`data.expand_vehicle`): it straddles both the FDY-helps (impulsive/harmonic) and FDY-hurts
(stationary broadband) regimes, and one class can't be tuned for both. They're merged back
for the class-agnostic submission.

**Watch `human_non_speech`** — coughs, breaths, lip smacks, 0.05–0.5 s. It behaves like
DESED's worst-performing class in every system in the literature and will likely be your
accuracy floor. It gets its own short cSEBBs filter window by default
(`step_len_s=0.08` vs `0.48` for engines). If it dominates your error budget, raise
`data.fps` from 25 to 50 for 20 ms frame resolution.

---

## Submission format

```json
{
  "clip_id_1": [{"onset": 1.24, "offset": 3.81}, {"onset": 4.31, "offset": 4.71}],
  "clip_id_2": [{"onset": 0.05, "offset": 2.10}]
}
```

Track 1 is scored on class-agnostic boundaries, so per-class detections are unioned at the
end. Training multi-class and unioning beats training a single "any noise" head — the class
structure is what makes the frame representation learnable. Per-class detections are still
available via `--per-class-out` for error analysis.

> Confirm the exact top-level container with the organisers before the final submission —
> the challenge page shows the per-event objects but not the wrapper.

---

## Layout

```
configs/default.yaml         all hyperparameters
notebooks/                   Colab notebook (start here)
scripts/smoke_test.py        end-to-end test on synthetic audio
scripts/tune_postproc.py     fit cSEBBs params against the metric
src/data/labels.py           alias-driven label space, vehicle subtype split
src/data/prepare.py          HF -> wavs + manifest + tier assignment
src/data/dataset.py          per-tier masks, tier-balanced batch sampler
src/models/beats_encoder.py  frozen BEATs wrapper (degrades to mel-only)
src/models/fdy_crnn.py       frequency-dynamic conv, CRNN, attention pooling
src/models/mixstyle.py       frequency MixStyle, within-tier permutation
src/models/sed_model.py      the single model
src/train/losses.py          masked per-tier loss + consistency
src/train/train.py           training loop
src/postproc/csebbs.py       change-detection SEBBs
src/evaluation/metrics.py    event F1 + segment Dice
src/infer/predict.py         -> submission.json
third_party/beats/           BEATs source, vendored from microsoft/unilm (MIT)
```

## Ablations

Each is one flag:

| Flag | Tests |
|---|---|
| `--no-beats` | what BEATs is worth on Vaani specifically |
| `model.n_basis: 1` | FDY vs plain CRNN (**check per class**) |
| `loss.lambda_cons: 0` | value of mean-teacher |
| `model.mixstyle_p: 0` | value of frequency MixStyle |
| `--method median` | cSEBBs vs frame thresholding |

Use `evaluate_per_class` in `src/evaluation/metrics.py` — the aggregate score hides which
categories a change actually moved.

## Credits

BEATs © Microsoft, MIT-licensed, vendored in `third_party/beats/`.
Checkpoint mirrored from `lpepino/beats_ckpts`.
Dataset: [Vaani Noise Event Timestamps](https://huggingface.co/datasets/PavanKumarJ-ARTPARK/Vaani_Noise_Event_TimeStamp)
(CC-BY-4.0), derived from Project Vaani (IISc Bangalore / ARTPARK).
