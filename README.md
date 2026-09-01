# Vaani Noise Event Detection — IndoML Datathon Track 1

Sound event detection on Indic speech recordings: detect noise events and their
onset/offset timestamps.

**Competition:** [Datathon@IndoML 2026 — Track 1](https://www.codabench.org/competitions/17825/)
on Codabench. Scored on **Event F1 + Segment Dice** (max 2.0) over an 11 h withheld
test set; submit a ZIP holding one `predictions.jsonl`.

**Architecture: frozen BEATs encoder + FDY-CRNN backbone, trained with mean-teacher
semi-supervision on a per-tier masked loss, with frequency MixStyle for domain shift
and cSEBBs for post-processing.**

This is the configuration DCASE 2024 Task 4 converged on, and that challenge posed
nearly the same problem: one model, multiple label qualities, missing labels, real
domain shift.

<p align="center"><a href="notebooks/Vaani_Track1_Colab.ipynb"><b>▶ Open the Colab notebook</b></a></p>

> ### 📦 Dataset: use the full corpus, not the sample
> The training corpus is **[ARTPARK-IISc/Vaani-Noise-Event-Dataset](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset)**
> — 182 shards, 16.5 GB, **90,637 clips / ~154.6 h**, and it is **gated** (accept the
> terms, then supply a token; in Colab store it as the secret `HF_TOKEN`).
>
> `PavanKumarJ-ARTPARK/Vaani_Noise_Event_TimeStamp` is a 9-clip *sample* of it. Both work,
> but only the full repo can train anything.

---

## Quick start (Colab)

1. New Colab notebook → `Runtime → Change runtime type → T4 GPU`
2. Open `notebooks/Vaani_Track1_Colab.ipynb` (File → Open notebook → GitHub → this repo)
3. Run all cells.

Or from a terminal:

```bash
git clone https://github.com/raut7218/vaani-sed-track1.git && cd vaani-sed-track1
pip install -r requirements.txt
python scripts/download_data.py --out data/vaani --max-shards 2   # ~1000 clips
python scripts/download_data.py --out data/vaani                  # full 154.6 h
python -m src.train.train --config configs/default.yaml --data data/vaani --out runs/baseline
python scripts/tune_postproc.py --run runs/baseline
python -m src.infer.predict --ckpt runs/baseline/best.pt --audio-dir data/test \
    --params runs/baseline/postproc_params.json --out submission.zip
```

Upload `submission.zip` to
[Codabench competition 17825](https://www.codabench.org/competitions/17825/).

### Getting the data

The corpus is gated. Accept the terms on the
[dataset page](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset),
create a read token, then expose it — in Colab as the secret `HF_TOKEN` (Secrets → enable
notebook access), elsewhere as `export HF_TOKEN=hf_...`. `download_data.py` finds it
automatically, in that order, falling back to a cached `huggingface-cli login`.

```bash
python scripts/download_data.py --out data/vaani --list-only    # what's on the server
python scripts/download_data.py --out data/vaani --max-shards 2 # ~1000 clips, minutes
python scripts/download_data.py --out data/vaani                # all 182 shards
```

| | |
|---|---|
| On the server | 182 shards, 16.5 GB parquet, 90,637 clips (~154.6 h) |
| Decoded | ~9 GB as FLAC (default), ~17 GB as `--format wav` |
| Peak disk | ≈ decoded size — each parquet blob is deleted once materialised (`--keep-parquet` to retain) |
| Resumable | Yes, at clip granularity. Re-run after a disconnect. |

~9 GB fits Colab's local disk comfortably but **not** a free 15 GB Drive, so keep the
audio on local disk. `--max-shards` / `--shard-start` page through the corpus if you want
to grow the training set gradually.

Verify the whole pipeline first — synthetic audio, no download, no GPU, ~1 minute:

```bash
python scripts/smoke_test.py     # end-to-end: train -> tune -> submission.zip
python tests/test_components.py  # 25 checks on metrics, cSEBBs decoding, tuner
python tests/test_overfit.py     # proves frame targets are time-aligned
python tests/test_amp_loss.py    # training step survives autocast (AMP)
python tests/test_schema.py      # annotationQuality + string timestamps
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

## Tier assignment

The full corpus carries an **`annotationQuality`** column, so tiers come from the data
rather than a guess. (The 9-clip sample had no such field, which is why earlier revisions
of this README treated it as an open question.)

`prepare.py` resolves it case- and separator-insensitively, so `annotationQuality`,
`annotation_quality` and `AnnotationQuality` all resolve, and maps generously —
`gold`/`verified`/`tier1`/`high` → gold, and so on (`QUALITY_ALIASES` in
[`src/data/prepare.py`](src/data/prepare.py)).

Two rules worth knowing:

* **Anything unmapped is reported, never guessed.** The download ends with the exact
  values it saw and flags any it could not place, so you add them to `QUALITY_ALIASES`
  and re-run — materialised clips are skipped, so that costs seconds.
* **A gold/silver clip with no timestamps is demoted to bronze.** Without timestamps
  there is nothing for the frame-level loss to consume, whatever the column claims.

Overrides remain available: `--gold-ids ids.txt`, or `--default-ts-tier gold` for the
fallback when the column is missing entirely.

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

## Training throughput (T4)

A T4 is a **Turing GPU**, not a TPU: it has fp16 tensor cores but *no* bf16 and no TF32.
fp16 autocast + `GradScaler` is therefore the right and only choice, and the log-mel
front-end stays pinned to float32 (fp16's smallest subnormal sits above the `1e-10` clamp
floor, so `log(0) = -inf` and the batch goes NaN — see `LogMel.forward`).

The step used to be dominated by work that was not buying anything:

| Fix | What it was doing | Where |
|---|---|---|
| **Share the BEATs pass** | The encoder is frozen and student and teacher see the identical waveform through *the same module object* — so it was computing two bit-identical tensors per step, for roughly a third of the wall clock. | `encode_beats` in [`sed_model.py`](src/models/sed_model.py) |
| **`cudnn.benchmark = True`** | The window is a fixed size, so every step has identical tensor shapes. The autotuner pays for itself in the first few steps. | [`train.py`](src/train/train.py) |
| **No per-step syncs** | The loss dict called `float()` on four live CUDA tensors *inside* every step, and `torch.isfinite(loss)` added a fifth. Each drains the CUDA queue. Logs are now accumulated on-device and read once per epoch. | [`losses.py`](src/train/losses.py) |
| **Fused EMA** | The mean-teacher update walked `named_parameters()` and issued two kernel launches per tensor — a few hundred per step, which a 2-vCPU Colab host cannot feed. Now two `_foreach` calls against a pairing built once at startup. | `build_ema_pairs` |
| **Teacher pinned to eval** | An `eval()`/`train()` round trip per step recursed through every submodule, BEATs' 12 transformer layers included. The teacher only ever runs in eval anyway. | `train.py` |
| **Dataloader** | 4 workers (from 2) with prefetch, and the val loader keeps its workers alive instead of respawning them every epoch. | `configs/default.yaml` |

Together these are worth roughly **2× end-to-end**, and none of them changes what the model
learns — the loss, the sampling and the update rule are all bit-for-bit what they were.

**Not yet done, in rough order of remaining value:** caching the frozen BEATs features to
disk (removes the encoder from the training loop entirely, ~5 GB for the full corpus);
trimming the 10 s window, which is ~39% zero padding at a 6.1 s corpus mean; `channels_last`
for the conv stack; and moving `fdy_from_block` to 3 so the two largest-spatial blocks use
plain convs. The last two change what the model learns and need an ablation before you
trust them.

Raising the batch size is **not** the lever it looks like. At batch 24 the step is
compute-bound, so doubling the batch roughly doubles the step time at the same throughput;
what it actually buys is amortising per-step overhead. The idle VRAM is mostly BEATs'
`(B·heads, 496, 496)` relative-position-bias and attention matrices, which the feature cache
above removes outright.

---

## Metrics

The competition publishes its scoring code, so
[`src/evaluation/metrics.py`](src/evaluation/metrics.py) is a **faithful reimplementation
of it** rather than an interpretation:

| | |
|---|---|
| **Event-based F1** | Greedy closest-first 1-to-1 matching. A prediction matches when its onset *and* offset are both within `max(0.20 × ref_duration, 0.05)` s. Micro-averaged. |
| **Segment Dice** | Events rasterised to a 10 ms grid with an inclusive offset frame; `2·|P ∩ G| / (|P| + |G|)`, **macro-averaged** across clips. Empty-vs-empty scores 1.0. |
| **Combined** | `F1 + Dice`, maximum **2.0**. This is the leaderboard rank and the tuning objective. |

Three details that are easy to get wrong and each cost real score:

* **The tolerance floor is 50 ms, not zero.** Earlier revisions of this repo tuned against
  a strict 20% collar, which is materially tighter for short events — a 0.1 s
  `human_non_speech` event gets 50 ms of slack, not 20 ms.
* **Dice is macro-averaged**, so a short clip counts as much as a long one. Optimising the
  micro (duration-weighted) version quietly favours the long stationary classes.
* **The score is a sum, not a mean.** A run scoring 1.42 is not 142%.

`tests/test_components.py` asserts these against hand-worked values, and the scorer's own
published snippets were used as a differential reference while porting.

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

`predict.py` writes a **ZIP holding a single `predictions.jsonl` at its root** — exactly
what Codabench expects. One JSON object per line, one line per clip:

```jsonl
{"clip_id": "vaani_eval_001", "events": [{"onset": 1.24, "offset": 3.81}, {"onset": 5.1, "offset": 6.55}]}
{"clip_id": "vaani_eval_002", "events": []}
```

Rules the writer enforces for you:

* **Every evaluation clip appears exactly once**, with `[]` when nothing is detected — a
  missing line is not the same as an empty one. If a clip fails to decode, `predict.py`
  logs it and emits an empty record rather than aborting the run and losing the rest.
* `predictions.jsonl` sits at the archive **root**, not inside a folder.
* Times are milliseconds-rounded, non-negative, and non-decreasing. This is not cosmetic:
  the scorer rasterises with `int(onset / 0.01)`, so a negative onset would index the frame
  mask from the wrong end.
* Clips you predict that are *not* in the evaluation set count their events as false
  positives, so do not pad the file with extras.

Track 1 is scored on class-agnostic boundaries, so per-class detections are unioned at the
end. Training multi-class and unioning beats training a single "any noise" head — the class
structure is what makes the frame representation learnable.

`scripts/smoke_test.py` validates a real archive against every rule above.

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
src/infer/predict.py         -> submission.zip (predictions.jsonl)
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

## Credits

BEATs © Microsoft, MIT-licensed, vendored in `third_party/beats/`.
Checkpoint mirrored from `lpepino/beats_ckpts`.
Dataset: [Vaani Noise Event Timestamps](https://huggingface.co/datasets/PavanKumarJ-ARTPARK/Vaani_Noise_Event_TimeStamp)
(CC-BY-4.0), derived from Project Vaani (IISc Bangalore / ARTPARK).
