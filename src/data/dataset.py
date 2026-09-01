"""Dataset, tier-balanced batch sampler and collation.

One dataset serves all three annotation tiers. Every item carries the masks that
tell the loss which terms it is allowed to contribute to, so a single model and a
single training loop cover gold / silver / bronze without branching.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data.labels import LabelEncoder

TIER_IDS = {"gold": 0, "silver": 1, "bronze": 2}


def read_manifest(path: str | Path) -> List[dict]:
    recs = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def split_manifest(recs: List[dict], val_frac: float = 0.1, seed: int = 1234,
                   group_by: str = "state") -> tuple[List[dict], List[dict]]:
    """Hold out a validation set.

    Validation is drawn only from timestamped clips (bronze has no timestamps to
    score against). We group by `state` where possible so the val set measures
    generalisation across the domain shift the challenge actually cares about,
    rather than across clips from the same recording conditions.
    """
    ts = [r for r in recs if r.get("events")]
    bronze = [r for r in recs if not r.get("events")]
    if not ts:
        return recs, []

    rng = random.Random(seed)
    groups: Dict[str, List[dict]] = {}
    for r in ts:
        groups.setdefault(str(r.get(group_by, "")) or "_", []).append(r)

    keys = sorted(groups)
    rng.shuffle(keys)
    target = max(1, int(len(ts) * val_frac))
    val: List[dict] = []
    # Whole-group holdout, unless a single group would swallow the whole split
    # (which happens when the uploaded batch covers only one state).
    for k in keys:
        if len(val) >= target:
            break
        if len(groups[k]) <= max(target, 1) * 2:
            val.extend(groups[k])
    if not val or len(val) > len(ts) * 0.5:
        shuffled = list(ts)
        rng.shuffle(shuffled)
        val = shuffled[:target]

    val_uids = {r["uid"] for r in val}
    train = [r for r in ts if r["uid"] not in val_uids] + bronze
    return train, val


class VaaniSED(Dataset):
    def __init__(self, records: Sequence[dict], root: str | Path, le: LabelEncoder,
                 clip_len: float = 10.0, sr: int = 16000, fps: float = 25.0,
                 train: bool = True, augment: bool = True):
        self.recs = list(records)
        self.root = Path(root)
        self.le = le
        self.clip_len = float(clip_len)
        self.sr = int(sr)
        self.fps = float(fps)
        self.train = train
        self.augment = augment and train
        self.n_samples = int(round(self.clip_len * self.sr))
        self.n_frames = int(round(self.clip_len * self.fps))

    def __len__(self) -> int:
        return len(self.recs)

    def _load_wav(self, rec: dict) -> np.ndarray:
        import soundfile as sf
        y, sr = sf.read(str(self.root / rec["path"]), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != self.sr:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        return y.astype("float32")

    def __getitem__(self, i: int) -> dict:
        rec = self.recs[i]
        y = self._load_wav(rec)
        events = rec.get("events") or []

        # --- crop / pad to a fixed window, shifting event times with the crop ---
        offset = 0
        if len(y) > self.n_samples:
            offset = random.randint(0, len(y) - self.n_samples) if self.train \
                else (len(y) - self.n_samples) // 2
            y = y[offset:offset + self.n_samples]
        valid_samples = len(y)
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)))
        t_off = offset / self.sr

        if self.augment:
            y = y * float(np.random.uniform(0.85, 1.15))          # gain jitter
            if random.random() < 0.5:
                y = y + np.random.randn(len(y)).astype("float32") * 1e-3  # mild noise

        C, F = len(self.le), self.n_frames
        frame_t = np.zeros((F, C), dtype="float32")
        for ev in events:
            ci = self.le.idx.get(ev["cls"])
            if ci is None:
                continue
            s = (float(ev["start"]) - t_off) * self.fps
            e = (float(ev["end"]) - t_off) * self.fps
            a, b = int(math.floor(max(0.0, s))), int(math.ceil(min(float(F), e)))
            if b > a:
                frame_t[a:b, ci] = 1.0
            elif 0 <= a < F and e > s:
                frame_t[a, ci] = 1.0  # event shorter than one frame -> keep 1 frame

        clip_t = np.zeros((C,), dtype="float32")
        if events:
            clip_t = frame_t.max(axis=0)
        else:
            for ci in self.le.encode_clip_categories(rec.get("clip_labels")):
                clip_t[ci] = 1.0

        tier = rec.get("tier", "bronze")
        n_valid = int(min(F, math.ceil(valid_samples / self.sr * self.fps)))
        frame_valid = np.zeros((F,), dtype="float32")
        frame_valid[:max(1, n_valid)] = 1.0

        return {
            "wav": torch.from_numpy(y),
            "frame_target": torch.from_numpy(frame_t),
            "clip_target": torch.from_numpy(clip_t),
            "frame_valid": torch.from_numpy(frame_valid),
            "strong_mask": torch.tensor(0.0 if tier == "bronze" else 1.0),
            "tier": torch.tensor(TIER_IDS.get(tier, 2), dtype=torch.long),
            "uid": rec["uid"],
        }


def collate(batch: List[dict]) -> dict:
    out = {}
    for k in batch[0]:
        if k == "uid":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


class TierBatchSampler(Sampler):
    """Compose every batch from fixed per-tier quotas.

    Mean-teacher consistency and per-tier MixStyle both need each tier present in
    each step; a plain random sampler would give batches with no bronze (or no
    gold) once the tiers are as imbalanced as they are here.
    """

    def __init__(self, records: Sequence[dict], batch_size: int,
                 quotas: Dict[str, float] | None = None, seed: int = 0,
                 drop_last: bool = True):
        self.records = list(records)
        self.batch_size = int(batch_size)
        self.seed = seed
        self.drop_last = drop_last
        self.by_tier: Dict[str, List[int]] = {"gold": [], "silver": [], "bronze": []}
        for i, r in enumerate(self.records):
            self.by_tier.setdefault(r.get("tier", "bronze"), []).append(i)
        self.by_tier = {k: v for k, v in self.by_tier.items() if v}

        quotas = quotas or {"gold": 0.4, "silver": 0.4, "bronze": 0.2}
        quotas = {k: v for k, v in quotas.items() if k in self.by_tier and v > 0}
        tot = sum(quotas.values()) or 1.0
        # Largest-remainder allocation so the counts always sum to batch_size.
        raw = {k: self.batch_size * v / tot for k, v in quotas.items()}
        self.counts = {k: int(math.floor(v)) for k, v in raw.items()}
        rem = self.batch_size - sum(self.counts.values())
        for k in sorted(raw, key=lambda k: raw[k] - math.floor(raw[k]), reverse=True):
            if rem <= 0:
                break
            self.counts[k] += 1
            rem -= 1
        self.counts = {k: v for k, v in self.counts.items() if v > 0}
        # Epoch length is set by the tier that runs out first relative to its quota.
        self._nb = max(1, min(len(self.by_tier[k]) // c for k, c in self.counts.items()))

    def __len__(self) -> int:
        return self._nb

    def __iter__(self):
        rng = random.Random(self.seed)
        self.seed += 1
        pools = {k: list(v) for k, v in self.by_tier.items()}
        for v in pools.values():
            rng.shuffle(v)
        ptr = {k: 0 for k in pools}
        for _ in range(self._nb):
            batch: List[int] = []
            for k, c in self.counts.items():
                pool = pools[k]
                if ptr[k] + c > len(pool):  # wrap around, reshuffling
                    rng.shuffle(pool)
                    ptr[k] = 0
                batch.extend(pool[ptr[k]:ptr[k] + c])
                ptr[k] += c
            rng.shuffle(batch)
            yield batch
