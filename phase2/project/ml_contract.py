"""Evaluation contract for the Lichess project (Phase 2), adapted from Labs 8-9.

Copied into command-job project folders because Azure uploads  code: .  only.
One game is known at prediction time after 20 plies. The result, rating changes,
termination and ply count are never predictors. The split is by whole UTC days.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400
METHOD = "utc-day-chronological-v1"
TARGET = "label"                      # 0 White wins, 1 draw, 2 Black wins (Phase 1 gold)
TIME_COL = "game_ts"
KEY_COL = "game_id"
CLASS_NAMES = {0: "white_win", 1: "draw", 2: "black_win"}

# Feature groups for the ablation (each group adds to the one before it).
RATING = ["white_elo", "black_elo", "elo_diff", "elo_mean"]
TIME_CONTROL = ["tc_base", "tc_inc", "speed_class"]
OPENING = ["eco", "eco_family", "opening_family"]
CLOCKS = ["w_clk_1", "b_clk_1", "w_used_10", "b_used_10", "used_diff_10", "unequal_start"]
GROUPS = {"elo_diff_only": ["elo_diff"],
          "rating": RATING,
          "rating_tc": RATING + TIME_CONTROL,
          "rating_tc_opening": RATING + TIME_CONTROL + OPENING,
          "full": RATING + TIME_CONTROL + OPENING + CLOCKS}
CATEGORICAL = ["eco", "eco_family", "opening_family"]
EXCLUDED = {"white_diff", "black_diff", "termination", "n_plies", "result", "moves20", "clocks20",
            KEY_COL, TIME_COL, TARGET}


def snapshot_fingerprint(folder):
    """Verify retained bytes before training or evaluating them."""
    root = Path(folder)
    manifest = root / "_manifest.json"
    if not manifest.exists():
        raise ValueError("retained snapshot manifest missing: run retain_data.py and register a new data version")
    data = json.loads(manifest.read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*.parquet")}
    if actual != {p["path"] for p in data["parts"]}:
        raise ValueError("snapshot part list differs from its retained manifest")
    for part in data["parts"]:
        h = hashlib.sha256()
        with (root / part["path"]).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != part["sha256"]:
            raise ValueError(f"snapshot bytes changed: {part['path']}")
    payload = {k: v for k, v in data.items() if k != "snapshot_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if digest != data["snapshot_sha256"]:
        raise ValueError("snapshot manifest hash differs")
    return digest


def approved_features(columns):
    bad = [c for c in columns if c in EXCLUDED]
    if bad:
        raise ValueError(f"never predictors: {bad}")
    return list(columns)


def part_files(folder):
    files = sorted(Path(folder).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {folder}")
    return files


def epoch_seconds(series):
    return pd.to_datetime(series).astype("datetime64[s]").astype("int64").to_numpy()


def id_hash(series):
    """Stable 64-bit hash of each game id (same value in every run and on every machine)."""
    return pd.util.hash_pandas_object(series.astype(str), index=False).to_numpy()


def membership_hash(hashes):
    return hashlib.sha256(np.sort(np.asarray(hashes, dtype=np.uint64)).tobytes()).hexdigest()


def day_counts(folder):
    """Games per UTC day, read one file at a time (small memory)."""
    import pyarrow.parquet as pq
    counts = {}
    for f in part_files(folder):
        t = pq.read_table(f, columns=[TIME_COL]).to_pandas()
        days, n = np.unique(epoch_seconds(t[TIME_COL]) // DAY, return_counts=True)
        for d, c in zip(days, n):
            counts[int(d)] = counts.get(int(d), 0) + int(c)
    return pd.Series(counts).sort_index()


def split_boundaries(counts, val_size=.15, test_size=.15, deploy_size=.10):
    """Three boundaries (epoch seconds, each at the start of a UTC day): train | val | test | deploy.
    No day is divided, so the shares are approximate. Same-day games never cross a boundary."""
    fractions = [1 - val_size - test_size - deploy_size, val_size, test_size, deploy_size]
    if min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("all four partition fractions must be positive and sum to 1")
    if len(counts) < 4:
        raise ValueError("at least four distinct UTC days are needed for four partitions")
    ends, day_values, total = counts.cumsum().to_numpy(), counts.index.to_numpy(), int(counts.sum())
    positions = []
    for i, fraction in enumerate(np.cumsum(fractions)[:3]):
        pos = int(np.searchsorted(ends, total * fraction, side="left")) + 1
        pos = max((positions[-1] + 1 if positions else 1), min(pos, len(counts) - (3 - i)))
        positions.append(pos)
    return [int(day_values[p]) * DAY for p in positions]


def partition_of(epoch, boundaries):
    """0 train, 1 val, 2 test, 3 deploy."""
    return np.searchsorted(np.asarray(boundaries), epoch, side="right")


def hash_sample_mask(hashes, fraction, seed=42):
    """A stable sample chosen by game-id hash: the same games in every run, independent of file order."""
    if fraction >= 1:
        return np.ones(len(hashes), dtype=bool)
    mix = np.uint64(int(hashlib.sha256(str(seed).encode()).hexdigest()[:16], 16))
    return (np.asarray(hashes, dtype=np.uint64) ^ mix) < np.uint64(int(fraction * (2 ** 64 - 1)))


def scan_split(folder, boundaries):
    """One pass over the retained files: rows, label counts, time range and membership hash per partition."""
    import pyarrow.parquet as pq
    rows = [0] * 4
    labels = [np.zeros(3, dtype=np.int64) for _ in range(4)]
    tmin, tmax, ids = [None] * 4, [None] * 4, [[] for _ in range(4)]
    for f in part_files(folder):
        df = pq.read_table(f, columns=[KEY_COL, TIME_COL, TARGET]).to_pandas()
        ep, ih, y = epoch_seconds(df[TIME_COL]), id_hash(df[KEY_COL]), df[TARGET].to_numpy()
        part = partition_of(ep, boundaries)
        for k in range(4):
            m = part == k
            if not m.any():
                continue
            rows[k] += int(m.sum())
            labels[k] += np.bincount(y[m], minlength=3)
            ids[k].append(ih[m])
            lo, hi = int(ep[m].min()), int(ep[m].max())
            tmin[k] = lo if tmin[k] is None else min(tmin[k], lo)
            tmax[k] = hi if tmax[k] is None else max(tmax[k], hi)
    merged = [np.concatenate(x) if x else np.array([], dtype=np.uint64) for x in ids]
    everything = np.concatenate(merged)
    if len(np.unique(everything)) != len(everything):
        raise ValueError("duplicate game ids in the snapshot")
    return {"rows": rows, "labels": [l.tolist() for l in labels], "tmin": tmin, "tmax": tmax,
            "membership": [membership_hash(m) for m in merged]}


def split_manifest(scan, boundaries, seed=42, data_sha256=None):
    names = ["train", "val", "test", "deploy"]
    iso = lambda t: pd.to_datetime(t, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    out = {"method": METHOD, "seed": seed, "data_sha256": data_sha256,
           "rule": "partition = number of boundaries <= the game's UTC start time; boundaries are UTC day starts",
           "boundaries_epoch": boundaries, "boundaries_utc": [iso(b) for b in boundaries]}
    for i, n in enumerate(names):
        out[n] = {"rows": scan["rows"][i], "class_counts_white_draw_black": scan["labels"][i],
                  "first_game_utc": iso(scan["tmin"][i]), "last_game_utc": iso(scan["tmax"][i]),
                  "membership_sha256": scan["membership"][i]}
    out["deploy_from_ts"] = boundaries[2]
    out["split_id"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    return out
