"""Train and validate one Lichess outcome model (Phase 2). One run = one MLflow run.

Adapted from the structure of Lab-9 train.py (argparse -> MLflow params/metrics/model, latency timer,
`Pipeline` that carries its preprocessing). Changes: Lichess data, three-class target, UTC-day
chronological split from ml_contract.py, hash-based stable sampling, log loss / ECE / AUC metrics,
HistGradientBoosting candidate, leakage guard. Nothing here reads the TEST or DEPLOY partitions.

Local test:  python train.py --data ./some_snapshot_folder --model hgb --features full
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import ml_contract as mc

HERE = Path(__file__).resolve().parent
CAT = ["speed_class", "eco", "eco_family", "opening_family"]      # string columns -> encoded inside the pipeline
LABELS = [0, 1, 2]


def parse_args():
    p = argparse.ArgumentParser(description="Train one Lichess outcome model on the training days only.")
    p.add_argument("--data", required=True, help="folder with the retained parquet parts + _manifest.json")
    p.add_argument("--model", choices=["majority", "logreg", "hgb"], default="hgb")
    p.add_argument("--features", choices=sorted(mc.GROUPS), default="full", help="feature group (see ml_contract.py)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_rows", type=int, default=300000, help="hash-sampled training rows; 0 = all training rows")
    p.add_argument("--val_rows", type=int, default=200000, help="hash-sampled validation rows; 0 = all validation rows")
    p.add_argument("--C", type=float, default=1.0, help="logreg: inverse regularisation")
    p.add_argument("--max_iter", type=int, default=500, help="hgb: maximum boosting rounds (early stopping applies)")
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--max_leaf_nodes", type=int, default=31)
    p.add_argument("--l2", type=float, default=0.0, help="hgb: l2_regularization")
    p.add_argument("--latency_rows", type=int, default=200, help="single-row predictions to time")
    p.add_argument("--git_commit", default="", help="commit of the code that produced this run")
    p.add_argument("--verify", action="store_true", help="re-hash every part against the manifest (slow; final runs)")
    p.add_argument("--run_name", default="")
    return p.parse_args()


# ---------------------------------------------------------------- data
def cheap_snapshot_check(folder):
    """Fast check on every trial: the part list equals the manifest; returns the recorded snapshot hash."""
    root = Path(folder)
    man = json.loads((root / "_manifest.json").read_text())
    have = {str(p.relative_to(root)) for p in root.rglob("*.parquet")}
    if have != {p["path"] for p in man["parts"]}:
        raise SystemExit("snapshot parts differ from _manifest.json - wrong data asset?")
    return man["snapshot_sha256"]


def load_train_val(folder, cols, a):
    """One file at a time: keep only the training and validation days, then a stable hash sample of each."""
    counts = mc.day_counts(folder)
    bounds = mc.split_boundaries(counts)
    day_part = mc.partition_of(counts.index.to_numpy() * mc.DAY, bounds)
    size = {k: int(counts.to_numpy()[day_part == k].sum()) for k in range(4)}
    want = {0: a.train_rows, 1: a.val_rows}
    frac = {k: (1.0 if want[k] <= 0 else min(1.0, want[k] / size[k])) for k in (0, 1)}
    keep = list(dict.fromkeys(cols + [mc.KEY_COL, mc.TIME_COL, mc.TARGET]))
    out = {0: [], 1: []}
    for f in mc.part_files(folder):
        df = pd.read_parquet(f, columns=keep)
        ep = mc.epoch_seconds(df[mc.TIME_COL])
        part = mc.partition_of(ep, bounds)
        h = mc.id_hash(df[mc.KEY_COL])
        for k in (0, 1):
            m = (part == k) & mc.hash_sample_mask(h, frac[k], a.seed + k)
            if m.any():
                out[k].append(df.loc[m, cols + [mc.TARGET]])
        del df
    tr, va = (pd.concat(out[k], ignore_index=True) for k in (0, 1))
    for d in (tr, va):
        for c in cols:
            if d[c].dtype == bool:
                d[c] = d[c].astype("int64")
    return tr, va, bounds, size, frac


# ---------------------------------------------------------------- model
def build_model(a, cols):
    cat = [c for c in cols if c in CAT]
    num = [c for c in cols if c not in CAT]
    if a.model == "majority":
        return Pipeline([("model", DummyClassifier(strategy="prior"))])
    if a.model == "logreg":
        parts = [("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                   ("scale", StandardScaler())]), num)]
        if cat:
            parts.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=40,
                                               sparse_output=True), cat))
        return Pipeline([("prep", ColumnTransformer(parts)),
                         ("model", LogisticRegression(C=a.C, max_iter=1000))])
    # hgb: categories become integer codes learned on TRAINING rows; unseen/rare -> missing (-1)
    parts = []
    if cat:
        parts.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                            encoded_missing_value=-1, max_categories=60), cat))
    parts.append(("num", "passthrough", num))
    clf = HistGradientBoostingClassifier(
        learning_rate=a.learning_rate, max_leaf_nodes=a.max_leaf_nodes, max_iter=a.max_iter,
        l2_regularization=a.l2, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
        categorical_features=list(range(len(cat))) if cat else None, random_state=a.seed)
    return Pipeline([("prep", ColumnTransformer(parts)), ("model", clf)])


# ---------------------------------------------------------------- metrics
def ece(p, y, bins=15):
    """Top-label expected calibration error: |accuracy - confidence| averaged over confidence bins."""
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def evaluate(pipe, X, y):
    p = pipe.predict_proba(X)
    full = np.zeros((len(p), 3))
    full[:, pipe.classes_] = p                      # a class absent from a tiny sample stays probability 0
    p = np.clip(full, 1e-15, 1)
    p = p / p.sum(axis=1, keepdims=True)
    return {"log_loss": float(log_loss(y, p, labels=LABELS)), "accuracy": float(accuracy_score(y, p.argmax(axis=1))),
            "auc_white": float(roc_auc_score((y == 0).astype(int), p[:, 0])), "ece": ece(p, y)}


def time_single_rows(pipe, X, n):
    """Model-compute latency only (the endpoint's client latency is measured separately)."""
    rows = X.iloc[: min(n, len(X))]
    pipe.predict_proba(rows.iloc[[0]])
    ms = []
    for i in range(len(rows)):
        t0 = time.perf_counter()
        pipe.predict_proba(rows.iloc[[i]])
        ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95))


def main():
    a = parse_args()
    cols = mc.approved_features(mc.GROUPS[a.features])          # raises if a leaking column sneaks in
    snap = cheap_snapshot_check(a.data)
    if a.verify:
        snap_full = mc.snapshot_fingerprint(a.data)
        assert snap_full == snap, "snapshot hash differs from its manifest"
    t0 = time.time()
    tr, va, bounds, size, frac = load_train_val(a.data, cols, a)
    print(f"train {len(tr):,} of {size[0]:,} days-0 rows (fraction {frac[0]:.3f}); "
          f"val {len(va):,} of {size[1]:,} (fraction {frac[1]:.3f}); load {time.time()-t0:.0f}s")
    split_id = ""
    sj = HERE / "split.json"
    if sj.exists():
        ref = json.loads(sj.read_text())
        assert ref["boundaries_epoch"] == bounds, "boundaries differ from split.json - not the split you checked"
        split_id = ref["split_id"]
    Xtr, ytr, Xva, yva = tr[cols], tr[mc.TARGET].to_numpy(), va[cols], va[mc.TARGET].to_numpy()
    prior = np.bincount(ytr, minlength=3) / len(ytr)

    pipe = build_model(a, cols)
    t1 = time.time()
    pipe.fit(Xtr, ytr)
    fit_s = time.time() - t1
    val = evaluate(pipe, Xva, yva)
    sub = Xtr.iloc[: min(100000, len(Xtr))]
    trn = evaluate(pipe, sub, ytr[: len(sub)])
    prior_ll = float(log_loss(yva, np.tile(prior, (len(yva), 1)), labels=LABELS))
    p50, p95 = time_single_rows(pipe, Xva, a.latency_rows)

    code_sha = hashlib.sha256((HERE / "train.py").read_bytes() + (HERE / "ml_contract.py").read_bytes()).hexdigest()
    with mlflow.start_run(run_name=a.run_name or None):
        mlflow.log_params({"model": a.model, "features": "none" if a.model == "majority" else a.features, "feature_list": "" if a.model == "majority" else ",".join(cols), "seed": a.seed,
                           "train_rows_requested": a.train_rows, "val_rows_requested": a.val_rows,
                           "train_rows_used": len(tr), "val_rows_used": len(va), "train_days_rows_total": size[0],
                           "val_days_rows_total": size[1], "sampling": "game-id hash, seeded, per partition",
                           "C": a.C, "max_iter": a.max_iter, "learning_rate": a.learning_rate,
                           "max_leaf_nodes": a.max_leaf_nodes, "l2": a.l2, "split_method": mc.METHOD,
                           "boundaries_epoch": json.dumps(bounds), "deploy_from_ts": bounds[2], "split_id": split_id,
                           "data_snapshot_sha256": snap, "code_sha256": code_sha, "git_commit": a.git_commit,
                           "excluded_columns": ",".join(sorted(mc.EXCLUDED)), "python": platform.python_version(),
                           **{f"lib_{n}": importlib.metadata.version(n) for n in
                              ("scikit-learn", "pandas", "numpy", "scipy", "pyarrow")}})
        if a.model == "hgb":
            mlflow.log_param("n_boosting_rounds", int(pipe.named_steps["model"].n_iter_))
        mlflow.log_metrics({"val_log_loss": val["log_loss"], "val_accuracy": val["accuracy"],
                            "val_auc_white": val["auc_white"], "val_ece": val["ece"],
                            "val_log_loss_prior": prior_ll, "val_gain_vs_prior": prior_ll - val["log_loss"],
                            "train_log_loss": trn["log_loss"], "train_accuracy": trn["accuracy"],
                            "train_val_gap": val["log_loss"] - trn["log_loss"], "fit_seconds": fit_s,
                            "latency_p50_ms": p50, "latency_p95_ms": p95})
        mlflow.set_tags({"phase": "2", "stage": "validation", "data_asset": "lichess-features:1"})
        ex = Xva.head(5)
        sig = infer_signature(ex, pipe.predict_proba(ex))
        mlflow.sklearn.log_model(pipe, "model", signature=sig, input_example=ex,
                                 pip_requirements=["scikit-learn==1.5.2", "pandas==2.2.3", "numpy==1.26.4",
                                                   "scipy==1.13.1", "cloudpickle"])
    print(f"{a.model:8s} {a.features:18s} val_log_loss {val['log_loss']:.5f} (prior {prior_ll:.5f}) "
          f"acc {val['accuracy']:.4f} auc {val['auc_white']:.4f} ece {val['ece']:.4f} "
          f"train_ll {trn['log_loss']:.5f} fit {fit_s:.0f}s p95 {p95:.1f}ms")


if __name__ == "__main__":
    main()
