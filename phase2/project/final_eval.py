"""final_eval.py - the ONE evaluation on the untouched TEST days (Phase 2, II.2).

Scores three already-trained models on exactly the same test games: the majority reference, the
rating-only logistic baseline and the selected gradient-boosted candidate. Then:
  * checks the test games are exactly the split recorded in split.json (membership hash),
  * paired bootstrap of the per-game log-loss improvement (candidate vs baseline) -> the H1 verdict,
  * error analysis: slices (time control, Elo band, rating gap, clock availability), confusion matrix,
    calibration table, representative errors.
Models are loaded from their MLflow runs (the fitted preprocessing is inside each model).
Runs as an Azure ML job (eval_job.yml) so it uses the same Python and scikit-learn as training.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

import ml_contract as mc
from train import ece

HERE = Path(__file__).resolve().parent
THRESHOLD = 0.005                       # H1: improvement of at least 0.005 log loss, 95% CI excluding 0
SLICE_COLS = ["speed_class", "white_elo", "black_elo", "elo_mean", "elo_diff", "tc_base", "tc_inc", "eco", "w_clk_1"]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--candidate", required=True, help="run id of the selected gradient-boosted model")
    p.add_argument("--baseline", required=True, help="run id of the rating-only logistic baseline (same training rows)")
    p.add_argument("--majority", required=True, help="run id of the majority/prior reference (same training rows)")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--git_commit", default="")
    p.add_argument("--out", default="holdout")
    return p.parse_args()


def probs(pipe, X):
    p = pipe.predict_proba(X)
    full = np.zeros((len(p), 3))
    full[:, pipe.classes_] = p
    full = np.clip(full, 1e-15, 1)
    return full / full.sum(axis=1, keepdims=True)


def metrics(p, y):
    ll = -np.log(p[np.arange(len(y)), y])
    return {"log_loss": float(ll.mean()), "accuracy": float(accuracy_score(y, p.argmax(1))),
            "auc_white": float(roc_auc_score((y == 0).astype(int), p[:, 0])), "ece": ece(p, y)}, ll


def sliced(df, key, base_ll, cand_ll, cand_p, y):
    rows = []
    for name, idx in df.groupby(key, observed=True).indices.items():
        rows.append({"slice": str(name), "games": len(idx), "share": len(idx) / len(df),
                     "white_win_rate": float((y[idx] == 0).mean()), "draw_rate": float((y[idx] == 1).mean()),
                     "logloss_baseline": float(base_ll[idx].mean()), "logloss_candidate": float(cand_ll[idx].mean()),
                     "gain": float(base_ll[idx].mean() - cand_ll[idx].mean()),
                     "accuracy_candidate": float((cand_p[idx].argmax(1) == y[idx]).mean())})
    return pd.DataFrame(rows)


def main():
    a = args()
    t0 = time.time()
    client = MlflowClient()
    runs = {k: client.get_run(v) for k, v in {"candidate": a.candidate, "baseline": a.baseline, "majority": a.majority}.items()}
    ref = json.loads((HERE / "split.json").read_text())
    snap = mc.snapshot_fingerprint(a.data)                       # re-hashes every part: the bytes are the retained bytes
    for k, r in runs.items():
        pr = r.data.params
        assert pr["data_snapshot_sha256"] == snap, f"{k}: trained on a different snapshot"
        assert pr["split_id"] == ref["split_id"], f"{k}: trained under a different split"
    used = {k: r.data.params["train_rows_used"] for k, r in runs.items()}
    assert len(set(used.values())) == 1, f"models were trained on different training rows: {used}"

    # ---- load the test days only, and prove they are the recorded split
    counts = mc.day_counts(a.data)
    bounds = mc.split_boundaries(counts)
    assert bounds == ref["boundaries_epoch"], "boundaries differ from split.json"
    feats = {k: [c for c in r.data.params["feature_list"].split(",") if c] for k, r in runs.items()}
    feats["majority"] = feats["majority"] or ["elo_diff"]
    keep = list(dict.fromkeys(sum(feats.values(), []) + SLICE_COLS + [mc.KEY_COL, mc.TIME_COL, mc.TARGET]))
    parts, hashes = [], []
    for f in mc.part_files(a.data):
        df = pd.read_parquet(f, columns=keep)
        m = mc.partition_of(mc.epoch_seconds(df[mc.TIME_COL]), bounds) == 2
        if m.any():
            parts.append(df[m])
            hashes.append(mc.id_hash(df.loc[m, mc.KEY_COL]))
    test = pd.concat(parts, ignore_index=True)
    assert len(test) == ref["test"]["rows"], "test row count differs from split.json"
    assert mc.membership_hash(np.concatenate(hashes)) == ref["test"]["membership_sha256"], "test membership differs from split.json"
    print(f"test days: {len(test):,} games, membership matches split.json ({time.time()-t0:.0f}s)")
    for d in [test]:
        for c in d.columns:
            if d[c].dtype == bool:
                d[c] = d[c].astype("int64")
    y = test[mc.TARGET].to_numpy()

    # ---- predictions on identical rows
    P, M, L = {}, {}, {}
    for k, r in runs.items():
        pipe = mlflow.sklearn.load_model(f"runs:/{r.info.run_id}/model")
        P[k] = probs(pipe, test[feats[k]])
        M[k], L[k] = metrics(P[k], y)
        print(f"{k:10s} test log loss {M[k]['log_loss']:.5f}  acc {M[k]['accuracy']:.4f}  auc {M[k]['auc_white']:.4f}  ece {M[k]['ece']:.4f}")

    # ---- paired bootstrap of the improvement (baseline loss - candidate loss, per game)
    d = L["baseline"] - L["candidate"]
    rng = np.random.default_rng(a.seed)
    boots = np.empty(a.boot)
    for b in range(a.boot):
        boots[b] = d[rng.integers(0, len(d), len(d))].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    imp = float(d.mean())
    supported = bool(imp >= THRESHOLD and lo > 0)
    verdict = {"hypothesis": "H1: full gradient-boosted model lowers test log loss by >= 0.005 vs rating-only logistic, 95% CI excludes 0",
               "improvement_mean": imp, "ci95_low": float(lo), "ci95_high": float(hi), "bootstrap_resamples": a.boot,
               "threshold": THRESHOLD, "supported": supported, "test_games": int(len(test)),
               "improvement_vs_majority": float(L["majority"].mean() - L["candidate"].mean()),
               "note": "resampling is by game; games sharing players or days are not independent, so the interval is somewhat optimistic"}

    # ---- error analysis
    t = test
    t["elo_band"] = pd.cut(t["elo_mean"], [0, 1200, 1500, 1800, 2100, 5000], right=False,
                           labels=["<1200", "1200-1499", "1500-1799", "1800-2099", "2100+"])
    t["gap_band"] = pd.cut(t["elo_diff"].abs(), [0, 50, 150, 300, 5000], right=False, labels=["<50", "50-149", "150-299", "300+"])
    t["clock_info"] = np.where(t["w_clk_1"].notna(), "has_clock", "no_clock")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    for key in ["speed_class", "elo_band", "gap_band", "clock_info"]:
        sliced(t, key, L["baseline"], L["candidate"], P["candidate"], y).to_csv(out / f"slice_{key}.csv", index=False)
    pd.DataFrame(confusion_matrix(y, P["candidate"].argmax(1), labels=[0, 1, 2]), index=["true_white", "true_draw", "true_black"],
                 columns=["pred_white", "pred_draw", "pred_black"]).to_csv(out / "confusion_matrix_candidate.csv")
    conf, pred = P["candidate"].max(1), P["candidate"].argmax(1)
    edges = np.linspace(0, 1, 16)
    bi = np.clip(np.digitize(conf, edges[1:-1]), 0, 14)
    pd.DataFrame([{"bin": b, "games": int((bi == b).sum()), "mean_confidence": float(conf[bi == b].mean()),
                   "accuracy": float((pred[bi == b] == y[bi == b]).mean())} for b in range(15) if (bi == b).any()]
                 ).to_csv(out / "calibration_candidate.csv", index=False)
    wrong = np.where(pred != y)[0]
    pick = {"worst_loss": np.argsort(-L["candidate"])[:10],
            "confident_wrong": wrong[np.argsort(-conf[wrong])[:10]],
            "random_wrong": np.random.default_rng(a.seed).choice(wrong, 10, replace=False)}
    cols = [mc.KEY_COL, mc.TIME_COL, "speed_class", "tc_base", "tc_inc", "white_elo", "black_elo", "eco"]
    rep = []
    for kind, idx in pick.items():
        x = t.iloc[idx][cols].copy()
        x.insert(0, "kind", kind)
        x["true"] = np.array(["white", "draw", "black"])[y[idx]]
        for j, n in enumerate(["white", "draw", "black"]):
            x[f"p_{n}"] = P["candidate"][idx, j]
        x["loss_candidate"], x["loss_baseline"] = L["candidate"][idx], L["baseline"][idx]
        rep.append(x)
    pd.concat(rep).to_csv(out / "representative_errors.csv", index=False)
    res = {"test_metrics": M, "verdict": verdict, "runs": {k: r.info.run_id for k, r in runs.items()},
           "train_rows_used": used, "split_id": ref["split_id"], "snapshot_sha256": snap,
           "test_membership_sha256": ref["test"]["membership_sha256"]}
    (out / "holdout_results.json").write_text(json.dumps(res, indent=2))
    lines = [f"TEST DAYS {ref['test']['first_game_utc']} .. {ref['test']['last_game_utc']} UTC, {len(test):,} games (membership verified)",
             f"{'model':10s}{'log loss':>10s}{'accuracy':>10s}{'AUC white':>11s}{'ECE':>8s}"]
    lines += [f"{k:10s}{M[k]['log_loss']:>10.5f}{M[k]['accuracy']:>10.4f}{M[k]['auc_white']:>11.4f}{M[k]['ece']:>8.4f}" for k in ["majority", "baseline", "candidate"]]
    lines += ["", f"improvement (baseline - candidate) = {imp:.5f}   95% CI [{lo:.5f}, {hi:.5f}]   threshold {THRESHOLD}",
              f"H1 {'SUPPORTED' if supported else 'NOT SUPPORTED'}"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    code = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16] for f in ["final_eval.py", "train.py", "ml_contract.py"]}
    with mlflow.start_run():
        mlflow.log_params({"candidate_run": a.candidate, "baseline_run": a.baseline, "majority_run": a.majority,
                           "split_id": ref["split_id"], "data_snapshot_sha256": snap, "git_commit": a.git_commit,
                           "test_rows": len(test), "test_membership_ok": "true", "bootstrap_resamples": a.boot,
                           "train_rows_used": used["candidate"], **{f"code_{k}": v for k, v in code.items()}})
        mlflow.log_metrics({"test_logloss_candidate": M["candidate"]["log_loss"], "test_logloss_baseline": M["baseline"]["log_loss"],
                            "test_logloss_majority": M["majority"]["log_loss"], "test_improvement": imp, "ci95_low": float(lo),
                            "ci95_high": float(hi), "test_acc_candidate": M["candidate"]["accuracy"],
                            "test_acc_baseline": M["baseline"]["accuracy"], "test_auc_candidate": M["candidate"]["auc_white"],
                            "test_auc_baseline": M["baseline"]["auc_white"], "test_ece_candidate": M["candidate"]["ece"],
                            "test_ece_baseline": M["baseline"]["ece"]})
        mlflow.set_tags({"phase": "2", "stage": "holdout", "h1_supported": str(supported)})
        mlflow.log_artifacts(str(out), artifact_path="holdout")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
