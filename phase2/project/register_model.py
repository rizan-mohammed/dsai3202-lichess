"""register_model.py - write the registration file for the selected model, from what its runs logged (adapted from Lab-9 model_yml.py).

    python3 project/register_model.py <final-hgb-run-id> <holdout-eval-run-id> <version>
Nothing is typed by hand: code commit, data/feature versions, split, parameters and metrics come from the training
run and the one holdout evaluation. Then run:  az ml model create -f project/model-v<version>.yml
The fitted preprocessing is INSIDE the registered model (it is one sklearn Pipeline), so it travels with it.
"""
import json
import sys
from aml_common import WEEK, connect

LIMITATIONS = ("One month (May 2017), rated standard games that reach ply 20 with clocks; trained on a hash sample of the training "
               "days (size in training_rows); other months, variants and rating drift untested; draws (about 4%) are rarely the top class.")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run_id, eval_id, version = sys.argv[1:4]
    _, c = connect()
    tr, ev = c.get_run(run_id), c.get_run(eval_id)
    p, m, ep, em = tr.data.params, tr.data.metrics, ev.data.params, ev.data.metrics
    assert ev.data.tags.get("stage") == "holdout", "second argument must be the holdout evaluation run"
    assert ep["candidate_run"] == run_id, "that holdout evaluation scored a different candidate"
    assert p["model"] == "hgb" and p["features"] == "full", "the selected model must be the full-feature gradient-boosted model"
    assert p["data_snapshot_sha256"] == ep["data_snapshot_sha256"] and p["split_id"] == ep["split_id"], "lineage mismatch"
    r = lambda d, k, n=5: str(round(d[k], n))
    t = {
        "stage": "selected", "code_commit": p["git_commit"], "evaluation_commit": ep.get("git_commit", ""),
        "training_run": run_id, "holdout_run": eval_id, "data_asset": "lichess-features:1",
        "data_snapshot_sha256": p["data_snapshot_sha256"], "split_id": p["split_id"], "split_method": p["split_method"],
        "deploy_from_ts": p["deploy_from_ts"], "features": p["feature_list"], "excluded_columns": p["excluded_columns"],
        "algorithm": "HistGradientBoostingClassifier (multiclass); preprocessing fitted on training rows inside the model",
        "hyperparameters": f"learning_rate={p['learning_rate']}, max_leaf_nodes={p['max_leaf_nodes']}, max_iter={p['max_iter']}, "
                           f"rounds_used={p.get('n_boosting_rounds', '?')}, early_stopping, seed={p['seed']}",
        "training_rows": f"{p['train_rows_used']} of {p['train_days_rows_total']} (hash sample by game id, seeded)",
        "libs": f"python {p['python']}, scikit-learn {p['lib_scikit-learn']}, pandas {p['lib_pandas']}, numpy {p['lib_numpy']}",
        "val_log_loss": r(m, "val_log_loss"), "val_accuracy": r(m, "val_accuracy"), "val_auc_white": r(m, "val_auc_white"),
        "val_ece": r(m, "val_ece"), "latency_p95_ms_model_only": r(m, "latency_p95_ms", 2),
        "test_log_loss": r(em, "test_logloss_candidate"), "test_log_loss_baseline": r(em, "test_logloss_baseline"),
        "test_improvement": r(em, "test_improvement"), "test_ci95": f"[{r(em, 'ci95_low')}, {r(em, 'ci95_high')}]",
        "h1_supported": ev.data.tags.get("h1_supported", "?"), "test_accuracy": r(em, "test_acc_candidate"),
        "input_contract": "computed features, 16 fields; see serving/INPUT_CONTRACT.md",
        "limitations": LIMITATIONS,
    }
    t = {k: str(v)[:250] for k, v in t.items()}
    desc = (f"Lichess game outcome after 20 plies (White win / draw / Black win). HistGradientBoosting on the full feature group, "
            f"trained from run {run_id} at commit {p['git_commit'][:10]}, evaluated once on the untouched test days (run {eval_id}): "
            f"test log loss {t['test_log_loss']} vs {t['test_log_loss_baseline']} for the rating-only logistic baseline "
            f"(improvement {t['test_improvement']}, 95% CI {t['test_ci95']}; H1 supported: {t['h1_supported']}). Limitations: {LIMITATIONS}")
    lines = ["$schema: https://azuremlschemas.azureedge.net/latest/model.schema.json", "name: lichess-outcome",
             f"version: {version}", "type: mlflow_model", f"path: azureml://jobs/{run_id}/outputs/artifacts/paths/model/",
             f"description: {json.dumps(desc)}", "tags:"] + [f"  {k}: {json.dumps(v)}" for k, v in t.items()]
    out = WEEK / "project" / f"model-v{version}.yml"
    out.write_text("\n".join(lines) + "\n")
    (WEEK / ".model_version").write_text(version + "\n")
    print("\n".join(lines) + f"\n\nwrote {out.relative_to(WEEK)} - now run: az ml model create -f {out.relative_to(WEEK)}")


if __name__ == "__main__":
    main()
