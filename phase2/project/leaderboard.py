"""leaderboard.py - every finished validation run of the experiment, sorted on validation log loss (lower is better).

    python3 project/leaderboard.py                 # all runs of experiment lichess-phase2
    python3 project/leaderboard.py --sweep <name>  # only the trials of one sweep
    python3 project/leaderboard.py --sort val_auc_white --desc
Writes evidence/leaderboard.txt and evidence/leaderboard.json. It does not choose for you:
the run you pick goes into .chosen by hand, with the reason written in the README.
Adapted from Lab-9 leaderboard.py.
"""
import argparse
import json
from aml_common import WEEK, connect

COLS = [("val_log_loss", "val_ll", 9), ("val_gain_vs_prior", "gain", 8), ("val_accuracy", "acc", 7), ("val_auc_white", "auc", 7),
        ("val_ece", "ece", 7), ("train_val_gap", "gap", 8), ("fit_seconds", "fit_s", 7), ("latency_p95_ms", "p95ms", 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="lichess-phase2")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--sort", default="val_log_loss")
    ap.add_argument("--desc", action="store_true")
    a = ap.parse_args()
    _, client = connect()
    exp = client.get_experiment_by_name(a.experiment)
    if exp is None:
        raise SystemExit(f"no experiment {a.experiment} yet - submit a job first (scripts/submit.sh)")
    flt = "attributes.status = 'FINISHED'"
    if a.sweep:
        flt += f" and tags.mlflow.parentRunId = '{a.sweep}'"
    runs = [r for r in client.search_runs([exp.experiment_id], filter_string=flt, max_results=500)
            if "val_log_loss" in r.data.metrics]
    if not runs:
        raise SystemExit("no finished runs with val_log_loss yet")
    rows = []
    for r in runs:
        p, m = r.data.params, r.data.metrics
        rows.append({"run_id": r.info.run_id, "name": r.info.run_name, "model": p.get("model"), "features": p.get("features"),
                     "lr": p.get("learning_rate"), "leaves": p.get("max_leaf_nodes"), "rounds": p.get("n_boosting_rounds"),
                     "train_rows": p.get("train_rows_used"), "val_rows": p.get("val_rows_used"),
                     "snapshot": (p.get("data_snapshot_sha256") or "")[:12], "git": (p.get("git_commit") or "")[:8],
                     "code": (p.get("code_sha256") or "")[:8], **{k: m.get(k) for k, _, _ in COLS}})
    rows.sort(key=lambda r: (r.get(a.sort) is None, r.get(a.sort) or 0), reverse=a.desc)
    f = lambda v, w: f"{v:>{w}.4f}" if isinstance(v, float) else f"{'-':>{w}}"
    lines = [f"experiment {a.experiment}   {len(rows)} finished runs   sorted on {a.sort} ({'high' if a.desc else 'low'} first)",
             f"{'#':<3}{'run id':<40}{'model':<9}{'features':<19}{'lr':>5}{'leaf':>5}" + "".join(f"{h:>{w}}" for _, h, w in COLS)
             + "  snap/git/code"]
    for i, r in enumerate(rows, 1):
        lines.append(f"{i:<3}{r['run_id']:<40}{r['model'] or '-':<9}{r['features'] or '-':<19}{(r['lr'] or '-') if r['model']=='hgb' else '-':>5}"
                     f"{(r['leaves'] or '-') if r['model']=='hgb' else '-':>5}" + "".join(f(r[k], w) for k, _, w in COLS)
                     + f"  {r['snapshot']}/{r['git']}/{r['code']}")
    out = "\n".join(lines)
    print(out)
    (WEEK / "evidence").mkdir(exist_ok=True)
    (WEEK / "evidence" / "leaderboard.txt").write_text(out + "\n")
    (WEEK / "evidence" / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote evidence/leaderboard.txt and evidence/leaderboard.json")


if __name__ == "__main__":
    main()
