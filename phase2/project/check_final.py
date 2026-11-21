"""check_final.py - Day-3 checkpoint (used by scripts/verify3.sh)."""
import json
import sys
from aml_common import WEEK, connect

_, client = connect()
exp = client.get_experiment_by_name("lichess-phase2")
runs = [r for r in client.search_runs([exp.experiment_id], max_results=500) if r.info.status == "FINISHED"]
ev = [r for r in runs if r.data.tags.get("stage") == "holdout"]
ok = 0
def chk(name, cond, fix):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"  -> {fix}"))
    ok += bool(cond)
fin = {n: [r for r in runs if r.info.run_name == n] for n in ("final-majority", "final-logreg-elo", "final-hgb")}
for i, n in enumerate(fin, 1):
    chk(f"{i} {n} finished with >= 1,000,000 training rows",
        any(float(r.data.params.get("train_rows_used", 0)) >= 1_000_000 for r in fin[n]), f"submit it (see the Day-3 steps)")
e = ev[0] if ev else None
chk("4 holdout evaluation finished", e is not None, "submit project/eval_job.yml")
chk("5 test membership matched split.json", bool(e) and e.data.params.get("test_membership_ok") == "true", "rerun the evaluation")
chk("6 all three models were trained on the same rows", bool(e), "rerun the evaluation")
res = WEEK / "evidence" / "final" / "holdout" / "holdout_results.json"
chk("7 results downloaded to evidence/final/holdout", res.exists(), "python3 project/fetch_artifacts.py <eval-run-id> holdout evidence/final")
chk("8 H1 verdict recorded", res.exists() and "supported" in json.loads(res.read_text())["verdict"], "download the results first")
print(f"{ok}/8 checks passed")
sys.exit(0 if ok == 8 else 1)
