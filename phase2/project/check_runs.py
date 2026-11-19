"""check_runs.py - are the Day-2 runs traceable and comparable? (used by scripts/verify2.sh; prints PASS/FAIL)"""
import json
import re
import sys
from aml_common import WEEK, connect, read_file

_, client = connect()
exp = client.get_experiment_by_name("lichess-phase2")
runs = [] if exp is None else [r for r in client.search_runs([exp.experiment_id], "attributes.status = 'FINISHED'", max_results=500)
                              if "val_log_loss" in r.data.metrics]
snap = json.loads((WEEK / ".data_source.json").read_text())["snapshot_sha256"]
by = lambda **kw: [r for r in runs if all(r.data.params.get(k) == v for k, v in kw.items())]
ok = 0
def chk(name, cond, fix):
    global ok
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"  -> {fix}"))
    ok += bool(cond)
sha = read_file(".git_commit")
chk("1 .git_commit holds a 40-character commit id", bool(re.fullmatch(r"[0-9a-f]{40}", sha)), "echo <full sha from git rev-parse HEAD> > .git_commit")
chk("2 majority baseline finished", len(by(model="majority")) >= 1, "bash scripts/submit.sh baseline-majority project/job.yml --set inputs.model=majority")
chk("3 rating-only logistic baseline (elo_diff only) finished", len(by(model="logreg", features="elo_diff_only")) >= 1,
    "bash scripts/submit.sh baseline-logreg-elo project/job.yml --set inputs.model=logreg --set inputs.features=elo_diff_only")
combos = {(r.data.params.get("model"), r.data.params.get("features")) for r in runs if r.data.params.get("features") != "elo_diff_only"}
need = {(m, f) for m in ("logreg", "hgb") for f in ("rating", "rating_tc", "rating_tc_opening", "full")}
chk("4 sweep covers all 8 model x feature-group combinations", need <= combos, f"missing {sorted(need - combos)}")
chk("5 every run read the same retained snapshot", runs and all(r.data.params.get("data_snapshot_sha256") == snap for r in runs), "rerun on lichess-features:1")
chk("6 every run used the same split rule", runs and len({r.data.params.get("split_id") for r in runs}) == 1 and all(r.data.params.get("split_id") for r in runs),
    "copy split.json into project/ (submit.sh does it) and rerun")
chk("7 every run recorded the same code hash", runs and len({r.data.params.get("code_sha256") for r in runs}) == 1, "all runs must come from one version of train.py")
chk("8 every run recorded the commit", runs and all(r.data.params.get("git_commit") == sha for r in runs), "submit after writing .git_commit")
print(f"{ok}/8 checks passed")
sys.exit(0 if ok == 8 else 1)
