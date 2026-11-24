"""check_deploy.py - Day 4/5 checkpoint (used by scripts/verify4.sh): registered, deployed, validated."""
import json
import subprocess
import sys
from aml_common import WEEK, read_file


def az(*a):
    r = subprocess.run(["az", *a, "-o", "json"], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def jf(name):
    f = WEEK / "evidence" / name
    return json.loads(f.read_text()) if f.exists() else None


ver, ep = read_file(".model_version"), read_file(".endpoint_name")
m = az("ml", "model", "show", "-n", "lichess-outcome", "-v", ver) if ver else None
d = az("ml", "online-deployment", "show", "-n", "blue", "--endpoint-name", ep) if ep else None
e = az("ml", "online-endpoint", "show", "-n", ep) if ep else None
tags = (m or {}).get("tags", {})
ok = 0
def chk(n, c, fix):
    global ok
    print(("PASS  " if c else "FAIL  ") + n + ("" if c else f"  -> {fix}"))
    ok += bool(c)
need = ["code_commit", "training_run", "holdout_run", "data_asset", "data_snapshot_sha256", "split_id", "features", "val_log_loss", "test_log_loss", "limitations"]
chk(f"1 model lichess-outcome:{ver} registered with commit, run, data/feature versions, metrics, limitations", bool(m) and all(tags.get(k) for k in need), "az ml model create -f project/model-v<N>.yml")
chk("2 endpoint exists and is Succeeded", bool(e) and e.get("provisioning_state") == "Succeeded", "bash scripts/deploy.sh")
chk("3 deployment blue is Succeeded and serves the registered version", bool(d) and d.get("provisioning_state") == "Succeeded" and (str(d.get("model", "")).endswith(f"lichess-outcome:{ver}") or str(d.get("model", "")).endswith(f"/models/lichess-outcome/versions/{ver}")), "bash scripts/deploy.sh")
chk("4 blue takes 100% of the traffic", bool(e) and (e.get("traffic") or {}).get("blue") == 100, "az ml online-endpoint update -n $(cat .endpoint_name) --traffic blue=100")
f = jf("functional_checks.json")
chk("5 functional checks: valid and invalid inputs", bool(f) and f["passed"] == f["total"] and f["expected_model"] == f"lichess-outcome:{ver}", "python3 project/client_tests.py")
s = jf("same_input.json")
chk("6 same-input check: deployed answers as the saved model", bool(s) and s["result"] == "SAME" and s["expected_model"] == f"lichess-outcome:{ver}", "python3 project/same_input_check.py ...")
l = jf("latency.json")
chk("7 client latency measured against the stated target", bool(l) and l.get("passed"), "python3 project/latency_test.py")
print(f"{ok}/7 checks passed")
sys.exit(0 if ok == 7 else 1)
