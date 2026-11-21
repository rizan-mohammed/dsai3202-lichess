"""same_input_check.py - offline vs deployed consistency on IDENTICAL fixed inputs (PDF III).

Scores evidence/fixed_inputs.json with the registered model downloaded to this machine, and sends the same file to the
endpoint; rows are matched by id. SAME needs: every id matched, every probability within TOLERANCE, every label equal and
the endpoint's answer naming the expected model version.

    az ml model download -n lichess-outcome -v 1 --download-path models
    ~/pv/bin/python project/same_input_check.py --model-dir models
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import endpoint_client as ec

TOLERANCE = 1e-5       # the endpoint rounds probabilities to 6 decimals, so rounding alone moves a value by at most 5e-7


def main():
    ap = argparse.ArgumentParser()
    ec.add_args(ap)
    ap.add_argument("--model-dir", required=True, help="folder containing the downloaded model (an MLmodel file somewhere below)")
    ap.add_argument("--inputs", default=str(ec.WEEK / "evidence" / "fixed_inputs.json"))
    a = ap.parse_args()
    import mlflow.sklearn
    uri, key, version, ep = ec.connect(a)
    payload = json.loads(Path(a.inputs).read_text())
    rows = payload["data"]
    mlmodel = sorted(Path(a.model_dir).rglob("MLmodel"))
    assert mlmodel, f"no MLmodel under {a.model_dir}"
    model = mlflow.sklearn.load_model(str(mlmodel[0].parent))
    cols = list(model.feature_names_in_)
    X = pd.DataFrame([{c: r[c] for c in cols} for r in rows])[cols]
    p_local = np.zeros((len(X), 3))
    p_local[:, model.classes_] = model.predict_proba(X)
    status, resp, dt = ec.post(uri, key, json.dumps(payload).encode())
    assert status == 200, f"endpoint answered HTTP {status}: {resp}"
    got = pd.DataFrame({"id": resp["ids"], "y_endpoint": resp["predictions"]})
    for j, n in enumerate(resp["classes"]):
        got[f"p_{n}_endpoint"] = [pr[j] for pr in resp["probabilities"]]
    mine = pd.DataFrame({"id": [r["id"] for r in rows], "y_local": p_local.argmax(1)})
    for j, n in enumerate(resp["classes"]):
        mine[f"p_{n}_local"] = p_local[:, j]
    both = mine.merge(got, on="id", how="inner")
    diffs = np.max([(both[f"p_{n}_local"] - both[f"p_{n}_endpoint"]).abs().to_numpy() for n in resp["classes"]], axis=0)
    max_diff = float(diffs.max())
    labels_equal = int((both["y_local"] == both["y_endpoint"]).sum())
    expect = f"lichess-outcome:{version}"
    same = (len(both) == len(rows) == len(set(r["id"] for r in rows)) and max_diff <= TOLERANCE
            and labels_equal == len(both) and resp.get("model") == expect)
    out = {"result": "SAME" if same else "DIFFERENT", "rows_sent": len(rows), "matched_by_id": len(both), "max_abs_probability_difference": max_diff,
           "tolerance": TOLERANCE, "labels_equal": labels_equal, "endpoint_model": resp.get("model"), "expected_model": expect,
           "endpoint": ep, "request_seconds": round(dt, 3), "inputs_file": str(Path(a.inputs).name)}
    (ec.WEEK / "evidence").mkdir(exist_ok=True)
    (ec.WEEK / "evidence" / "same_input.json").write_text(json.dumps(out, indent=1))
    both.to_csv(ec.WEEK / "evidence" / "same_input_rows.csv", index=False)
    print(both.head(5).to_string(index=False))
    print(f"\nmatched by id        : {len(both)} of {len(rows)}\nlargest |difference| : {max_diff:.2e}   (tolerance {TOLERANCE:.0e})\n"
          f"labels that agree    : {labels_equal} of {len(both)}\nendpoint answered as : {resp.get('model')} (expected {expect})\n"
          f"RESULT: {out['result']}")
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
