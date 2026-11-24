"""client_tests.py - functional checks of the deployed endpoint (PDF III): valid and invalid inputs.
Saves every request, response, HTTP status and the model version under evidence/functional/, and a summary.

    python3 project/client_tests.py            # reads the endpoint from .endpoint_name
    python3 project/client_tests.py --uri http://localhost:5055/score --key "" --version 1   # local test
"""
import argparse
import json
import math
from pathlib import Path

import endpoint_client as ec

SERV = ec.WEEK / "serving"
OUT = ec.WEEK / "evidence" / "functional"


def main():
    ap = argparse.ArgumentParser()
    ec.add_args(ap)
    a = ap.parse_args()
    uri, key, version, ep = ec.connect(a)
    OUT.mkdir(parents=True, exist_ok=True)
    expect = f"lichess-outcome:{version}"
    sample = json.loads((SERV / "sample-request.json").read_text())
    one = {"data": [dict(sample["data"][0])]}
    one["data"][0].pop("id")
    big = {"data": [sample["data"][0]] * 1200}
    inconsistent = {"data": [{**sample["data"][0], "elo_diff": 999}]}
    cases = [  # name, body bytes, expected HTTP, expected error code (None = success)
        ("valid-three-rows", json.dumps(sample).encode(), 200, None),
        ("valid-one-row-no-id", json.dumps(one).encode(), 200, None),
        ("valid-with-warning", json.dumps(inconsistent).encode(), 200, None),
        ("invalid-missing-field", (SERV / "bad-requests" / "missing-field.json").read_bytes(), 424, "missing_field"),
        ("invalid-wrong-type", (SERV / "bad-requests" / "wrong-type.json").read_bytes(), 424, "invalid_value"),
        ("invalid-out-of-range", (SERV / "bad-requests" / "out-of-range.json").read_bytes(), 424, "invalid_value"),
        ("invalid-empty-body", b"", 424, "empty_body"),
        ("invalid-not-json", (SERV / "bad-requests" / "not-json.json").read_bytes(), 424, "invalid_json"),
        ("invalid-too-many-rows", json.dumps(big).encode(), 424, "too_many_rows"),
    ]
    rows, passed = [], 0
    for name, body, want_status, want_code in cases:
        status, resp, dt = ec.post(uri, key, body)
        ok = status == want_status
        if want_code is None and ok:
            ok = (isinstance(resp, dict) and resp.get("model") == expect and len(resp.get("probabilities", [])) == resp.get("n")
                  and all(abs(sum(p) - 1) < 1e-4 and all(math.isfinite(x) for x in p) for p in resp["probabilities"]))
            if name == "valid-with-warning":
                ok = ok and bool(resp.get("warnings"))
        elif want_code and ok:
            ok = isinstance(resp, dict) and resp.get("code") == want_code
        passed += ok
        (OUT / f"{name}.request.json").write_bytes(body if len(body) < 20000 else body[:20000])
        (OUT / f"{name}.response.json").write_text(json.dumps({"http_status": status, "seconds": round(dt, 4), "response": resp}, indent=1))
        rows.append({"case": name, "expected_http": want_status, "got_http": status, "model": resp.get("model") if isinstance(resp, dict) else None,
                     "passed": bool(ok)})
        print(f"{'PASS' if ok else 'FAIL'}  {name:24s} expected {want_status} got {status}  {resp.get('model') if isinstance(resp, dict) else ''}")
    summary = {"endpoint": ep, "expected_model": expect, "passed": passed, "total": len(cases), "cases": rows}
    (ec.WEEK / "evidence" / "functional_checks.json").write_text(json.dumps(summary, indent=1))
    print(f"{passed}/{len(cases)} functional checks passed (model {expect})")
    raise SystemExit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
