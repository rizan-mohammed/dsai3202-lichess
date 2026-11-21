"""latency_test.py - client-observed latency of the deployed endpoint (PDF III), measured from this machine.
TARGET (stated before measuring): single-record request p95 <= 500 ms; 100-record request p95 <= 1000 ms.
The first requests after a deployment are slower, so 5 warm-up calls are made and not counted (and reported).

    python3 project/latency_test.py [--single 200] [--batch 30]
"""
import argparse
import json
import statistics

import endpoint_client as ec

TARGET_SINGLE_MS, TARGET_BATCH_MS = 500, 1000


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ec.add_args(ap)
    ap.add_argument("--single", type=int, default=200)
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--inputs", default=str(ec.WEEK / "evidence" / "fixed_inputs.json"))
    a = ap.parse_args()
    uri, key, version, ep = ec.connect(a)
    rows = json.loads(open(a.inputs).read())["data"]
    one = json.dumps({"data": rows[:1]}).encode()
    hundred = json.dumps({"data": rows[:100]}).encode()
    warm = [ec.post(uri, key, one)[2] * 1000 for _ in range(5)]
    s_ms, fails = [], 0
    for _ in range(a.single):
        st, _, dt = ec.post(uri, key, one)
        fails += st != 200
        s_ms.append(dt * 1000)
    b_ms = []
    for _ in range(a.batch):
        st, _, dt = ec.post(uri, key, hundred)
        fails += st != 200
        b_ms.append(dt * 1000)
    res = {"endpoint": ep, "model": f"lichess-outcome:{version}", "client": "VM in the same region (see README)",
           "single_record": {"requests": a.single, "p50_ms": pct(s_ms, 50), "p95_ms": pct(s_ms, 95), "p99_ms": pct(s_ms, 99), "max_ms": max(s_ms),
                             "target_p95_ms": TARGET_SINGLE_MS, "target_met": pct(s_ms, 95) <= TARGET_SINGLE_MS},
           "batch_100_records": {"requests": a.batch, "p50_ms": pct(b_ms, 50), "p95_ms": pct(b_ms, 95), "records_per_second": 100000 / statistics.mean(b_ms),
                                 "target_p95_ms": TARGET_BATCH_MS, "target_met": pct(b_ms, 95) <= TARGET_BATCH_MS},
           "warmup_ms": [round(x, 1) for x in warm], "failed_requests": fails}
    res["passed"] = bool(res["single_record"]["target_met"] and res["batch_100_records"]["target_met"] and fails == 0)
    (ec.WEEK / "evidence").mkdir(exist_ok=True)
    (ec.WEEK / "evidence" / "latency.json").write_text(json.dumps(res, indent=1))
    s, b = res["single_record"], res["batch_100_records"]
    print(f"single record : p50 {s['p50_ms']:.1f} ms  p95 {s['p95_ms']:.1f} ms  p99 {s['p99_ms']:.1f} ms  (target p95 <= {TARGET_SINGLE_MS} ms: {s['target_met']})")
    print(f"100 records   : p50 {b['p50_ms']:.1f} ms  p95 {b['p95_ms']:.1f} ms  {b['records_per_second']:.0f} records/s  (target p95 <= {TARGET_BATCH_MS} ms: {b['target_met']})")
    print(f"failed requests: {fails}   warm-up (not counted): {res['warmup_ms']}   passed: {res['passed']}")
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
