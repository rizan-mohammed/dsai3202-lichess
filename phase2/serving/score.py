"""score.py - the scoring script of the Lichess outcome endpoint (adapted from the Lab-10 pattern:
init() loads the registered model once, run() validates, then predicts).

Contract: see serving/INPUT_CONTRACT.md. The interface accepts COMPUTED FEATURES (the 16 Phase-1 model inputs),
one record per game after 20 plies, and returns P(White win), P(draw), P(Black win).
Client errors answer HTTP 400 with {"error", "code", ...}; a request is never half-answered.
"""
import json
import logging
import math
import os
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

try:                                           # present in the Azure inference server
    from azureml_inference_server_http.api.aml_response import AMLResponse
except Exception:                              # pragma: no cover - unit-test fallback
    AMLResponse = None

log = logging.getLogger("score")
log.setLevel(logging.INFO)        # INFO lines (init, one per request) are the only window into the running container
MAX_ROWS = 1000
CLASSES = ["white_win", "draw", "black_win"]
MODEL_NAME = os.environ.get("MODEL_NAME", "lichess-outcome")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")
# name: (kind, low, high). num = a JSON number, str = non-empty text, flag = 0/1 (or false/true)
FIELDS = {
    "white_elo": ("num", 100, 4000), "black_elo": ("num", 100, 4000),
    "elo_diff": ("num", -4000, 4000), "elo_mean": ("num", 100, 4000),
    "tc_base": ("num", 0, 21600), "tc_inc": ("num", 0, 600), "speed_class": ("num", 0, 3),
    "eco": ("str",), "eco_family": ("str",), "opening_family": ("str",),
    "w_clk_1": ("num", 0, 100000), "b_clk_1": ("num", 0, 100000),
    "w_used_10": ("num", -100000, 100000), "b_used_10": ("num", -100000, 100000),
    "used_diff_10": ("num", -200000, 200000), "unequal_start": ("flag",),
}
model = None
columns = []


def init():
    global model, columns
    base = Path(os.environ.get("AZUREML_MODEL_DIR", "."))
    found = sorted(base.rglob("MLmodel"))
    if not found:
        raise RuntimeError(f"no MLmodel file under AZUREML_MODEL_DIR={base}: the registered model is not an MLflow model")
    model = mlflow.sklearn.load_model(str(found[0].parent))
    columns = list(model.feature_names_in_)
    unknown = [c for c in columns if c not in FIELDS]
    if unknown:
        raise RuntimeError(f"model expects fields the contract does not define: {unknown}")
    log.info("init: loaded %s:%s from %s expects %s", MODEL_NAME, MODEL_VERSION, found[0].parent, columns)


def _respond(body, status=200):
    body = {**body, "model": f"{MODEL_NAME}:{MODEL_VERSION}"}
    if status == 200 or AMLResponse is None:
        return body
    return AMLResponse(body, status, json_str=True)      # json_str=True makes the server JSON-encode the dict once


def _error(code, message, status=400, **extra):
    log.warning("rejected request: %s: %s", code, message)
    return _respond({"error": message, "code": code, "expected_fields": columns or list(FIELDS), **extra}, status)


def _check_value(name, v):
    kind = FIELDS[name][0]
    if kind == "str":
        if not isinstance(v, str) or not v.strip():
            return "must be a non-empty string"
        return None
    if kind == "flag":
        if isinstance(v, bool) or v in (0, 1) and isinstance(v, (int, float)):
            return None
        return "must be 0 or 1"
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return f"must be a number, got {v!r}"
    lo, hi = FIELDS[name][1:]
    if not lo <= v <= hi:
        return f"out of range [{lo}, {hi}]: {v}"
    return None


def _warnings(r, i):
    """Derived fields should agree with their parts (Phase-1 definitions). A mismatch is reported, not refused."""
    w = []
    add = lambda f, m: w.append({"row": i, "field": f, "message": m})
    if abs(r["elo_diff"] - (r["white_elo"] - r["black_elo"])) > 0.5:
        add("elo_diff", "differs from white_elo - black_elo")
    if abs(r["elo_mean"] - (r["white_elo"] + r["black_elo"]) / 2) > 0.5:
        add("elo_mean", "differs from (white_elo + black_elo) / 2")
    base = r["tc_base"] + 40 * r["tc_inc"]
    sc = 0 if base < 180 else 1 if base < 480 else 2 if base < 1500 else 3
    if r["speed_class"] != sc:
        add("speed_class", f"time control implies class {sc}")
    if r["eco_family"] != r["eco"][:1]:
        add("eco_family", "differs from the first letter of eco")
    if abs(r["used_diff_10"] - (r["w_used_10"] - r["b_used_10"])) > 0.5:
        add("used_diff_10", "differs from w_used_10 - b_used_10")
    if int(r["unequal_start"]) != int(r["w_clk_1"] != r["b_clk_1"]):
        add("unequal_start", "differs from (w_clk_1 != b_clk_1)")
    return w


def run(raw_data):
    # 1. Is there a body at all?
    if raw_data is None or not str(raw_data).strip():
        return _error("empty_body", 'empty body: send {"data": [{...}, ...]}')
    # 2. Is it JSON?
    try:
        payload = json.loads(raw_data)
    except (ValueError, TypeError) as exc:
        return _error("invalid_json", f"body is not valid JSON: {exc}")
    # 3. Is it the shape we published?
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or not payload["data"]:
        return _error("bad_shape", "body must be an object with a non-empty list under 'data'")
    rows = payload["data"]
    if len(rows) > MAX_ROWS:
        return _error("too_many_rows", f"too many rows: {len(rows)} (max {MAX_ROWS} per request); split the request")
    if not all(isinstance(r, dict) for r in rows):
        return _error("bad_shape", "every item in 'data' must be an object")
    # 4. Are all the fields there?
    missing = sorted({c for c in columns for r in rows if c not in r})
    if missing:
        return _error("missing_field", f"missing field(s): {', '.join(missing)}")
    # 5. Right types and plausible ranges?
    problems = []
    for i, r in enumerate(rows):
        for c in columns:
            msg = _check_value(c, r[c])
            if msg:
                problems.append(f"row {i} {c}: {msg}")
    if problems:
        return _error("invalid_value", "; ".join(problems[:10]) + (" ..." if len(problems) > 10 else ""))
    ids = [r.get("id", i) for i, r in enumerate(rows)]
    # 6. Predict.
    try:
        df = pd.DataFrame([{c: (int(r[c]) if FIELDS[c][0] == "flag" else r[c]) for c in columns} for r in rows])[columns]
        p = model.predict_proba(df)
        full = np.zeros((len(p), 3))
        full[:, model.classes_] = p
        warns = [x for i, r in enumerate(rows) for x in _warnings(r, i)]
    except Exception:
        log.exception("prediction failed")
        return _error("internal_error", "the model could not score this request", status=500)
    log.info("scored n=%d model=%s:%s warnings=%d", len(rows), MODEL_NAME, MODEL_VERSION, len(warns))
    return _respond({"n": len(rows), "ids": ids, "classes": CLASSES,
                     "predictions": [int(k) for k in full.argmax(1)],
                     "labels": [CLASSES[k] for k in full.argmax(1)],
                     "probabilities": [[round(float(x), 6) for x in row] for row in full],
                     "warnings": warns[:20]})
