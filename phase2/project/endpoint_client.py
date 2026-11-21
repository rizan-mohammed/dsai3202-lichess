"""Tiny HTTP client shared by client_tests.py, same_input_check.py and latency_test.py (standard library only)."""
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

WEEK = Path(__file__).resolve().parent.parent


def add_args(ap: argparse.ArgumentParser):
    ap.add_argument("--uri", help="scoring URL (default: read from Azure for the endpoint in .endpoint_name)")
    ap.add_argument("--key", default=None, help="endpoint key (default: read from Azure; never printed or saved)")
    ap.add_argument("--version", default=None, help="expected model version (default: .model_version)")


def az(*cmd):
    return subprocess.run(["az", *cmd], capture_output=True, text=True, check=True).stdout.strip()


def connect(a):
    ep = (WEEK / ".endpoint_name").read_text().strip() if (WEEK / ".endpoint_name").exists() else ""
    uri = a.uri or az("ml", "online-endpoint", "show", "-n", ep, "--query", "scoring_uri", "-o", "tsv")
    key = a.key if a.key is not None else az("ml", "online-endpoint", "get-credentials", "-n", ep, "--query", "primaryKey", "-o", "tsv")
    version = a.version or ((WEEK / ".model_version").read_text().strip() if (WEEK / ".model_version").exists() else "1")
    return uri, key, version, ep


def post(uri, key, body: bytes):
    """Returns (http_status, parsed_json_or_text, seconds)."""
    req = urllib.request.Request(uri, data=body, method="POST", headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    dt = time.perf_counter() - t0
    try:
        return status, json.loads(raw), dt
    except ValueError:
        return status, raw.decode("utf-8", "replace")[:500], dt
