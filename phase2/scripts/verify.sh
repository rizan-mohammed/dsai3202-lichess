#!/usr/bin/env bash
# Phase 2 checkpoint 1: data retained, asset and environment registered, split computed.
set -uo pipefail
cd "$(dirname "$0")/.."
source .env 2>/dev/null || { echo "FAIL: run bash scripts/setup.sh first"; exit 1; }
pass=0; total=4
chk() { if eval "$2"; then echo "PASS  $1"; pass=$((pass+1)); else echo "FAIL  $1 -> $3"; fi; }
SNAP="$(python3 -c "import json;print(json.load(open('.data_source.json'))['snapshot_sha256'])" 2>/dev/null)"
chk "1 retained snapshot recorded (.data_source.json)" '[ -n "$SNAP" ]' "run: python3 project/retain_data.py $SA"
TAG="$(az ml data show -n lichess-features -v 1 --query tags.snapshot_sha256 -o tsv 2>/dev/null)"
chk "2 data asset lichess-features:1 points at that snapshot" '[ -n "$SNAP" ] && [ "$TAG" = "$SNAP" ]' "run: az ml data create -f project/data-asset.yml"
chk "3 environment lichess-env:1 registered" 'az ml environment show -n lichess-env -v 1 -o none 2>/dev/null' "run: az ml environment create -f project/environment.yml"
chk "4 split.json has four non-empty partitions" 'python3 -c "import json;d=json.load(open(\"split.json\"));assert all(d[k][\"rows\"]>0 for k in [\"train\",\"val\",\"test\",\"deploy\"])" 2>/dev/null' "run: python3 project/check_split.py"
echo "$pass/$total checks passed"; [ "$pass" = "$total" ]
