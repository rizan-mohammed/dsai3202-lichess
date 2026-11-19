#!/usr/bin/env bash
# bash scripts/submit.sh <label> <yaml> [--set key=value ...]
# Submits one Azure ML job, passes the commit id, and appends a line to evidence/jobs.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
source .env 2>/dev/null || { echo "run bash scripts/setup.sh first"; exit 1; }
LABEL="${1:?label}"; YAML="${2:?yaml}"; shift 2
GIT="$(cat .git_commit 2>/dev/null | tr -d '[:space:]')"
[ -n "$GIT" ] || { echo "STOP: .git_commit is empty. Commit the code on your laptop, then: echo <sha> > ~/Lab-P2/.git_commit"; exit 1; }
cp -f split.json project/split.json 2>/dev/null || { echo "STOP: split.json missing - run python3 project/check_split.py"; exit 1; }
mkdir -p evidence
EXTRA=(); grep -q "git_commit" "project/$(basename "$YAML")" 2>/dev/null && EXTRA=(--set "inputs.git_commit=$GIT")
NAME=$(az ml job create -f "$YAML" --set display_name="$LABEL" "${EXTRA[@]}" "$@" --query name -o tsv) || exit 1
echo "$(date -u +%FT%TZ)  $LABEL  $NAME  $YAML  $*" | tee -a evidence/jobs.txt
echo "$NAME" > ".last_job"
echo "wait:  bash scripts/wait.sh $NAME      portal: ml.azure.com -> Jobs -> $LABEL"
