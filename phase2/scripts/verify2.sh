#!/usr/bin/env bash
# Day-2 checkpoint: baselines + sweep finished, one snapshot, one split, one code version, one commit.
cd "$(dirname "$0")/.."
python3 project/check_runs.py 2>&1 | tee evidence/day2-verify.txt
