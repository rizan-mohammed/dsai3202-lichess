#!/usr/bin/env bash
cd "$(dirname "$0")/.."
mkdir -p evidence
python3 project/check_final.py 2>&1 | grep -v -i -E "warning|from google" | tee evidence/day3-verify.txt
