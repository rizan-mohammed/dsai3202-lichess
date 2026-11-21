#!/usr/bin/env bash
cd "$(dirname "$0")/.."
mkdir -p evidence
python3 project/check_deploy.py 2>&1 | grep -v -i -E "warning|from google|experimental" | tee evidence/day4-verify.txt
