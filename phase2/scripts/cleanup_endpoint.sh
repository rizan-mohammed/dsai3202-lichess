#!/usr/bin/env bash
# Delete YOUR endpoint (and its blue deployment) so it stops billing. The registered model, data and code are kept.
cd "$(dirname "$0")/.."
source .env 2>/dev/null || { echo "run bash scripts/setup.sh first"; exit 1; }
EP="eplichess${SA#amazonlake}"
echo "deleting endpoint $EP (model lichess-outcome and all data stay) ..."
az ml online-endpoint delete -n "$EP" --yes --no-wait 2>&1 | grep -v -i "experimental class"
for i in $(seq 1 40); do
  n=$(az ml online-endpoint list --query "[?name=='$EP'] | length(@)" -o tsv 2>/dev/null)
  [ "${n:-1}" = "0" ] && { echo "OK: $EP is gone ($(date -u +%FT%TZ))" | tee evidence/cleanup.txt; exit 0; }
  printf "."; sleep 15
done
echo; echo "STILL DELETING - run this script again in a few minutes; do not leave without seeing OK"; exit 1
