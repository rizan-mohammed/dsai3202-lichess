#!/usr/bin/env bash
# Phase 2, step 1: find your workspace, set the az defaults, read your quota. Changes nothing in Azure.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/.config/dsai3202/lab3.env 2>/dev/null || { echo "STOP: ~/.config/dsai3202/lab3.env is missing (Lab-3 settings)"; exit 1; }
HANDOFF=~/.config/dsai3202/ml-handoff.json
WS="$( [ -f "$HANDOFF" ] && python3 -c "import json;print(json.load(open('$HANDOFF')).get('WS',''))" || true )"
[ -n "$WS" ] || WS="$(az ml workspace list -g "$RG" --query "[?starts_with(name,'mlw')].name | [0]" -o tsv 2>/dev/null)"
[ -n "$WS" ] || { echo "STOP: no Azure ML workspace found in $RG (did Lab-8 create one?)"; exit 1; }
FS="$(az ml workspace list -g "$RG" --query "[?starts_with(name,'fs')].name | [0]" -o tsv 2>/dev/null)"
az extension show -n ml >/dev/null 2>&1 || az extension add -n ml --only-show-errors
az configure --defaults group="$RG" workspace="$WS" >/dev/null
printf 'RG=%s\nWS=%s\nFS=%s\nSA=%s\n' "$RG" "$WS" "${FS:-}" "$SA" > .env
echo "Group:     $RG"; echo "Workspace: $WS"; echo "Feature store: ${FS:-not found (optional)}"; echo "Lake:      $SA"
az ml workspace show -n "$WS" -g "$RG" --query "{Name:name,Region:location}" -o table
echo; echo "Compute quota in this workspace's region (rows with a limit above 0):"
az ml compute list-usage --resource-group "$RG" --workspace-name "$WS" \
  --query "[?limit>\`0\`].{Family:name.localized_value,Used:current_value,Limit:limit}" -o table 2>&1 | head -30
python3 -c "import adlfs, azure.identity, pyarrow, pandas; print('python packages: OK')" 2>&1 | tail -1
