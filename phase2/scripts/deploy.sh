#!/usr/bin/env bash
# Create (or update) the endpoint and its blue deployment for one registered model version. Safe to run again = REDEPLOY.
#   bash scripts/deploy.sh [model-version]        (default: the version in .model_version)
# First deployment: 5-15 minutes (Azure builds the serving image). Later ones are quicker. The window can drop: Azure continues.
set -uo pipefail
cd "$(dirname "$0")/.."
source .env 2>/dev/null || { echo "run bash scripts/setup.sh first"; exit 1; }
VER="${1:-$(tr -d '[:space:]' < .model_version 2>/dev/null)}"
[ -n "$VER" ] || { echo "usage: bash scripts/deploy.sh <model-version>"; exit 1; }
EP="eplichess${SA#amazonlake}"; echo "$EP" > .endpoint_name; echo "$VER" > .model_version
mkdir -p evidence
LOG="evidence/deploy-$(date -u +%Y%m%dT%H%M%SZ).txt"
{
echo "== endpoint $EP   model lichess-outcome:$VER   $(date -u +%FT%TZ)"
az ml model show -n lichess-outcome -v "$VER" --query "{name:name,version:version,type:type}" -o table || { echo "STOP: lichess-outcome:$VER is not registered"; exit 1; }
if az ml environment show -n lichess-inference-env -v 1 -o none 2>/dev/null; then echo "serving environment lichess-inference-env:1 exists"
else az ml environment create -f serving/inference-env.yml --query "{name:name,version:version}" -o table || exit 1; fi
if az ml online-endpoint show -n "$EP" -o none 2>/dev/null; then echo "endpoint $EP exists"
else az ml online-endpoint create -f serving/endpoint.yml --name "$EP" --query "{name:name,state:provisioning_state,auth:auth_mode}" -o table || exit 1; fi
sed -e "s/__VER__/$VER/g" -e "s/eplichess-placeholder/$EP/" serving/blue.yml > serving/.blue.generated.yml
if az ml online-deployment show -n blue --endpoint-name "$EP" -o none 2>/dev/null; then ACTION="update"; EXTRA=""; else ACTION="create"; EXTRA="--all-traffic"; fi
echo "== az ml online-deployment $ACTION blue (this is the slow step)"
az ml online-deployment $ACTION -f serving/.blue.generated.yml --endpoint-name "$EP" $EXTRA --query "{name:name,state:provisioning_state,model:model}" -o table || { echo "DEPLOYMENT FAILED - read: az ml online-deployment get-logs -n blue --endpoint-name $EP --lines 100"; exit 1; }
az ml online-endpoint update -n "$EP" --traffic "blue=100" -o none
echo "== state"; az ml online-deployment show -n blue --endpoint-name "$EP" --query "{state:provisioning_state,model:model,instance_type:instance_type,count:instance_count}" -o table
az ml online-endpoint show -n "$EP" --query "{endpoint:name,state:provisioning_state,traffic:traffic,uri:scoring_uri}" -o json
echo "== one request"; az ml online-endpoint invoke -n "$EP" --request-file serving/sample-request.json | cut -c1-400
echo "== the lines score.py wrote"; az ml online-deployment get-logs -n blue --endpoint-name "$EP" --lines 300 2>/dev/null | grep -E "INFO:score|WARNING:score" | tail -5
} 2>&1 | grep -v -i -E "experimental class|warning: .*protobuf" | tee "$LOG"
exit "${PIPESTATUS[0]}"
