#!/usr/bin/env bash
# Wait for one Azure ML job to finish, printing its status as it changes.
#   bash scripts/wait.sh <job-name>
# Used instead of `az ml job stream`, whose log viewer sometimes stops with
# "binascii.Error: Incorrect padding" even though the job itself is fine.
J="$1"
[ -n "$J" ] || { echo "usage: bash scripts/wait.sh <job-name>"; exit 1; }
WSARGS=""; [ -n "$WS" ] && WSARGS="-w $WS"; [ -n "$RG" ] && WSARGS="$WSARGS -g $RG"
last=""
while :; do
  s=$(az ml job show -n "$J" $WSARGS --query status -o tsv 2>/dev/null)
  if [ "$s" != "$last" ]; then printf "\n  %s " "${s:-unknown}"; last="$s"; else printf "."; fi
  case "$s" in Completed|Failed|Canceled) echo; break ;; esac
  sleep 15
done
[ "$s" = Completed ]
