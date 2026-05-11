#!/usr/bin/env bash
# Kill anything listening on the Phonemer dev ports.
set -euo pipefail

for port in 8000 3000; do
  pids="$(lsof -ti :"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "→ killing pid(s) $pids on :$port"
    kill -9 $pids
  else
    echo "✓ :$port already free"
  fi
done
