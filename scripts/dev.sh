#!/usr/bin/env bash
# Boot backend + frontend together. Logs are tailed in this terminal;
# Ctrl-C kills both. Run from the repo root or via `make dev`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="$ROOT/server"
CLIENT="$ROOT/client"
VENV="$SERVER/.venv"
LOG_DIR="$ROOT/.logs"
SERVER_LOG="$LOG_DIR/server.log"
CLIENT_LOG="$LOG_DIR/client.log"
PORT_SERVER=8000
PORT_CLIENT=3000

mkdir -p "$LOG_DIR"

if [[ ! -x "$VENV/bin/uvicorn" ]]; then
  echo "✗ Python venv missing. Run 'make setup' first." >&2
  exit 1
fi
if [[ ! -d "$CLIENT/node_modules" ]]; then
  echo "✗ node_modules missing. Run 'make setup' first." >&2
  exit 1
fi

# Free the ports if something is already on them.
if lsof -ti :"$PORT_SERVER" >/dev/null 2>&1; then
  echo "⚠ port $PORT_SERVER busy — killing existing process"
  lsof -ti :"$PORT_SERVER" | xargs kill -9 2>/dev/null || true
fi
if lsof -ti :"$PORT_CLIENT" >/dev/null 2>&1; then
  echo "⚠ port $PORT_CLIENT busy — killing existing process"
  lsof -ti :"$PORT_CLIENT" | xargs kill -9 2>/dev/null || true
fi

cleanup() {
  echo
  echo "→ shutting down…"
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  [[ -n "${CLIENT_PID:-}" ]] && kill "$CLIENT_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "✓ stopped"
}
trap cleanup INT TERM EXIT

echo "→ backend  → http://localhost:$PORT_SERVER  (log: $SERVER_LOG)"
(
  cd "$SERVER"
  exec "$VENV/bin/uvicorn" app.main:app \
    --host 0.0.0.0 --port "$PORT_SERVER" --reload \
    >"$SERVER_LOG" 2>&1
) &
SERVER_PID=$!

echo "→ frontend → http://localhost:$PORT_CLIENT  (log: $CLIENT_LOG)"
(
  cd "$CLIENT"
  exec npm run dev >"$CLIENT_LOG" 2>&1
) &
CLIENT_PID=$!

# Wait for backend health (model load can take ~60s on first run).
echo -n "→ waiting for backend to be ready"
for i in {1..180}; do
  if curl -fsS "http://localhost:$PORT_SERVER/healthz" >/dev/null 2>&1; then
    echo " ✓"
    break
  fi
  echo -n "."
  sleep 1
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo
    echo "✗ backend exited. Last lines of $SERVER_LOG:" >&2
    tail -n 40 "$SERVER_LOG" >&2
    exit 1
  fi
done

echo
echo "──────────────────────────────────────────────────────────"
echo "  Open http://localhost:$PORT_CLIENT/debug"
echo "  Backend log: tail -f $SERVER_LOG"
echo "  Frontend log: tail -f $CLIENT_LOG"
echo "  Ctrl-C to stop both."
echo "──────────────────────────────────────────────────────────"

# Tail both logs into the foreground until either child dies or user hits Ctrl-C.
tail -f "$SERVER_LOG" "$CLIENT_LOG" &
TAIL_PID=$!

wait -n "$SERVER_PID" "$CLIENT_PID" || true
kill "$TAIL_PID" 2>/dev/null || true
