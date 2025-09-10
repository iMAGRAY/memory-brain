#!/usr/bin/env sh
set -eu

PORT="${PORT:-8099}"
DIR="${1:-reports}"
API_BASE="${API_BASE:-http://127.0.0.1:8080}"

mkdir -p "$DIR"

PIDFILE="${TMPDIR:-/tmp}/dashboard_static.pid"

# If already running and healthy, keep it
if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE" 2>/dev/null || echo "")
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "[dashboard] Static server already running (pid=$PID)"
    echo "Open: http://127.0.0.1:${PORT}/dashboard.html?api=${API_BASE}"
    exit 0
  fi
fi

nohup python3 -m http.server "$PORT" --directory "$DIR" >/tmp/dashboard_static.log 2>&1 &
echo $! > "$PIDFILE"
echo "[dashboard] Serving $DIR on http://127.0.0.1:${PORT} (static)"
echo "Open: http://127.0.0.1:${PORT}/dashboard.html?api=${API_BASE}"

