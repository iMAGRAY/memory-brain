#!/usr/bin/env bash
set -euo pipefail

echo "== AI Memory Service: Deterministic Verification =="

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

echo "[1/6] Starting REAL embedding server on :8090"
# Detect model path
EMBED_MODEL_DIR="${EMBEDDING_MODEL_PATH:-./models/embeddinggemma-300m}"
if [ ! -d "$EMBED_MODEL_DIR" ]; then
  echo "ERROR: Embedding model directory not found: $EMBED_MODEL_DIR" >&2
  echo "Please place EmbeddingGemma-300M at ./models/embeddinggemma-300m or set EMBEDDING_MODEL_PATH" >&2
  exit 2
fi

# Try to start real embedding server if not running
if ! ss -ltn '( sport = :8090 )' | grep -q 8090; then
  if [ ! -d .venv ]; then python3 -m venv .venv >/dev/null 2>&1 || true; fi
  . .venv/bin/activate
  # Ensure minimal deps (full stack may already be satisfied on the host)
  pip install -q aiohttp aiohttp_cors numpy sentence_transformers >/dev/null 2>&1 || true
  # torch installation may be environment-specific; assume preinstalled, otherwise user installs manually
  EMBEDDING_MODEL_PATH="$EMBED_MODEL_DIR" EMBEDDING_SERVER_PORT=8090 python embedding_server.py >/tmp/real_embed.log 2>&1 &
  sleep 2
fi
curl -sS http://127.0.0.1:8090/health | sed -n '1,200p' || {
  echo "ERROR: Real embedding server not responding on :8090" >&2; exit 2;
}

echo "[2/6] Ensuring neo4j-test container"
if ! docker ps --format '{{.Names}}' | grep -q '^neo4j-test$'; then
  docker rm -f neo4j-test >/dev/null 2>&1 || true
  docker run -d --name neo4j-test -p 7475:7474 -p 7688:7687 -e NEO4J_AUTH=neo4j/testpass neo4j:5-community >/dev/null
  sleep 3
fi

echo "[3/6] Building (release)"
RUST_LOG=info cargo build --release

echo "[4/6] Starting memory-server"
set +e
pkill -f target/release/memory-server >/dev/null 2>&1
set -e

RUST_LOG=info EMBEDDING_SERVER_URL=http://127.0.0.1:8090 \
NEO4J_URI=bolt://localhost:7688 NEO4J_USER=neo4j NEO4J_PASSWORD=testpass \
ORCHESTRATOR_FORCE_DISABLE=true DISABLE_SCHEDULERS=true \
SERVICE_HOST=0.0.0.0 SERVICE_PORT=8080 \
./target/release/memory-server >/tmp/memory_server_verify.log 2>&1 &
PID=$!
echo "PID=$PID"

echo "[5/6] Waiting for health"
READY=0
for i in $(seq 1 30); do
  if curl -sS http://127.0.0.1:8080/health >/dev/null; then READY=1; break; fi
  sleep 1
done

if [ "$READY" != "1" ]; then
  echo "Server did not become ready. Last 200 lines:"
  tail -n 200 /tmp/memory_server_verify.log || true
  exit 2
fi

echo "[6/6] Running API checks"
echo "HEALTH:"; curl -sS http://127.0.0.1:8080/health | sed -n '1,200p'

echo "STORE:"; curl -sS -X POST http://127.0.0.1:8080/memory -H 'Content-Type: application/json' \
  -d '{"content":"Deterministic test memory FINAL","context_hint":"tests/deterministic"}' | sed -n '1,200p'

echo "SEARCH (primary):"; curl -sS -X POST http://127.0.0.1:8080/search -H 'Content-Type: application/json' \
  -d '{"query":"Deterministic test memory FINAL","limit":5}' | sed -n '1,200p'

echo "SEARCH (compat):"; curl -sS -X POST http://127.0.0.1:8080/memories/search -H 'Content-Type: application/json' \
  -d '{"query":"Deterministic test memory FINAL","limit":5}' | sed -n '1,200p'

echo "RECENT:"; curl -sS 'http://127.0.0.1:8080/memories/recent?limit=5' | sed -n '1,200p'

echo "CONTEXTS:"; curl -sS http://127.0.0.1:8080/contexts | sed -n '1,200p'

echo "STATS:"; curl -sS http://127.0.0.1:8080/stats | sed -n '1,200p'

echo "[7/6] Synthetic test: 50 similar fragments -> consolidate -> tick -> stats should show fewer active"

# Seed 50 similar items in a dedicated context
for i in $(seq 1 50); do
  curl -sS -X POST http://127.0.0.1:8080/memory -H 'Content-Type: application/json' \
    -d "{\"content\":\"Synthetic repeated content number $i about deterministic consolidation test\",\"context_hint\":\"tests/synthetic\"}" >/dev/null
done

# Stats before
BEFORE=$(curl -sS http://127.0.0.1:8080/stats | python - <<'PY'
import sys, json
data=json.load(sys.stdin)
print(data.get('statistics',{}).get('active_memories',0))
PY
)
echo "Active before: $BEFORE"

# Consolidate duplicates within the context
curl -sS -X POST http://127.0.0.1:8080/maintenance/consolidate -H 'Content-Type: application/json' \
  -d '{"context":"tests/synthetic","similarity_threshold":0.92,"max_items":120}' | sed -n '1,200p'

# Apply several decay ticks (virtual days)
curl -sS -X POST http://127.0.0.1:8080/maintenance/tick -H 'Content-Type: application/json' -d '{"ticks":5}' | sed -n '1,200p'

# Stats after
AFTER=$(curl -sS http://127.0.0.1:8080/stats | python - <<'PY'
import sys, json
data=json.load(sys.stdin)
print(data.get('statistics',{}).get('active_memories',0))
PY
)
echo "Active after: $AFTER"

if [ "${AFTER}" -lt "${BEFORE}" ]; then
  echo "OK: Active memories decreased after consolidate+tick"
else
  echo "ERROR: Active memories did not decrease (before=${BEFORE}, after=${AFTER})" >&2
  kill $PID >/dev/null 2>&1 || true
  exit 3
fi

echo "Stopping server"
kill $PID >/dev/null 2>&1 || true
sleep 0.5
echo "== Verification completed =="
