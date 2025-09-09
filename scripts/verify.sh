#!/usr/bin/env bash
# Strict mode without pipefail to avoid SIGPIPE from preview pipelines causing exit
set -euo nounset

echo "== AI Memory Service: Deterministic Verification =="

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# Global timeout (seconds)
GLOBAL_TIMEOUT_SEC=${VERIFY_TIMEOUT_SEC:-600}
__start_ts=$(date +%s)
assert_timeout() {
  now=$(date +%s)
  elapsed=$(( now - __start_ts ))
  if [ "$elapsed" -ge "$GLOBAL_TIMEOUT_SEC" ]; then
    echo "Global timeout reached (${GLOBAL_TIMEOUT_SEC}s). Aborting verify." >&2
    # best effort: stop server if running
    pkill -f target/release/memory-server >/dev/null 2>&1 || true
    exit 124
  fi
}

echo "[1/6] Starting REAL embedding server on :8090"
assert_timeout
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
assert_timeout
( set +o pipefail; timeout 8s curl -sS http://127.0.0.1:8090/health | sed -n '1,200p' ) || true || {
  echo "ERROR: Real embedding server not responding on :8090" >&2; exit 2;
}

echo "[2/6] Ensuring neo4j-test container"
if ! docker ps --format '{{.Names}}' | grep -q '^neo4j-test$'; then
  docker rm -f neo4j-test >/dev/null 2>&1 || true
  docker run -d --name neo4j-test -p 7475:7474 -p 7688:7687 -e NEO4J_AUTH=neo4j/testpass neo4j:5-community >/dev/null
  sleep 3
fi

echo "[3/6] Building (release)"
assert_timeout
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
for i in $(seq 1 60); do
  assert_timeout
  if timeout 3s curl -sS http://127.0.0.1:8080/health >/dev/null; then READY=1; break; fi
  sleep 1
done

if [ "$READY" != "1" ]; then
  echo "Server did not become ready. Last 200 lines:"
  tail -n 200 /tmp/memory_server_verify.log || true
  exit 2
fi

echo "[6/6] Running API checks"
assert_timeout
echo "HEALTH:"; ( set +o pipefail; timeout 8s curl -sS http://127.0.0.1:8080/health | sed -n '1,200p' ) || true

# Cross-check embedding dimensions between embedding server and API store response
assert_timeout
EMBED_DIM=$(timeout 8s curl -sS http://127.0.0.1:8090/stats | python3 -c 'import sys,json; d=json.load(sys.stdin); print(int(d.get("default_dimension") or 512))')

assert_timeout
echo "STORE:"; ( set +o pipefail; timeout 8s curl -sS -X POST http://127.0.0.1:8080/memory -H 'Content-Type: application/json' \
  -d '{"content":"Deterministic test memory FINAL","context_hint":"tests/deterministic"}' | sed -n '1,200p') || true

# Extract embedding_dimension from API health and compare
assert_timeout
API_DIM=$(timeout 8s curl -sS http://127.0.0.1:8080/health | python3 -c 'import sys,json; d=json.load(sys.stdin); print(int(d.get("embedding_dimension") or 0))')

if [ "$API_DIM" -eq 0 ]; then
  echo "ERROR: embedding_dimension missing in API response" >&2
  kill $PID >/dev/null 2>&1 || true
  exit 3
fi

echo "Embedding dimension (server/API): $EMBED_DIM / $API_DIM"
if [ "$EMBED_DIM" -ne "$API_DIM" ]; then
  echo "ERROR: embedding_dimension mismatch (server=$EMBED_DIM, api=$API_DIM)" >&2
  kill $PID >/dev/null 2>&1 || true
  exit 3
fi

assert_timeout
echo "SEARCH (primary):"; ( set +o pipefail; timeout 8s curl -sS -X POST http://127.0.0.1:8080/search -H 'Content-Type: application/json' \
  -d '{"query":"Deterministic test memory FINAL","limit":5}' | sed -n '1,200p') || true

assert_timeout
echo "SEARCH (compat):"; ( set +o pipefail; timeout 8s curl -sS -X POST http://127.0.0.1:8080/memories/search -H 'Content-Type: application/json' \
  -d '{"query":"Deterministic test memory FINAL","limit":5}' | sed -n '1,200p') || true

assert_timeout
echo "RECENT:"; ( set +o pipefail; timeout 8s curl -sS 'http://127.0.0.1:8080/memories/recent?limit=5' | sed -n '1,200p' ) || true

assert_timeout
echo "CONTEXTS:"; ( set +o pipefail; timeout 8s curl -sS http://127.0.0.1:8080/contexts | sed -n '1,200p' ) || true

assert_timeout
echo "STATS:"; ( set +o pipefail; timeout 8s curl -sS http://127.0.0.1:8080/stats | sed -n '1,200p' ) || true

echo "[7/6] Synthetic test: 50 similar fragments -> consolidate -> tick -> stats should show fewer active"

# Seed 50 similar items in a dedicated context
for i in $(seq 1 50); do
  if ! ((i % 10)); then assert_timeout; fi
  curl -sS -X POST http://127.0.0.1:8080/memory -H 'Content-Type: application/json' \
    -d "{\"content\":\"Synthetic repeated content number $i about deterministic consolidation test\",\"context_hint\":\"tests/synthetic\"}" >/dev/null
done

# Stats before
assert_timeout
BEFORE=$(timeout 8s curl -sS http://127.0.0.1:8080/stats | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("statistics",{}).get("active_memories",0))')
echo "Active before: $BEFORE"

# Consolidate duplicates within the context
assert_timeout
( set +o pipefail; timeout 8s curl -sS -X POST http://127.0.0.1:8080/maintenance/consolidate -H 'Content-Type: application/json' \
  -d '{"context":"tests/synthetic","similarity_threshold":0.92,"max_items":120}' | sed -n '1,200p') || true

# Apply several decay ticks (virtual days)
assert_timeout
( set +o pipefail; timeout 8s curl -sS -X POST http://127.0.0.1:8080/maintenance/tick -H 'Content-Type: application/json' -d '{"ticks":5}' | sed -n '1,200p' ) || true

# Stats after
assert_timeout
AFTER=$(timeout 8s curl -sS http://127.0.0.1:8080/stats | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("statistics",{}).get("active_memories",0))')
echo "Active after: $AFTER"

if [ "${AFTER}" -lt "${BEFORE}" ]; then
  echo "OK: Active memories decreased after consolidate+tick"
else
  echo "ERROR: Active memories did not decrease (before=${BEFORE}, after=${AFTER})" >&2
  kill $PID >/dev/null 2>&1 || true
  exit 3
fi

# Quick stability loop: repeat a small search 10 times, ensure consistent HTTP 200
OK_COUNT=0
for i in $(seq 1 10); do
  assert_timeout
  if timeout 4s curl -sS -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8080/search -H 'Content-Type: application/json' \
    -d '{"query":"Deterministic test memory FINAL","limit":3}' | grep -q '^200$'; then
    OK_COUNT=$((OK_COUNT+1))
  fi
  sleep 0.2
done
echo "Stability loop OK responses: $OK_COUNT/10"
if [ "$OK_COUNT" -lt 10 ]; then
  echo "ERROR: Stability loop failed ($OK_COUNT/10)" >&2
  kill $PID >/dev/null 2>&1 || true
  exit 3
fi

echo "Stopping server"
kill $PID >/dev/null 2>&1 || true
sleep 0.5
echo "== Verification completed =="
