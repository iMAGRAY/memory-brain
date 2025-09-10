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

echo "[1/6] Starting REAL embedding server on :8091 (forced)"
assert_timeout
# Detect model path
EMBED_MODEL_DIR="${EMBEDDING_MODEL_PATH:-./models/embeddinggemma-300m}"
if [ ! -d "$EMBED_MODEL_DIR" ]; then
  echo "ERROR: Embedding model directory not found: $EMBED_MODEL_DIR" >&2
  echo "Please place EmbeddingGemma-300M at ./models/embeddinggemma-300m or set EMBEDDING_MODEL_PATH" >&2
  exit 2
fi

pkill -f "embedding_server.py" >/dev/null 2>&1 || true
if ss -ltn '( sport = :8091 )' | grep -q 8091; then
  fuser -k 8091/tcp >/dev/null 2>&1 || true
fi
# Prefer venv, but fallback to system python if imports fail
PYTHON_BIN="./.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv .venv >/dev/null 2>&1 || true
fi
if [ -x "$PYTHON_BIN" ]; then
  . .venv/bin/activate
  python -m pip install -q --upgrade pip >/dev/null 2>&1 || true
  # Try to install dependencies quietly (network might be restricted)
  python -m pip install -q aiohttp aiohttp_cors numpy sentence_transformers torch >/dev/null 2>&1 || true
  if ! $PYTHON_BIN - <<'PY' >/dev/null 2>&1; then
import aiohttp, aiohttp_cors, numpy, sentence_transformers, torch
PY
    echo "Venv missing deps, will fallback to system python if available"
    PYTHON_BIN="python3"
  fi
else
  PYTHON_BIN="python3"
fi
CUDA_VISIBLE_DEVICES="" EMBEDDING_MODEL_PATH="$EMBED_MODEL_DIR" EMBEDDING_SERVER_PORT=8091 "$PYTHON_BIN" embedding_server.py >/tmp/real_embed.log 2>&1 &
# Wait for health up to 90s (model warmup may take a few seconds)
READY_EMB=0
for i in $(seq 1 90); do
  assert_timeout
  if timeout 2s curl -sS http://127.0.0.1:8091/health >/dev/null; then READY_EMB=1; break; fi
  sleep 1
done
if [ "$READY_EMB" != "1" ]; then
  echo "Embedding server failed to start on :8091. Last 100 lines of log:" >&2
  tail -n 100 /tmp/real_embed.log || true
  exit 2
fi
assert_timeout
timeout 8s curl -sS http://127.0.0.1:8091/health | sed -n '1,200p'

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

RUST_LOG=info EMBEDDING_SERVER_URL=http://127.0.0.1:8091 \
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

# Optional: start live metrics stream in background
if [ "${ENABLE_METRICS_STREAM:-0}" = "1" ]; then
  echo "[5.1/6] Starting live metrics collector (interval=${METRICS_INTERVAL_SEC:-2}s)"
  ( set +e; python3 scripts/metrics_collector.py \
      --api http://127.0.0.1:8080 \
      --emb http://127.0.0.1:8091 \
      --interval "${METRICS_INTERVAL_SEC:-2}" \
      --out /tmp/metrics_timeseries.jsonl \
      >/tmp/metrics_collector.log 2>&1 & echo $! > /tmp/metrics_collector.pid ) || true
fi

echo "[6/6] Running API checks"
assert_timeout
echo "HEALTH:"; ( set +o pipefail; timeout 8s curl -sS http://127.0.0.1:8080/health | sed -n '1,200p' ) || true

# Cross-check embedding dimensions between embedding server and API store response
assert_timeout
EMBED_DIM=$(timeout 8s curl -sS http://127.0.0.1:8091/stats | python3 -c 'import sys,json; d=json.load(sys.stdin); print(int(d.get("default_dimension") or 512))')

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

# Test aliases under /api/*
assert_timeout
echo "STORE (alias /api/memory):"; ( set +o pipefail; timeout 8s curl -sS -X POST http://127.0.0.1:8080/api/memory -H 'Content-Type: application/json' \
  -d '{"content":"Alias store memory","context_hint":"tests/alias","memory_type":"semantic"}' | sed -n '1,200p') || true

assert_timeout
echo "SEARCH (alias /api/memory/search):"; ( set +o pipefail; timeout 8s curl -sS -X POST http://127.0.0.1:8080/api/memory/search -H 'Content-Type: application/json' \
  -d '{"query":"Alias store memory","limit":3}' | sed -n '1,200p') || true

assert_timeout
echo "MAINTENANCE (alias /api/v1/maintenance/decay):"; ( set +o pipefail; timeout 8s curl -sS -X POST http://127.0.0.1:8080/api/v1/maintenance/decay -H 'Content-Type: application/json' \
  -d '{"dry_run":false}' | sed -n '1,200p') || true

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

# Optional: quality evaluation (synthetic dataset); does not fail verify
if [ "${RUN_QUALITY_EVAL:-1}" = "1" ]; then
  assert_timeout
  echo "[7.1/6] Quality evaluation (synthetic)"
  # Content-based relevance gates (tunable). Start realistic, then raise.
  MIN_P5=${MIN_P5:-0.70}
  MIN_MRR=${MIN_MRR:-0.85}
  MIN_NDCG=${MIN_NDCG:-0.70}
  if ! python3 scripts/quality_eval.py --host 127.0.0.1 --port 8080 --k 5 \
      --dataset datasets/quality/dataset.json --relevance content \
      --out /tmp/quality_report.json --min-p5 "$MIN_P5" --min-mrr "$MIN_MRR" --min-ndcg "$MIN_NDCG" \
      | sed -n '1,200p'; then
    echo "ERROR: Quality gates failed (min_p5=$MIN_P5, min_mrr=$MIN_MRR, min_ndcg=$MIN_NDCG)" >&2
    kill $PID >/dev/null 2>&1 || true
    exit 4
  fi
  # quick live stream (5 iters) and extended metrics append
  echo "[7.2/6] Quality stream (5 iters)"
  python3 scripts/quality_stream.py --host 127.0.0.1 --port 8080 --k 5 --interval 1 --max-iterations 5 --out /tmp/quality_stream.jsonl --seed-if-missing | sed -n '1,5p' || true
  echo "[7.3/6] Extended metrics (RETENTION_TICKS=${RETENTION_TICKS:-10})"
  RETENTION_TICKS=${RETENTION_TICKS:-10} python3 scripts/quality_extended.py --host 127.0.0.1 --port 8080 --k 5 --out /tmp/quality_stream.jsonl || true
fi

echo "Stopping server"
kill $PID >/dev/null 2>&1 || true
sleep 0.5
if [ -f /tmp/metrics_collector.pid ]; then
  COL_PID=$(cat /tmp/metrics_collector.pid || true)
  if [ -n "$COL_PID" ]; then kill "$COL_PID" >/dev/null 2>&1 || true; fi
  echo "Metrics timeseries captured at /tmp/metrics_timeseries.jsonl"
fi
# Persist reports to ./reports for dashboard
mkdir -p reports
if [ -f /tmp/quality_report.json ]; then cp /tmp/quality_report.json ./reports/quality_report.json || true; fi
if [ -f /tmp/metrics_timeseries.jsonl ]; then cp /tmp/metrics_timeseries.jsonl ./reports/metrics_timeseries.jsonl || true; fi
if [ -f /tmp/quality_stream.jsonl ]; then cp /tmp/quality_stream.jsonl ./reports/quality_stream.jsonl || true; fi
echo "== Verification completed =="
