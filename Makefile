.PHONY: verify build run neo4j-up neo4j-down stop clean \
        dashboard dashboard-down \
        stack-up stack-down \
        memory-up embedding-up metrics-up quality-up extended-up

# Default environment
SERVICE_HOST ?= 0.0.0.0
SERVICE_PORT ?= 8080
EMBEDDING_SERVER_URL ?= http://127.0.0.1:8090
EMBEDDING_MODEL_PATH ?= ./models/embeddinggemma-300m
NEO4J_URI ?= bolt://localhost:7688
NEO4J_USER ?= neo4j
NEO4J_PASSWORD ?= testpass
RETENTION_TICKS ?= 10

ifeq ($(OS),Windows_NT)
VERIFY := powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\verify.ps1
else
VERIFY := bash scripts/verify.sh
endif


verify: build
	@$(VERIFY)

build:
	cargo build --release

run: build
	@echo "Starting memory-server on $(SERVICE_HOST):$(SERVICE_PORT)"
	RUST_LOG=info \
	EMBEDDING_SERVER_URL=$(EMBEDDING_SERVER_URL) \
	NEO4J_URI=$(NEO4J_URI) NEO4J_USER=$(NEO4J_USER) NEO4J_PASSWORD=$(NEO4J_PASSWORD) \
	ORCHESTRATOR_FORCE_DISABLE=true DISABLE_SCHEDULERS=true \
	SERVICE_HOST=$(SERVICE_HOST) SERVICE_PORT=$(SERVICE_PORT) \
	./target/release/memory-server

.PHONY: memory-up-tuned
memory-up-tuned: build
	@bash -lc 'if ! curl -sf --max-time 2 http://$(SERVICE_HOST):$(SERVICE_PORT)/health >/dev/null; then \
	  nohup env RUST_LOG=info EMBEDDING_SERVER_URL=http://127.0.0.1:8091 \
	  NEO4J_URI=$(NEO4J_URI) NEO4J_USER=$(NEO4J_USER) NEO4J_PASSWORD=$(NEO4J_PASSWORD) \
	  ORCHESTRATOR_FORCE_DISABLE=true DISABLE_SCHEDULERS=true \
	  SERVICE_HOST=$(SERVICE_HOST) SERVICE_PORT=$(SERVICE_PORT) \
	  CAND_SEM_MULT=10 CAND_SEM_CAP=300 CAND_SEM_THRESH=0.08 \
	  CAND_LEX_MULT=5 CAND_LEX_CAP=100 \
	  CONTEXT_BOOST=0.35 ENABLE_MMR=1 MMR_LAMBDA=0.3 MMR_TOP=150 \
	  HYBRID_ALPHA=0.5 ALPHA_SHORT=0.4 ALPHA_LONG=0.6 \
	  ./target/release/memory-server >/tmp/memory_server_tuned.log 2>&1 & echo $$! > /tmp/memory_server_tuned.pid; \
	fi'
	@echo "Memory API (tuned): http://$(SERVICE_HOST):$(SERVICE_PORT)/health"


# Start everything needed for the live dashboard and print the URL
# Serve only the dashboard HTML (static) on :8099. Backend сервисы запускайте отдельными командами.
dashboard:
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dashboard.ps1 -Port 8099 -Dir reports -ApiBase http://127.0.0.1:$(SERVICE_PORT)
else
	@PORT=8099 API_BASE=http://$(SERVICE_HOST):$(SERVICE_PORT) sh scripts/dashboard.sh
endif

# Stop dashboard background processes and neo4j-test container
dashboard-down:
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dashboard.ps1 -Stop
else
	-@kill $$(cat /tmp/dashboard_static.pid) 2>/dev/null || true
	@echo "Dashboard static server stopped."
endif

# Bring up the full stack (embedding, neo4j, memory-server, streams)
stack-up: build embedding-up neo4j-up memory-up metrics-up quality-up extended-up
	@echo "Stack is up. Dashboard (backend API): http://$(SERVICE_HOST):$(SERVICE_PORT)/dashboard"
	@echo "Static dashboard (optional): make dashboard"

stack-down:
	-@kill $$(cat /tmp/embedding_server_dashboard.pid) 2>/dev/null || true
	-@kill $$(cat /tmp/memory_server_dashboard.pid) 2>/dev/null || true
	-@kill $$(cat /tmp/metrics_collector_dashboard.pid) 2>/dev/null || true
	-@kill $$(cat /tmp/quality_stream_dashboard.pid) 2>/dev/null || true
	-@kill $$(cat /tmp/quality_extended_dashboard.pid) 2>/dev/null || true
	-@docker rm -f neo4j-test >/dev/null 2>&1 || true
	@echo "Stack stopped."

embedding-up:
	@bash -lc 'if ! curl -sf --max-time 2 http://127.0.0.1:8091/health >/dev/null; then \
	  nohup env CUDA_VISIBLE_DEVICES="" EMBEDDING_MODEL_PATH=$(EMBEDDING_MODEL_PATH) EMBEDDING_SERVER_PORT=8091 python3 embedding_server.py >/tmp/embedding_server_dashboard.log 2>&1 & echo $$! > /tmp/embedding_server_dashboard.pid; \
	fi'
	@echo "Embedding: http://127.0.0.1:8091/health"

memory-up:
	@bash -lc 'if ! curl -sf --max-time 2 http://$(SERVICE_HOST):$(SERVICE_PORT)/health >/dev/null; then \
	  nohup env RUST_LOG=info EMBEDDING_SERVER_URL=http://127.0.0.1:8091 \
	  NEO4J_URI=$(NEO4J_URI) NEO4J_USER=$(NEO4J_USER) NEO4J_PASSWORD=$(NEO4J_PASSWORD) \
	  ORCHESTRATOR_FORCE_DISABLE=true DISABLE_SCHEDULERS=true \
	  SERVICE_HOST=$(SERVICE_HOST) SERVICE_PORT=$(SERVICE_PORT) \
	  ./target/release/memory-server >/tmp/memory_server_dashboard.log 2>&1 & echo $$! > /tmp/memory_server_dashboard.pid; \
	fi'
	@echo "Memory API: http://$(SERVICE_HOST):$(SERVICE_PORT)/health"

metrics-up:
	@bash -lc 'if [ ! -f /tmp/metrics_collector_dashboard.pid ] || ! ps -p $$(cat /tmp/metrics_collector_dashboard.pid 2>/dev/null) >/dev/null 2>&1; then \
	  nohup python3 scripts/metrics_collector.py --api http://$(SERVICE_HOST):$(SERVICE_PORT) --emb http://127.0.0.1:8091 --interval 2 --out reports/metrics_timeseries.jsonl >/tmp/metrics_collector_dashboard.log 2>&1 & echo $$! > /tmp/metrics_collector_dashboard.pid; \
	fi'
	@echo "Metrics stream -> reports/metrics_timeseries.jsonl"

quality-up:
	@bash -lc 'if [ ! -f /tmp/quality_stream_dashboard.pid ] || ! ps -p $$(cat /tmp/quality_stream_dashboard.pid 2>/dev/null) >/dev/null 2>&1; then \
	  nohup python3 scripts/quality_stream.py --host $(SERVICE_HOST) --port $(SERVICE_PORT) --interval 15 --out reports/quality_stream.jsonl --dataset datasets/quality/dataset.json --seed-if-missing >/tmp/quality_stream_dashboard.log 2>&1 & echo $$! > /tmp/quality_stream_dashboard.pid; \
	fi'
	@echo "Quality stream -> reports/quality_stream.jsonl"

extended-up:
	@bash -lc 'if [ ! -f /tmp/quality_extended_dashboard.pid ] || ! ps -p $$(cat /tmp/quality_extended_dashboard.pid 2>/dev/null) >/dev/null 2>&1; then \
	  nohup bash -lc "while true; do RETENTION_TICKS=$(RETENTION_TICKS) python3 scripts/quality_extended.py --host $(SERVICE_HOST) --port $(SERVICE_PORT) --out reports/quality_stream.jsonl --dataset datasets/quality/dataset.json; sleep 60; done" >/tmp/quality_extended_dashboard.log 2>&1 & echo $$! > /tmp/quality_extended_dashboard.pid; \
	fi'
	@echo "Extended metrics appender running (60s)"

.PHONY: dataset-seed
dataset-seed:
	@echo "Seeding dataset datasets/quality/dataset.json into http://$(SERVICE_HOST):$(SERVICE_PORT)"
	python3 scripts/seed_quality_dataset.py --host $(SERVICE_HOST) --port $(SERVICE_PORT) --dataset datasets/quality/dataset.json --out reports/dataset_seed.json

.PHONY: dataset-purge
dataset-purge:
	@echo "Purging contexts with prefix 'quality/' from http://$(SERVICE_HOST):$(SERVICE_PORT)"
	python3 scripts/purge_quality_contexts.py --host $(SERVICE_HOST) --port $(SERVICE_PORT) --prefix quality/


# Continuous metrics collection (requires a running memory-server and embedding server)
.PHONY: metrics-stream
metrics-stream:
	@echo "Sampling metrics from $(SERVICE_HOST):$(SERVICE_PORT) and embedding server..."
	python3 scripts/metrics_collector.py --api http://127.0.0.1:8080 --emb http://127.0.0.1:8091 --interval 2 --out /tmp/metrics_timeseries.jsonl

# Live quality evaluation stream (optionally seed synthetic dataset once)
.PHONY: quality-stream
quality-stream:
	@echo "Running live quality stream (no reseed by default; pass SEED=1 to seed if missing)"
	python3 scripts/quality_stream.py --host 127.0.0.1 --port 8080 --interval 15 --out /tmp/quality_stream.jsonl $(if $(SEED),--seed-if-missing,)


neo4j-up:
	@docker rm -f neo4j-test >/dev/null 2>&1 || true
	@docker run -d --name neo4j-test -p 7475:7474 -p 7688:7687 -e NEO4J_AUTH=$(NEO4J_USER)/$(NEO4J_PASSWORD) neo4j:5-community

neo4j-down:
	@docker rm -f neo4j-test >/dev/null 2>&1 || true

stop:
	@pkill -f target/release/memory-server || true

clean:
	cargo clean
