.PHONY: verify build run mock-embed neo4j-up neo4j-down stop clean

# Default environment
SERVICE_HOST ?= 0.0.0.0
SERVICE_PORT ?= 8080
EMBEDDING_SERVER_URL ?= http://127.0.0.1:8090
NEO4J_URI ?= bolt://localhost:7688
NEO4J_USER ?= neo4j
NEO4J_PASSWORD ?= testpass

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

mock-embed:
	@python3 -m venv .venv >/dev/null 2>&1 || true
	@. .venv/bin/activate && pip install -q aiohttp numpy >/dev/null 2>&1
	@. .venv/bin/activate && EMBEDDING_SERVER_PORT=8090 python scripts/mock_embedding_server.py

neo4j-up:
	@docker rm -f neo4j-test >/dev/null 2>&1 || true
	@docker run -d --name neo4j-test -p 7475:7474 -p 7688:7687 -e NEO4J_AUTH=$(NEO4J_USER)/$(NEO4J_PASSWORD) neo4j:5-community

neo4j-down:
	@docker rm -f neo4j-test >/dev/null 2>&1 || true

stop:
	@pkill -f target/release/memory-server || true

clean:
	cargo clean
