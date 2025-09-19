# AI Memory Service (Agent-first)

Minimal skeleton for an MCP-compatible memory backend that AI agents perceive as their own cognition.

## Quick Start
```bash
make start-dev          # docker compose up -d --build
curl -s http://localhost:8080/.well-known/mcp-spec | jq
make agent-test-baseline
make stop-dev
```

## Automation targets
| Command | Purpose |
|---------|---------|
| `make agent-test-baseline` | store/query determinism |
| `make agent-test-continuity` | timeline & time resolution |
| `make agent-test-session` | session snapshot/restore |
| `make agent-test-errors` | error policy mapping |
| `make agent-test-similarity` | vector search plumbing |
| `make agent-test-human` | cognitive summary (stub) |
| `make agent-test-plan` | plan lifecycle (stub) |
| `make agent-test-conflict` | conflict handling (stub) |
| `make agent-test-fallback` | offline fallback guarantees |
| `make verify-agent` | aggregated determinism checks |
| `make quality-report` | generate stub quality artefacts |
| `make maintenance-run` | run compaction (stub) |
| `make status-export` | write `STATUS.md` snapshot |
| `make docs-check` | ensure docs/manifest schemas exist |

## Directory layout
- `src/` — Rust memory server skeleton.
- `embedding_server.py` — placeholder embedding service.
- `docs/` — machine-readable manifests and specifications.
- `simulator/` — stub agent tests (Python 3.11+).
- `scripts/` — automation helpers.
- `models/` — EmbeddingGemma assets (not bundled).

## Compose profile
`docker-compose.pro.yml` defines services: memory, embedding, vector-index, neo4j, minio, agent-simulator. Observability profile is off by default.

## Next steps
1. Replace stub simulators with real HTTP flows.
2. Implement vector index, maintenance logic, cognitive analytics.
3. Fill quality-report with real metrics and gate releases.
4. Extend docs when schemas evolve (remember `make docs-check`).
