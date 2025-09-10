# Project TODO (Deterministic Roadmap)

## Scope & Goals
- Deliver robust, human-like long‑term memory with inter‑session context.
- Keep architecture intact; apply surgical, verifiable improvements only.
- Ensure deterministic build, run, and tests via real services and scripted checks.

## Deterministic Environment
- Embedding Server (real, 512D): run `python embedding_server.py` (port 8090)
- Neo4j (bolt://localhost:7688): `make neo4j-up`
- Build/Verify: `make verify` (runs scripts/verify.sh)
- Stability flags: `ORCHESTRATOR_FORCE_DISABLE=true`, `DISABLE_SCHEDULERS=true`

## Milestones (with Acceptance Criteria)
1) Stabilize API server lifecycle
   - Add/keep graceful shutdown; no spontaneous exits for ≥ 60s idle.
   - Criteria: make verify passes twice in a row; port 8080 stays open ≥ 60s.
   - Status: OK (verify.ps1 green; сервер стабилен)

2) API compatibility & UX
   - Keep existing routes; support aliases: /memories, /memories/search, /api/memory/*.
   - Compat search returns {"memories":[], "count":N, "success":true}.
   - Criteria: curl suite in verify shows 200s and expected shapes.
   - Status: OK (совместимые маршруты /memories, /recall, ответы health/metrics)

3) Embedding: real-only, no mocks (512D Matryoshka)
   - Default 512D; autodetect from embedding server (/stats) on dedicated port.
   - Criteria: verify spins real embedding_server.py (no external mocks); /memory response embedding_dimension equals runtime; vectors truncated consistently.
   - Status: OK (verify.sh форсирует поднятие real embedding на :8091; API health отдаёт embedding_dimension, совпадающий с сервером)

4) Inter‑session context quality
   - Confirm context path saving and retrieval; enrich relationships when available.
   - Criteria: After storing memories in different contexts, /contexts and /context/:path reflect counts; /search respects context filters.
   - Status: OK (list_contexts/get_context_info; link RELATED_TO on store; contextual expansion in recall)

5) Observability & limits
   - Trace critical steps (init, index rebuild, requests). Validate request size/limit guards.
   - Criteria: No panics in logs; 4xx on invalid inputs; no hangs under rapid 5 QPS for 30s.
   - Status: OK (trace spans; guards in API; verify stability loop 10/10)

6) Deterministic tests hardening (real embeddings)
   - verify.sh forced to run local embedding_server.py on :8091; retries/waits added.
   - Criteria: verify exits 0 on clean env with real embeddings; quality_eval reports sane metrics.
   - Status: OK (quality_eval интегрирован с гейтами; health/waits добавлены)

7) Context graph enrichment (human-like linking)
   - Add higher-level relationships: co-occurrence within time windows; link contexts discovered via search.
   - Keep writes minimal: reuse existing GraphStorage methods; add only lightweight relationship creation in service layer.
   - Criteria: after N inserts in related contexts, `/contexts` shows non-zero counts; advanced recall returns connected memories.
   - Status: OK (легковесные RELATED_TO между недавними из контекста; contextual_layer в recall)

8) Decay & consolidation (memory hygiene)
   - Implement decay: periodic importance attenuation using `brain.decay_rate`; floor at config threshold.
   - Consolidate: deduplicate near-duplicates; compress summaries into distilled nodes (via existing distillation engine when enabled).
   - API status: endpoints implemented — `POST /maintenance/decay`, `POST /maintenance/consolidate`, `POST /maintenance/tick` (+ aliases under `/api/v1/*` and `/api/memory/consolidate`).
   - Criteria: after 24h simulated ticks, low-importance items drop rank; duplicates reduced (Δcount ≥ 10% in synthetic set).

9) Observability & limits
   - Expose Prometheus metrics (counters: requests, errors; histograms: recall latency, embedding latency; gauges: cache size, service_available).
   - Tighten guards: total payload ≤ 1MB, limit ≤ 100, similarity threshold ∈ [0,1]; concurrency per route ≤ configurable.
   - Criteria: `/metrics` exports series; verify.sh shows p95 recall < 200ms for 10 stored items on dev box.
   - Status: Базовые метрики экспортируются (в т.ч. memory_store_duration_seconds); ограничения проверены

10) Failure handling (embedding required for store/search)
   - If embedding unavailable: return 503; health shows services.embedding=false; verify ensures real server.
   - Status: OK; store теперь запрещает пустые эмбеддинги; health/metrics отражают недоступность

11) Live metrics streaming during verify/development
   - Add scripts/metrics_collector.py to sample /metrics + embedding /stats to JSONL
   - Add scripts/quality_stream.py to periodically evaluate P@k/MRR/nDCG (optional seed-once)
   - Integrate optional ENABLE_METRICS_STREAM=1 into verify.sh
   - Criteria: real‑time JSONL артефакты в /tmp/*, стабильные записи на протяжении verify
   - Status: OK (скрипты добавлены, Makefile цели metrics-stream/quality-stream)

## Validation Protocol
- Before/after any change: `make verify`.
- If flakiness occurs, capture /tmp/* logs, fix root cause, re‑verify.
- Weekly: run a 30s smoke (5 QPS search) to check for stability (no panics, steady latency).

## Out‑of‑Scope (for now)
- Cloud infra, CI, GPU acceleration, production orchestrator.
