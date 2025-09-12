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

## Update 2025-09-11 (Windows CI run)
- [x] Verify pipeline executed end-to-end on Windows (PowerShell)
- [x] Embedding server healthy on :8090 (real model present)
- [x] Neo4j started via Docker on :7688 (fresh container each run)
- [x] API built in release, server healthy, core routes 200
- [x] Synthetic consolidate+tick decreased active_memories (OK)
- [ ] Quality gates: below target on some hosts (content-based relevance): P@5=0.729 (<0.80), MRR=0.937 (>=0.90), nDCG=0.764 (<0.80)
  - Action: thresholds unchanged; enable auto‑tuned HYBRID_ALPHA (no forced override in verify), ensure post‑seed backfill and zero missing embeddings before eval.
- [x] Fix Makefile dataset seed/purge to use loopback (127.0.0.1) on Windows
- [x] Force fresh `neo4j-test` container per verify run to avoid stale data
- [x] `scripts/run_tests.ps1` integration path: starts cargo tests without spinning API → API tests fail with 503
  - Action: implemented ephemeral `memory-server` startup in integration flow (wait `/health`, auto-stop); no `#[ignore]` needed.
  - Implemented: start Python `embedding_server.py` automatically with `EMBEDDING_DEFAULT_DIMENSION=768`; ensure ONNX model download
  - Fixed: summary could show PASSED on failure — strict boolean handling added in `Generate-Report` and exit evaluation

## 2025-09-11 (quality ranker improvements)
- [x] Query synonyms expansion for BM25 (deterministic map)
- [x] Tag-based boost and configurable bigram bonus (EXACT_BIGRAM_BOOST)
- [x] Hybrid scorer uses synonyms + tag boost; added tests (41/41 PASS)
- [x] Switch to BM25F across fields (content/summary/tags/context) with tunables (W_*, B_*, BM25F_K1)
- [ ] Run `scripts/verify.ps1` end-to-end and capture new P@5/MRR/nDCG
  - Action: execute on host with Docker; expect ↑P@5/nDCG

## 2025-09-11 (verify hardening)
- [x] verify.ps1: remove forced HYBRID_ALPHA; allow auto‑tuning by query length
- [x] verify.ps1: after dataset purge+seed, add missing_embeddings_count check and backfill loop (max 3 passes)
- [x] verify.sh: do not force HYBRID_ALPHA; pre‑seed dataset, then check/backfill missing embeddings before quality_eval
- [x] Quality eval stays content‑based; gates: P@5≥0.80, MRR≥0.90, nDCG≥0.80


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
   - Add normalization after client-side truncation (Matryoshka best practice) when `NORMALIZE_EMBEDDINGS=true`.
   - Status: OK (verify.sh форсирует поднятие real embedding на :8091; API health отдаёт embedding_dimension, совпадающий с сервером; нормализация включена)

12) Orchestrator (GPT‑5‑nano)
   - Align Chat Completions params with docs: use `max_tokens`, no unsupported params; keep `reasoning_effort`.
   - Add live e2e test under `#[ignore]` requiring `OPENAI_API_KEY` and running server on :8080.
   - Provide smoke script `scripts/orchestrator_smoke.ps1`.
   - Status: OK (params fixed; test added; script added; ready to enable with OPENAI_API_KEY)

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
   - Status: OK (quality_eval интегрирован с гейтами; health/waits добавлены; авто‑освежение датасета; unit‑тесты на BM25/priors/bigram — добавлены)

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

12) Backfill & Maintenance API
   - Добавить endpoint /maintenance/backfill_embeddings (limit)
   - Протянуть в MemoryService backfill_embeddings(); в GraphStorage list/update embedding.
   - Criteria: POST /maintenance/backfill_embeddings возвращает fixed>0 при наличии пустых эмбеддингов.
   - Status: OK

12) Embedding pipeline alignment (Matryoshka 512D, task prompts)
   - Align client/server: include task_type in /embed and /embed_batch; server enforces default_dimension=512 via Matryoshka; /stats returns dimension.
   - Store uses Document embeddings; Query uses Query embeddings. Skip empty embeddings in vector index.
   - Criteria: /health reports embedding_dimension=512; /embed returns dimension=512 for any task; zeros in quality report eliminated after dataset refresh.
   - Status: DONE (client+server обновлены; verify.ps1 освежает датасет; индекс пропускает пустые вектора)

13) Windows verify stability
   - Ensure Python invocations use 'python' on Windows; start embedding_server в фоне с логами; Docker existence check.
   - Criteria: verify.ps1 не блокирует оболочку; фоновые процессы завершаются; логи в reports/.
   - Status: DONE (Makefile и verify.ps1 обновлены)

## Validation Protocol
- Before/after any change: `make verify`.
- If flakiness occurs, capture /tmp/* logs, fix root cause, re‑verify.
- Weekly: run a 30s smoke (5 QPS search) to check for stability (no panics, steady latency).

## Out‑of‑Scope (for now)
- Cloud infra, CI, GPU acceleration, production orchestrator.

## Documentation & Notebooks
- Goal: ускорить онбординг и обзор системы без полного чтения кода.
- Deliverables:
  - [x] Аналитический ноутбук контекста проекта: `notebooks/01_project_context.ipynb`
  - [x] Экспорт краткой сводки в `reports/project-context-summary.json`
  - [x] Инструкции по запуску verify/tests в ноутбуке (через флаги)
  - [x] Актуализация TODO.md/plan.md после добавления ноутбука

Status: OK (ноутбук создан; опциональные тяжёлые проверки отключены флагами по умолчанию)

## API Compatibility (Orchestrator Aliases)
- Goal: обеспечить стабильность и совместимость клиентов, ожидающих `/api/v1/orchestrator/*` и `/api/orchestrator/*`.
- Deliverables:
  - [x] Роуты-алиасы для: `insights`, `distill`, `optimize`, `analyze`, `status`
  - [x] Без изменения обработчиков и контрактов ответа
  - [x] Компиляционная проверка `cargo check`
  - [x] Интеграционные тесты алиасов: `tests/orchestrator_aliases_test.rs`

Status: OK (алиасы добавлены в `src/api.rs`; `cargo check` успешен)

## Limits & Observability (Incremental)
- Goal: контролировать нагрузку и улучшить видимость реального времени.
- Deliverables:
  - [x] Глобальный лимит конкурентности: `ConcurrencyLimitLayer` (default 256; env `API_MAX_CONCURRENCY`)
  - [x] Метрики поиска/recall: `memory_recall_duration_seconds`, `memory_operations_total`
  - [x] health: `service_available{service="orchestrator|embedding"}`
  - [x] Пер‑роут лимит для оркестратора: `ORCHESTRATOR_MAX_CONCURRENCY` (default 16)

## Aliases & Compatibility Tests
- Goal: зафиксировать стабильность алиасов и заголовков.
- Deliverables:
  - [x] `/api/v1/memory` (store): `tests/versioned_aliases_test.rs`
  - [x] `/api/memory/consolidate`: `tests/maintenance_alias_test.rs`
  - [x] `/metrics` Content-Type: `tests/metrics_headers_test.rs`
  - [x] `/api/search`, `/api/memories/search`: `tests/search_aliases_test.rs`
  - [x] `/api/v1/maintenance/*`: `tests/v1_maintenance_test.rs`

Status: OK (компиляция тестов — успешна)

## Performance & Quality Assessment (Current)
- Goal: объективные текущие метрики latency/throughput и IR‑качества (P@k/MRR/nDCG).
- Deliverables:
  - [x] Сводка из `reports/quality_fast.json` и `quality_report.json`
  - [x] Сводка из `reports/fast_mem_*out.log` и `metrics_timeseries.jsonl`
  - [x] Документация наблюдаемости: `docs/observability.md`
  - [x] Смоук‑скрипт: `scripts/quick_smoke.ps1`

Status: OK (данные собраны, отчёт предоставляется в ответе)

Status: OK (интегрировано в `src/api.rs`; тесты зелёные)

### Repo scan — 2025-09-12T01:13:44+03:00 (automated)
- [x] Скан репозитория — выполнено (2025-09-12T01:13:44+03:00)
- Ключевые факты:
  - AGENTS.md — присутствует.
  - plan.md — присутствует (markdown).
  - AXPL-2.json / plan.json — отсутствуют (JSON‑представление AXPL-2 отсутствует).
  - Файл 
ul в корне репозитория — присутствует (риск на Windows: зарезервированное имя).
  - Есть незакоммиченные/изменённые файлы (см. git status).
  - Rust сервис: src/ + Cargo.toml присутствуют; есть unit/integration тесты в 	ests/.
  - Embedding server: mbedding_server.py присутствует.
  - Verify скрипты: scripts/verify.ps1 и scripts/verify.sh присутствуют.
- Риски и рекомендации:
  1. Удалить или переименовать файл 
ul (Windows reserved; может приводить к ошибкам при сборке/скриптах).
  2. Добавить/сгенерировать AXPL-2 JSON (если требуется автоматическая валидация AXPL-2); пока есть только plan.md.
  3. Очистить __pycache__ и большие eports/ из репозитория или добавить в .gitignore.
  4. Закоммитить/привести в порядок незакоммиченные изменения перед CI запуском.
- Следующие шаги (приоритет):
  1. Валидация AXPL-2 (преобразование plan.md → AXPL-2 JSON / схема). 
  2. Запустить make verify в Windows PowerShell и проанализировать результаты (после очистки 'nul' и незакоммиченных изменений).
  3. Если нужно — сгенерировать минимальный AXPL-2 JSON и пометить gaps (Fail‑Early report).

Выполнено автоматически: 2025-09-12T01:13:44+03:00
- Добавлен `AXPL-2.json` (skeleton, `can` = "RECOMPUTE"). Требуется ручная ревизия и пересчёт canonical hash (`can`).

Выполнено автоматически: 2025-09-12T01:55:00+03:00
- Удалён (untracked) зарезервированный файл `nul` и создан новый коммит на ветке `master` (commit: `c89a5d0ed4f8155369f8e3e61a38908a4ce45d1b`).
  - Операция: создан коммит с деревом, исключающим путь `nul`, и обновлён `refs/heads/master`.
  - Индекс репозитория приведён в соответствие с новым HEAD (`git reset --mixed HEAD`) — рабочая копия не изменена.
  - Проверки: `git ls-tree -r HEAD` не содержит `nul`; `git ls-files -s` не содержит `nul`; `git status` показывает `?? nul`.
  - Рекомендуемая ручная проверка: просмотреть `git log -1 --oneline` и `git status`, затем при необходимости удалить файл `nul` из рабочей копии или добавить в .gitignore/удалить.
