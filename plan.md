# plan.md — актуальный план (реальные эмбеддинги, качество, maintenance)

- [x] Реализовать POST /maintenance/decay (возвращает количество обновлённых узлов)
- [x] Реализовать POST /maintenance/consolidate (context/similarity_threshold/max_items)
- [x] Реализовать POST /maintenance/tick (виртуальные сутки: ticks=N)
- [x] Добавить алиасы: /api/memory/consolidate, /api/search, /api/v1/memory, /api/v1/memory/search, /api/v1/maintenance/*
- [x] Обновить документацию (README.md) по maintenance и совместимым маршрутам
- [x] Отразить статус в TODO.md (Milestone 8)
- [x] Добавить synthetic-тест в scripts/verify.* (50 похожих, consolidate+tick → снижение active_memories)
- [x] Добавить /metrics и базовую инcтрументацию (store/recall, recall_latency)
- [x] Согласовать 512D Matryoshka (усечение векторов в store/search)
- [x] Протестировать совместимость curl‑скриптами и e2e‑сценариями (скрипты/scripts/verify.sh)
- [x] Оценить стабильность (детерминизм поиска, повторяемость hash) на 10 повторов
- [x] Убрать любые зависимости от mock‑эмбеддинга; запускать локальный embedding_server.py на :8091 в verify
- [x] Встроить quality_eval (P@5, MRR, nDCG) и зафиксировать минимальные пороги
- [x] Гибридное ранжирование (вектор + TF‑IDF) для устойчивости при слабых векторах

Новые задачи (качество и согласование embedding):
- [x] Включить task_type в HTTP-клиент (`/embed`, `/embed_batch`), проброс из Rust
- [x] На сервере всегда отдавать 512D (Matryoshka truncation) и добавить `dimension` в `/stats`
- [x] Перевести `store_memory` на Document‑эмбеддинги; Query оставить для запросов
- [x] Пропускать пустые эмбеддинги при перестроении индекса/обновлении индекса
- [x] В `verify.ps1` перед quality‑гейтами делать `dataset-purge` + `dataset-seed`
  - [x] После seed выполнять проверку `/maintenance/missing_embeddings_count` и backfill до нуля (≤3 прохода)
- [x] Fix Makefile (Windows): `python` вместо `python3` для seed/purge
 - [x] Нормализовать эмбеддинги после Matryoshka‑усечения на стороне клиента (Rust) при `NORMALIZE_EMBEDDINGS=true`
 - [x] Исправить параметры GPT‑5 Chat Completions (`max_tokens`) и добавить live‑тест оркестратора (`#[ignore]`)

Дополнительно (live‑метрики):
- [x] Добавить scripts/metrics_collector.py (стрим /metrics + /stats → JSONL)
- [x] Добавить scripts/quality_stream.py (периодическая оценка P@k/MRR/nDCG)
- [x] Интегрировать опциональный запуск коллекторов в verify (ENABLE_METRICS_STREAM=1)

Новые задачи (итог текущего этапа):
- [x] Backfill API: /maintenance/backfill_embeddings (limit)
- [x] Unit‑тесты на BM25/приоры/биграммы (детерминизм ранжирования)
- [x] Тюнинг ENV применён в Makefile(memory-up-tuned)/verify.ps1
  - [x] Не форсировать `HYBRID_ALPHA` в verify; включить авто‑тюнинг по длине запроса
- [x] Smoke‑проверка e2e (embedding 512D, store/search) — успешна, логи в reports/
 
- [x] Сгенерирован skeleton AXPL-2 JSON: `AXPL-2.json` (см. репозиторий) — требуется ручная ревизия и пересчёт `can`.
 - [x] Удалён tracked файл с зарезервированным именем `nul` (Windows). Создан новый коммит на `master` с деревом без пути `nul` (commit `c89a5d0`).
   - Примечание: индекс приведён в соответствие с новым HEAD; рабочая копия не модифицирована автоматически.
Новые задачи (документация и онбординг):
- [x] Создать Jupyter Notebook для быстрого контекстного онбординга: `notebooks/01_project_context.ipynb`
- [x] Сохранять сводку контекста в `reports/project-context-summary.json` из ноутбука
- [x] Актуализировать TODO.md/plan.md после добавления ноутбука

Совместимость API (алиасы orchestrator):
- [x] Добавить алиасы `/api/orchestrator/*` и `/api/v1/orchestrator/*` для всех orchestrator эндпоинтов
- [x] Добавить интеграционные тесты на алиасы: `tests/orchestrator_aliases_test.rs`

Нагрузочные и лимитирующие меры:
- [x] Ввести глобальный лимит конкурентности через `ConcurrencyLimitLayer` (по умолчанию 256; env `API_MAX_CONCURRENCY`)
- [x] Инструментировать метрики поиска: `record_recall_latency` и `memory_operations_total` в `/search*`, `/recall`, `/advanced`
- [x] Пер‑роут лимит конкурентности для `orchestrator/*` и алиасов через env `ORCHESTRATOR_MAX_CONCURRENCY` (по умолчанию 16)

Доп. совместимость и проверки:
- [x] Тест версии API: `tests/versioned_aliases_test.rs` (POST /api/v1/memory)
- [x] Тест maintenance алиаса: `tests/maintenance_alias_test.rs` (POST /api/memory/consolidate)
- [x] Тест заголовков `/metrics`: `tests/metrics_headers_test.rs`
- [x] Тест алиасов поиска: `tests/search_aliases_test.rs`
- [x] Тесты `/api/v1/maintenance/*`: `tests/v1_maintenance_test.rs`

Оценка производительности и качества (2025‑09‑10):
- [x] Собрать текущие метрики качества из `reports/quality_fast.json` и `quality_report.json`
- [x] Собрать текущие метрики производительности из `reports/fast_mem_*.out.log`, `metrics_timeseries.jsonl`
- [x] Сформировать Observability Guide и смоук‑скрипт (`docs/observability.md`, `scripts/quick_smoke.ps1`)


## Актуализация 2025‑09‑11 (Windows)
- Build: OK (`cargo build --release`), unit tests: 39/39 PASS.
- E2E verify: сервис поднялся, health/metrics/алиасы 200; synthetic consolidate+tick уменьшил active_memories.
- Quality eval (content relevance) ниже целевых порогов на части хостов: P@5=0.729, MRR=0.937, nDCG=0.764.
  - Пороговые значения не меняем; включён авто‑тюнинг `HYBRID_ALPHA` + post‑seed backfill для исключения пустых эмбеддингов.
- Integration cargo tests `api_integration_test` требуют живого API → при голом запуске 503; решено: добавлен эфемерный запуск `memory-server` в `scripts/run_tests.ps1` (ожидание `/health`, авто‑остановка), отдельный live‑профиль не требуется.
- Верификация улучшена:
  - Fresh Neo4j контейнер каждый прогон.
  - Loopback 127.0.0.1 для seed/purge на Windows (исправление make‑целей).
 - Интеграционные тесты стабилизированы: автозапуск Python `embedding_server.py` с `EMBEDDING_DEFAULT_DIMENSION=768` для согласования с тестовым конфигом; подготовка ONNX‑модели при отсутствии.
 - Починена строгая обработка булевых статусов в отчёте `run_tests.ps1` (исключены ложные PASSED при не‑bool значениях).

### Улучшения качества поиска (гибридный ранжировщик)
- [x] Синонимическое расширение запроса (детерминированная карта) для BM25.
- [x] Буст по тегам документов (`TAG_BOOST`, по умолчанию 0.35).
- [x] Настраиваемый буст точных биграмм в контенте (`EXACT_BIGRAM_BOOST`, по умолчанию 1.15–1.20).
- [x] Применено в `lexical_recall` и `score_hybrid_internal`; добавлены unit‑тесты.
 - [x] Переход на BM25F (поля: content/summary/tags/context; веса `W_*`, нормировка `B_*`, `BM25F_K1`), расчёт DF по всем полям и усреднение длины полей.
