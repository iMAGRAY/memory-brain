# TODO — AI Memory Service MCP (когнитивно-оптимизированный план, 2025-09-19)

## 0. Инварианты и когнитивные принципы
- **UTC + ISO‑8601**: все временные метки формата `2025-09-19T12:34:56Z`.
- **UUIDv7**: `req_id`, `Idempotency-Key`, `assembled_context_id`.
- **EmbeddingGemma**: dtype ∈ {bf16, fp32}; `truncate_dim` ∈ {128,256,512,768}; `normalize_embeddings=true`; промпты строго из спецификации.
- **GPT-5.0-nano**: Responses API; `max_output_tokens` обязателен; `reasoning.effort` ∈ {minimal,low,medium,high}; никаких `temperature/top_p`.
- **Evidence-First**: без `sha256_source` — отказ (422 MEM_NO_EVIDENCE).
- **SchemaGate**: jsonschema, `additionalProperties=false` для всех запросов/ответов.
- **DeterminismGate**: повторный запрос → идентичный ответ и `assembled_context.hash`.
- **Cognitive Ease**: единый набор терминов, предсказуемый формат сообщений, обязательные примеры в документации и автоинспекция схем.

## 1. Приоритеты и дедлайны
| Приоритет | Задачи | Старт | Дедлайн | Фокус |
|-----------|--------|-------|---------|-------|
| P0        | 20     | немедленно | 2025-10-03 | Контракты, детерминизм, когнитивная база |
| P1        | 24     | после P0   | 2025-10-17 | Качество ранжирования, UX агента |
| P2        | 16     | после P1   | 2025-10-31 | Расширения и масштабирование |

## 2. P0 — фундамент (выполнять последовательно)

### A. Контракт EmbeddingGemma и сервис
1. **[P0-EMB-01] Промпты и dtype**
   - В `embedding_server.py` задавать `task: search result | query: ...` и `title: ...| text: ...`.
   - Инициализация `SentenceTransformer(..., device=..., model_kwargs={"torch_dtype": ...})`.
   - Диагностика float16 → аварийное завершение с кодом EMBD_001.
   - `/embed` возвращает `dimension`, `dtype`, `norm_deviation`; |norm-1|>1e-6 ⇒ 422.
   - Тесты: `pytest tests/test_embedding_prompts_diagnosis.py`, `tests/test_embedding_norm.py`.

2. **[P0-EMB-02] Матрешка и самопроверка**
   - Проверка `truncate_dim` ∈ {128,256,512,768}; `normalize_embeddings`=true.
   - Возврат поля `self_check: {prompt_policy:true, matryoshka:true}`.
   - CLI `scripts/emb_health.py` для агента (одна команда, JSON).

### B. MCP-cлой и когнитивный интерфейс
3. **[P0-MCP-03] MCP Envelope + SchemaGate**
   - `McpEnvelope {action,input,idempotency_key,dry_run,meta}`.
   - jsonschema для каждого инструмента + автоинспекция `/mcp/schema/<tool>`.
   - Dry-run (валидация) / commit (исполнение). UUIDv7 проверка.
   - Тест: `tests/mcp_envelope_test.rs`.

4. **[P0-MCP-04] Интерактивная справка**
   - Эндпоинт `/mcp/help` → описание инструмента, пример запроса/ответа.
   - CLI `scripts/mcp_help.py <tool>` — выводит schema + sample.
   - Документация `docs/mcp_cheatsheet.md` (таблица инструментов, шаги).

### C. Детерминизм и контекст
5. **[P0-DET-05] DeterminismGate**
   - `assembled_context.hash = sha256("{id}|{type}|{score:.6f}|{text_norm}\n")`.
   - Логи содержат `{req_id, assembled_context_hash}`.
   - `scripts/verify.*` выполняют поиск 10×; расхождение ⇒ FAIL.
   - Артефакт `artifacts/determinism_report.json`.

6. **[P0-DET-06] Cognitive summary**
   - Каждое ответное сообщение содержит `summary` ≤ 240 символов + `details[]` с ссылками.
   - ИИ агент получает «TL;DR» без дополнительной обработки.

### D. GPT‑5.0-nano — «мозг» оркестратора
7. **[P0-GPT-07] Responses API миграция**
   - Модуль `secure_orchestration::call_responses_api` с `max_output_tokens`, `reasoning.effort`.
   - Удалить `temperature/top_p`; retries (>=3, exponential backoff + jitter).
   - Логи `{model, latency_ms, prompt_tokens, output_tokens, cost_usd}`.
   - Тест: `scripts/orchestrator_smoke.ps1` (авто вызывает help + sample).

8. **[P0-GPT-08] Контуры когнитивной простоты**
   - Единый формат промптов: system=«You are deterministic MCP brain», user=json.
   - Конвертор `/orchestrator/hints` → возвращает подходящий запрос (templates для агента).
   - Добавить `docs/gpt5_playbook.md` с 3 примерами (importance, enrich, summarize).

### E. Evidence, граф и память
9. **[P0-EVP-09] Evidence-first enforcement**
   - `MemoryService::store_memory` проверяет `sha256_source`. Нет → 422 MEM_NO_EVIDENCE.
   - Тест: `tests/evidence_gate.rs`.
   - Документ `docs/evidence_flow.md` (схема: input → sha256 → S3 → memory).

10. **[P0-GPH-10] Надёжные операции графа**
    - Параметризованные Cypher + транзакции `write_transaction`.
    - Перед вставкой RELATED_TO проверять существование; удвоение запрещено.
    - Метрика `graph_duplicates_blocked_total`.
    - Тест: `tests/graph_dedupe_test.rs`.

11. **[P0-MEM-11] Duplicate guard**
    - Проверка `sha256(text_norm)`; дубликат → 409.
    - Метрика `memory_duplicates_total`.

### F. Наблюдаемость, безопасность, когнитивные подсказки
12. **[P0-OBS-12] Структурные логи**
    - JSON-логи `{ts,level,op,req_id,context_hash,summary}`.
    - req_id прокидывать в embedding/GPT/Neo4j.
    - OTEL spans `openai.call`, `neo4j.write`, `embedding.encode`.

13. **[P0-OBS-13] Чёткие метрики**
    - Prometheus: `embedding_requests_total{dtype}`, `prompt_policy_violation_total`, `gpt5_cost_usd_total`, `determinism_failures_total`, `cognitive_hint_served_total`.
    - Документ `docs/observability.md` с примерами dashboard.

14. **[P0-SEC-14] Secrets & egress**
    - Секреты только через env/vault; `.env` → пример без значений.
    - `ENDPOINTS_ALLOWLIST` (OpenAI, Neo4j, MinIO). Нарушение → 403 + лог `security_violation`.

15. **[P0-QA-15] Verify pipeline v2**
    - Обновить `scripts/verify.ps1/.sh`: запускает real embedding + Neo4j, determinism, evidence, quality.
    - Артефакты: `determinism_report.json`, `evidence_report.json`, `quality_snapshot.json`, `cognitive_summary.txt`.

16. **[P0-REL-16] Retry & circuit breaker**
    - Обобщённый клиент (OpenAI/Neo4j/Embedding) с retries>=3, exponential backoff, circuit breaker.
    - Логи `{service, state=open|half|closed}`.

17. **[P0-REL-17] Idempotency**
    - Все мутации требуют `Idempotency-Key` и dry-run→commit сценарий.
    - Повтор → тот же assembled hash + HTTP 200 (идемпотент).

18. **[P0-SEC-18] Input sanitization**
    - NFKC нормализация; >2048 токенов → 413 Payload Too Large.
    - Метрика `security_violations_total`.

19. **[P0-DOC-19] Документация когнитивных практик**
    - `docs/cognitive_guide.md`: принципы «TL;DR + Details», формат ответов, примеры.
    - Обновить `README.md`/`plan.md` (highlights: deterministic brain, cognitive hints).

20. **[P0-UX-20] Cognitive onboarding**
    - CLI `scripts/getting_started.py` → за 1 команду показывает: список инструментов, примеры, проверка подключений.
    - Видео/анимация не требуется, только текстовые сценарии.

## 3. P1 — усиление качества и UX

### EmbeddingGemma & векторный слой
- **[P1-EMB-21] Авто-настройка batch_size**: адаптация к GPU/CPU; мониторинг latency; тесты 5×.
- **[P1-EMB-22] Prompt audit**: `scripts/check_prompts.py` → `artifacts/prompt_audit.json`.
- **[P1-EMB-23] Кэш-метрики**: hit/miss/evict, Prometheus + визуализация.
- **[P1-EMB-24] Vector rerank**: dot-product rerank после BM25F.

### Graph Intelligence
- **[P1-GPH-25] Rehydrate job**: массовый пересчёт связей, отчёт в S3.
- **[P1-GPH-26] Метрики графа**: avg_degree, clustering, expansion.
- **[P1-GPH-27] Weight aging**: экспоненциальное затухание, конфигурируемое.

### Orchestrator & GPT-5 когнитивные функции
- **[P1-ORC-28] Deterministic mode**: `deterministic=true` → reasoning=minimal, кеш по `inputs_hash`.
- **[P1-ORC-29] PII masker**: маскировка email/phone в логах/ответах.
- **[P1-ORC-30] Summaries with citations**: `summary`, `citations[]`, schema контрагента.
- **[P1-ORC-31] Cost manager**: динамическое понижение модели (nano→mini) при близости к бюджету, лог `downgrade_reason`.

### Quality & Monitoring
- **[P1-QLT-32] Quality monitor**: еженедельный отчёт P@5/MRR/nDCG (скрипт + артефакт).
- **[P1-QLT-33] Adaptive ranking weights**: offline tuning pipeline (vector/graph/importance) → сохранять профили.
- **[P1-QLT-34] Context clustering**: matryoshka-aware dedupe групп.

### Reliability & Safety
- **[P1-REL-35] Chaos test**: имитация отказа embedding/Neo4j; отчёт `artifacts/chaos.json`.
- **[P1-REL-36] Rate limiting**: глобальный и per-tool лимит из конфига.
- **[P1-REL-37] Health degrade**: fallback cache при частичных ошибках.
- **[P1-SEC-38] Audit trail**: сохранять `assembled_context.hash` + snapshot provenances.
- **[P1-SEC-39] GPT moderation**: фильтр OpenAI moderation + доменные правила.

### DX & Документация
- **[P1-DX-40] EmbeddingGemma handbook**: docs/embeddinggemma.md (best practices).
- **[P1-DX-41] Onboarding guide**: docs/getting_started.md (10-минутный сценарий).
- **[P1-DX-42] CLI цели**: `make determinism`, `make observability`, `make chaos`, `make cognitive-demo`.
- **[P1-DX-43] Cognitive demos**: скрипты примеров (`scripts/cognitive_demo_importance.py`, `...summarize.py`).

## 4. P2 — стратегические инициативы
- [P2-EMB-50] QAT (Q4_0/Q8_0) + автоматический бенчмаркинг.
- [P2-EMB-51] Edge профиль (CPU-only оптимизации).
- [P2-GPH-52] Advanced analytics (community detection, centrality) для рекомендаций.
- [P2-ORC-53] Async orchestration queue (idempotent tasks).
- [P2-QLT-54] Multi-vector adapters (FAISS, Chroma) — единый интерфейс.
- [P2-OBS-55] Grafana dashboards (JSON + инструкции).
- [P2-REL-56] Active/Active persistence (multi-region).
- [P2-DX-57] Автогенерация quality/cost отчётов (cron, Slack/email вывод).
- [P2-DX-58] Notebook smoke tests в CI.
- [P2-SEC-59] DLP сканирование на ingest (PII/secrets, redact).
- [P2-SEC-60] Подпись артефактов (cosign).
- [P2-OTH-61] Fine-tuning pipeline (MatryoshkaLoss) + eval.
- [P2-REL-62] Capacity planner (Neo4j, embedding) + алерты.
- [P2-REL-63] SLA dashboards (availability, latency, cost, evidence).
- [P2-UX-64] Cognitive coach: интерактивный туториал (CLI wizard) для новых агентов.

## 5. Definition of Done (каждая задача)
- `make verify` (включая determinism, evidence, quality) дважды подряд зелёный.
- Артефакты в `artifacts/` (determinism, evidence, quality, chaos, prompt audit, cognitive summary).
- Документация и README обновлены.
- Метрики и логи проверены (dashboards/alerts в рабочем состоянии).
- Нет новых предупреждений Clippy, Ruff, pytest.

## 6. Управление рисками
- **OpenAI стоимость**: монитор `gpt5_cost_usd_total`; alert ≥80% бюджета.
- **Neo4j рост**: регулярные rehydrate и aging (P1 задачи).
- **EmbeddingGemma обновления**: smoke-test новой версии до релиза.
- **Retried storms**: circuit breaker (P0-REL-16) с уведомлением в ops.

## 7. Ритуалы и прозрачность
- Ежедневный sync 09:00 UTC: статус P0/P1, блокеры, когнитивные метрики.
- Публичный `STATUS.md` (обновляется автоматически после verify) — TL;DR + детали.
- Блокировка >24h → эскалация Principal Engineer.
- Изменение приоритетов только после апдейта AGENTS.md и согласования с владельцем.

