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

Дополнительно (live‑метрики):
- [x] Добавить scripts/metrics_collector.py (стрим /metrics + /stats → JSONL)
- [x] Добавить scripts/quality_stream.py (периодическая оценка P@k/MRR/nDCG)
- [x] Интегрировать опциональный запуск коллекторов в verify (ENABLE_METRICS_STREAM=1)
