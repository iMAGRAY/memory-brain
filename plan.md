# plan.md — текущий план внедрения maintenance и алиасов

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
