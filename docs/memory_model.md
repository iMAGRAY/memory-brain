# Memory Model

- Fields: `timestamp_utc`, `agent_time`, `timezone`, `emotion`, `hypothesis`, `confidence`, `bias_indicator`.
- Relations: stored in Neo4j (`CAUSES`, `SUPPORTS`, `CONTRADICTS`, `REFERS_TO`).
- Dedup: identical `sha256` -> 409 `MEM_DUPLICATE`.
- Feedback adjusts trust/bias/emotion.
