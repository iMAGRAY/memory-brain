# Time Model
- UTC anchor (`timestamp_utc`) + agent-local time (`agent_time`).
- `/mcp/time/resolve` accepts expressions like `now + PT30M`.
- Timeline sorts by UTC then sha256.
- `timeline_hash` anchors pagination and session snapshots.
