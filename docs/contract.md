# MCP Contract (v1)

- Base URL: `http://localhost:8080`
- Accept-Version: header or `?version=1`
- Determinism: identical requests -> identical `context_hash`.
- Feature flags: `/mcp/feature-flags` returns runtime values.

## Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| POST | /mcp/memory | Store memory item. |
| POST | /mcp/query | Retrieve memories (summary/details). |
| POST | /mcp/similarity | Vector search. |
| GET  | /mcp/timeline | Chronological feed. |
| POST | /mcp/time/resolve | Resolve expressions. |
| POST | /mcp/session/save | Save agent snapshot. |
| GET  | /mcp/session/restore | Restore snapshot. |
| POST | /mcp/plan | Manage plans. |
| POST | /mcp/error-log | Record errors. |
| POST | /mcp/feedback | Adjust trust/bias. |
| GET  | /mcp/summary | Aggregate state. |
| GET  | /mcp/trust | Trust distribution. |
| GET  | /mcp/timeline/analytics | Timeline metrics. |
| POST | /mcp/maintenance/run | Compaction (flag). |

## Error response
```json
{
  "error": {
    "code": "MEM_DUPLICATE",
    "message": "Memory already exists",
    "error_policy_code": "USE_EXISTING_MEMORY",
    "suggested_action": "GET /mcp/memory/{existing_id}"
  }
}
```
