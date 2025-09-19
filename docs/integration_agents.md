# Integration Snippets
````bash
curl -s http://localhost:8080/.well-known/mcp-spec | jq

curl -s -X POST http://localhost:8080/mcp/memory \
  -H 'Content-Type: application/json' \
  -d '{"content":{"text":"Sync"},"timestamp_utc":"2025-09-19T12:00:00Z"}'
````
