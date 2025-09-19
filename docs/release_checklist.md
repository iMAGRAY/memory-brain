# Release Checklist
1. make start-dev
2. make agent-test baseline continuity session errors similarity
3. make verify-agent
4. ENABLE_TLDR=1 ENABLE_VECTOR_INDEX=1 make agent-test human plan conflict
5. make maintenance-run
6. make quality-report
7. make status-export
8. Review artifacts
9. Tag docker images
10. make stop-dev
