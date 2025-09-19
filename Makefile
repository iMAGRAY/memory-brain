DC=docker compose -f docker-compose.pro.yml
PY=python3

.PHONY: start-dev stop-dev status-dev logs-dev restart-% config-check docs-check \
        agent-test-% verify-agent quality-report maintenance-run status-export \
        observability-enable observability-disable

start-dev: config-check
	$(DC) up -d --build

stop-dev:
	$(DC) down -v

status-dev:
	$(DC) ps

logs-dev:
	$(DC) logs memory

restart-%:
	$(DC) restart $*

config-check:
	$(PY) scripts/config_check.py

docs-check:
	$(PY) scripts/docs_check.py

agent-test-%:
	$(PY) simulator/run.py $*

verify-agent: agent-test-continuity agent-test-session
	$(PY) simulator/run.py errors

quality-report:
	$(PY) scripts/quality_report.py

maintenance-run:
	$(PY) scripts/maintenance.py

status-export:
	$(PY) scripts/status_export.py

observability-enable:
	$(DC) --profile observability up -d

observability-disable:
	$(DC) --profile observability down
