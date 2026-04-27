.PHONY: up dev pull-models test lint clean

up:
	docker compose -f infra/docker-compose.yml up -d

dev:
	docker compose -f infra/docker-compose.yml -f infra/docker-compose.dev.yml up

pull-models:
	bash infra/ollama/pull_models.sh

test:
	pytest tests/ -v

lint:
	ruff check src/ && mypy src/

clean:
	docker compose -f infra/docker-compose.yml down -v
	rm -rf .cortex/
