.PHONY: up dev demo pull-models pull-models-demo test lint clean

up:
	docker compose -f infra/docker-compose.yml up -d

dev:
	docker compose -f infra/docker-compose.yml -f infra/docker-compose.dev.yml up

demo:
	docker compose -f infra/docker-compose.demo.yml up -d

pull-models:
	bash infra/ollama/pull_models.sh

pull-models-demo:
	bash infra/ollama/pull_models_demo.sh

test:
	pytest tests/ -v

lint:
	ruff check src/ && mypy src/

clean:
	docker compose -f infra/docker-compose.yml down -v
	rm -rf .cortex/
