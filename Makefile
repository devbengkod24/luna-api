.PHONY: help install dev migrate migration reset-db docker-up docker-down docker-logs test lint format

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Run development server
	uvicorn src.drivers.rest.main:app --reload --host 0.0.0.0 --port 8000

migrate: ## Run database migrations
	alembic upgrade head

migration: ## Create new migration (usage: make migration msg="description")
	alembic revision --autogenerate -m "$(msg)"

reset-db: ## Reset database (WARNING: deletes all data!)
	alembic downgrade base
	alembic upgrade head

docker-up: ## Start all services with Docker Compose
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-logs: ## View logs from all services
	docker compose logs -f

docker-build: ## Build Docker images
	docker compose build

test: ## Run tests
	pytest

lint: ## Run linter
	flake8 src/

format: ## Format code
	black src/
	isort src/

clean: ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

