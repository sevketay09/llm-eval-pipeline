TAG ?= 1.0.0
IMAGE_NAME ?= llm-eval-app
BASE_IMAGE ?= python:3.12-slim
COMPOSE_FILE ?= docker-compose.yml
DEBUG_COMPOSE_FILE ?= docker-compose.debug.yml
SERVICE_NAME ?= llm-eval-dashboard
LOCAL_UID ?= $(shell id -u)
LOCAL_GID ?= $(shell id -g)
PYTHON_BIN ?= ./.venv/bin/python
UVICORN_APP ?= api.main:app
API_HOST ?= 0.0.0.0
API_PORT ?= 8001
WEB_DIR ?= web
WEB_HOST ?= 0.0.0.0
WEB_PORT ?= 5173

.PHONY: help install-requirements clean docker-build docker-run docker-push \
	build-debug up-debug down-debug tail-logs restart-debug start-debug clear-results \
	dev dev-backend dev-frontend build-frontend preview-frontend check-api demo demo-docker

help:
	@echo "Available targets:"
	@echo "  install-requirements - Install Python dependencies"
	@echo "  demo                 - Offline demo eval with the mock model (no API keys)"
	@echo "  demo-docker          - Same demo eval inside the Docker image"
	@echo "  dev                  - Run FastAPI backend and Vite frontend together"
	@echo "  dev-backend          - Run FastAPI backend with reload"
	@echo "  dev-frontend         - Run Vite frontend dev server"
	@echo "  build-frontend       - Create production frontend build"
	@echo "  preview-frontend     - Preview the frontend production build"
	@echo "  check-api            - Ping the local FastAPI health endpoint"
	@echo "  clean                - Clean Python build/cache artifacts"
	@echo "  docker-build         - Build Docker image"
	@echo "  docker-run           - Run Docker image locally on port 8001"
	@echo "  docker-push          - Push Docker image"
	@echo "  build-debug          - Build debug compose service"
	@echo "  up-debug             - Start API + built frontend container"
	@echo "  down-debug           - Stop debug service"
	@echo "  tail-logs            - Tail debug service logs"
	@echo "  restart-debug        - Restart debug service"
	@echo "  start-debug          - down + build + up"

install-requirements:
	pip install -r requirements.txt

demo:
	python main.py --models demo-model --suite smoke --judge demo-model

demo-docker:
	docker compose build llm-eval-dashboard
	docker run --rm --user $$(id -u):$$(id -g) -e HOME=/tmp \
		-v $(CURDIR):/app -w /app \
		llm-eval-pipeline-llm-eval-dashboard:latest \
		python main.py --models demo-model --suite smoke --judge demo-model

dev:
	@trap 'kill 0' EXIT INT TERM; \
	@set -a && . ./.env && set +a && \
	trap 'kill 0' EXIT INT TERM; \
	$(PYTHON_BIN) -m uvicorn $(UVICORN_APP) --host $(API_HOST) --port $(API_PORT) --reload & \
	cd $(WEB_DIR) && npm run dev -- --host $(WEB_HOST) --port $(WEB_PORT)

dev-backend:
	@set -a && . ./.env && set +a && \
	$(PYTHON_BIN) -m uvicorn $(UVICORN_APP) --host $(API_HOST) --port $(API_PORT) --reload

dev-frontend:
	cd $(WEB_DIR) && npm run dev -- --host $(WEB_HOST) --port $(WEB_PORT)

build-frontend:
	cd $(WEB_DIR) && npm run build

preview-frontend:
	cd $(WEB_DIR) && npm run preview -- --host $(WEB_HOST) --port $(WEB_PORT)

check-api:
	curl -sf http://127.0.0.1:$(API_PORT)/api/health

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build --build-arg BASE_IMAGE=$(BASE_IMAGE) -t $(IMAGE_NAME):$(TAG) .

docker-run:
	docker run --rm -it -p 8001:8001 --env-file .env $(IMAGE_NAME):$(TAG)

docker-push:
	docker push $(IMAGE_NAME):$(TAG)

build-debug:
	BASE_IMAGE=$(BASE_IMAGE) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) docker compose -f $(DEBUG_COMPOSE_FILE) build

up-debug: down-debug
	BASE_IMAGE=$(BASE_IMAGE) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) docker compose -f $(DEBUG_COMPOSE_FILE) up $(SERVICE_NAME) -d

down-debug:
	BASE_IMAGE=$(BASE_IMAGE) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) docker compose -f $(DEBUG_COMPOSE_FILE) down

tail-logs:
	BASE_IMAGE=$(BASE_IMAGE) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) docker compose -f $(DEBUG_COMPOSE_FILE) logs -f $(SERVICE_NAME)

clear-results:
	@bash clear_cache.sh

restart-debug: down-debug clear-results up-debug

start-debug: down-debug build-debug up-debug
