TAG ?= 1.0.0
IMAGE_NAME ?= llm-eval-dashboard
BASE_IMAGE ?= python:3.12-slim
COMPOSE_FILE ?= docker-compose.yml
DEBUG_COMPOSE_FILE ?= docker-compose.debug.yml
SERVICE_NAME ?= llm-eval-dashboard
LOCAL_UID ?= $(shell id -u)
LOCAL_GID ?= $(shell id -g)

.PHONY: help install-requirements clean docker-build docker-run docker-push \
	build-debug up-debug down-debug tail-logs restart-debug start-debug clear-results

help:
	@echo "Available targets:"
	@echo "  install-requirements - Install Python dependencies"
	@echo "  clean                - Clean Python build/cache artifacts"
	@echo "  docker-build         - Build Docker image"
	@echo "  docker-run           - Run Docker image locally"
	@echo "  docker-push          - Push Docker image"
	@echo "  build-debug          - Build debug compose service"
	@echo "  up-debug             - Start debug service"
	@echo "  down-debug           - Stop debug service"
	@echo "  tail-logs            - Tail debug service logs"
	@echo "  restart-debug        - Restart debug service"
	@echo "  start-debug          - down + build + up"

install-requirements:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build --build-arg BASE_IMAGE=$(BASE_IMAGE) -t $(IMAGE_NAME):$(TAG) .

docker-run:
	docker run --rm -it -p 8501:8501 --env-file .env $(IMAGE_NAME):$(TAG)

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
