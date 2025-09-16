SHELL := /bin/bash

ifeq ($(OS),Windows_NT)
    SETUP_ENV := powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
else
    SETUP_ENV := bash scripts/setup_env.sh
endif

# Default to localhost for local development
HOST = localhost
REMOTE_HOST = 172.29.20.187

# Default to CPU build (explicitly disable GPU)
FORCE_GPU ?= 0

# Docker build arguments for CPU
DOCKER_BUILD_ARGS_CPU = --build-arg FORCE_GPU=0

# Docker build arguments for GPU
DOCKER_BUILD_ARGS_GPU = --build-arg FORCE_GPU=1

# Docker runtime arguments
DOCKER_RUN_ARGS = -e FORCE_GPU=$(FORCE_GPU)

.PHONY: up gpu-up up-dev down ps logs seed reset-admin help init-env

# Default CPU target
up:
	@echo "\n[+] Building and starting full system in CPU mode (proxy, frontend, backend, db)..."; \
	docker-compose -f docker-compose.yml build --no-cache $(DOCKER_BUILD_ARGS_CPU) backend && \
	docker-compose -f docker-compose.yml up -d proxy frontend backend db create-users; \
	echo "\n[✓] Local development server started (CPU mode)."; \
	echo "   URL:     http://$(HOST):8088"; \
        echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"

# GPU target
gpu-up:
	@echo "\n[+] Building and starting full system in GPU mode (proxy, frontend, gpu-backend, db)..."; \
	docker-compose -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache backend && \
	docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d proxy frontend backend db create-users; \
	echo "\n[✓] Local development server started (GPU mode)."; \
	echo "   URL:     http://$(HOST):8088"; \
        echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"; \
	echo "   GPU:     $(shell nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"

up-dev:
	@echo "\n[+] Building and starting full system with dev overrides (proxy, frontend, backend, db)..."; \
	echo "   Build type: CPU (default)"; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build $(DOCKER_BUILD_ARGS) backend && \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d proxy frontend backend db create-users; \
	echo "\n[✓] Local development server started with dev overrides (CPU mode)."; \
	echo "   URL:     http://$(HOST):8088"; \
        echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"

up-remote:
	@echo "\n[+] Building and starting full system for remote access..."; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build proxy frontend backend db create-users; \
	echo "\n[✓] Remote server started."; \
	echo "   URL:     http://$(REMOTE_HOST):8088"; \
        echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"; \
	echo "\n   For local development, use: make up-dev"

up-tls:
	@echo "\n[+] Building and starting full system with TLS proxy on 8443..."; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Local HTTPS server started (self-signed)."; \
	echo "   URL:     https://$(HOST):8443"; \
        echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"; \
	echo "\n   For remote HTTPS access, use: make up-remote-tls"

up-remote-tls:
	@echo "\n[+] Building and starting full system with TLS for remote access..."; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Remote HTTPS server started (self-signed)."; \
	echo "   URL:     https://$(REMOTE_HOST):8443"; \
        echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"

up-tls-only:
	@echo "\n[+] Starting TLS-only proxy (no HTTP proxy on 8088)..."; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Started. Open https://172.29.20.187:8443 in your browser (self-signed)."; \
        echo "   SystemAdmin login: systemadmin@daybreak.ai / Daybreak@2025"

rebuild-frontend:
	@echo "\n[+] Rebuilding frontend image with dev overrides..."; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build frontend; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d frontend; \
	echo "\n[✓] Frontend rebuilt and restarted."

rebuild-backend:
	@echo "\n[+] Rebuilding backend image..."; \
	echo "   Build type: $(if $(filter 1,$(FORCE_GPU)),GPU,CPU)"; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build $(DOCKER_BUILD_ARGS) backend; \
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d backend; \
	echo "\n[✓] Backend rebuilt and restarted."

# GPU-specific targets
gpu-up gpu-up-dev:
	$(MAKE) $(subst gpu-,,$@) FORCE_GPU=1

# CPU-specific targets
cpu-up cpu-up-dev:
	$(MAKE) $(subst cpu-,,$@) FORCE_GPU=0

down:
	@echo "\n[+] Stopping and removing containers and volumes..."; \
	docker-compose down -v

ps:
	@docker-compose ps

logs:
	@docker-compose logs -f --tail=200

# Proxy management
proxy-up:
	@echo "\n[+] Starting proxy service..."
	docker-compose -f docker-compose.yml up -d proxy

proxy-down:
	@echo "\n[+] Stopping proxy service..."
	docker-compose -f docker-compose.yml stop proxy

proxy-restart: proxy-down proxy-up

proxy-logs:
	@docker-compose logs -f --tail=200 proxy

seed:
	@echo "\n[+] Seeding default users..."; \
	docker-compose run --rm create-users

reset-admin:
	@echo "\n[+] Resetting superadmin password to Daybreak@2025..."; \
	docker compose exec backend python scripts/reset_admin_password.py

proxy-url:
	@echo "Current host: $(HOST) (set with HOST=ip make ...)"; \
	echo "HTTP:  http://$(HOST):8088"; \
	echo "HTTPS: https://$(HOST):8443 (enable with: make up-tls)"; \
        echo "Login: systemadmin@daybreak.ai / Daybreak@2025"; \
	echo "To change host: HOST=your-ip make ..."

help:
	@echo "Available targets:"; \
	echo ""; \
	echo "Local Development (CPU/GPU):"; \
	echo "  make up            - build/start HTTP proxy (8088), frontend, backend, db, seed (CPU mode)"; \
	echo "  make gpu-up        - build/start with GPU support (requires NVIDIA Docker)"; \
	echo "  make up FORCE_GPU=1 - enable GPU support if available (set to 1 to enable)"; \
	echo "  make up-dev        - same as up, with dev overrides"; \
	echo "  make up-tls        - start with HTTPS (8443) using self-signed cert"; \
	echo ""; \
	echo "GPU-Specific Commands:"; \
	echo "  make up FORCE_GPU=1 - build/start with GPU support"; \
	echo "  make up-dev FORCE_GPU=1 - same as above, with dev overrides"; \
	echo "  make up-tls FORCE_GPU=1 - start with HTTPS and GPU support"; \
	echo ""; \
	echo "Remote Server:"; \
	echo "  make up-remote     - start server for remote access (HTTP 8088)"; \
	echo "  make up-remote-tls - start with HTTPS (8443) for remote access"; \
	echo ""; \
	echo "Common Operations:"; \
	echo "  make down          - stop and remove containers and volumes"; \
	echo "  make ps            - show container status"; \
	echo "  make logs          - tail logs"; \
	echo "  make rebuild-frontend - rebuild and restart only frontend"; \
	echo "  make rebuild-backend  - rebuild and restart only backend"; \
        echo "  make seed          - run user seeder (system administrator user)"; \
        echo "  make reset-admin   - reset system administrator password to Daybreak@2025"; \
        echo "  make proxy-url     - print URLs and login info"; \
        echo "  make init-env      - set up .env from template or host-specific file"; \
        echo ""; \
        echo "Advanced Training:"; \
	echo "  make train-setup   - create Python venv and install training deps"; \
	echo "  make train-cpu     - run local CPU training"; \
	echo "  make train-gpu     - run local GPU training"; \
	echo "  make remote-train  - train on remote server"; \
	echo ""; \
	echo "Environment Variables:"; \
        echo "  FORCE_GPU=1        - Enable GPU support (e.g., make up FORCE_GPU=1)";

init-env:
	@$(SETUP_ENV)

# Remote training wrappers (see scripts/remote_train.sh for full help)
REMOTE        ?=
REMOTE_DIR    ?= ~/beer-game
EPOCHS        ?= 50
DEVICE        ?= cuda
WINDOW        ?= 12
HORIZON       ?= 1
NUM_RUNS      ?= 64
T             ?= 64
DATASET       ?=
SAVE_LOCAL    ?= backend/checkpoints/supply_chain_gnn.pth

remote-train:
	@if [ -z "$(REMOTE)" ]; then echo "REMOTE is required, e.g. make remote-train REMOTE=user@host"; exit 1; fi; \
	bash scripts/remote_train.sh \
	  --remote "$(REMOTE)" \
	  --remote-dir "$(REMOTE_DIR)" \
	  --epochs "$(EPOCHS)" \
	  --device "$(DEVICE)" \
	  --window "$(WINDOW)" \
	  --horizon "$(HORIZON)" \
	  --num-runs "$(NUM_RUNS)" \
	  --T "$(T)" \
	  --save-local "$(SAVE_LOCAL)"

remote-train-dataset:
	@if [ -z "$(REMOTE)" ]; then echo "REMOTE is required, e.g. make remote-train-dataset REMOTE=user@host DATASET=..."; exit 1; fi; \
	if [ -z "$(DATASET)" ]; then echo "DATASET is required for remote-train-dataset"; exit 1; fi; \
	bash scripts/remote_train.sh \
	  --remote "$(REMOTE)" \
	  --remote-dir "$(REMOTE_DIR)" \
	  --epochs "$(EPOCHS)" \
	  --device "$(DEVICE)" \
	  --dataset "$(DATASET)" \
	  --save-local "$(SAVE_LOCAL)"

# Local training helpers
train-setup:
	@echo "\n[+] Setting up local training environment (venv + deps)..."; \
	cd backend && bash scripts/setup_training_env.sh

train-cpu:
	@echo "\n[+] Running local CPU training..."; \
	cd backend && bash run_training.sh

train-gpu:
	@echo "\n[+] Running local GPU training..."; \
	cd backend && bash run_training_gpu.sh
