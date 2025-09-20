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

# Default configuration name and training parameters (overridable via environment)
CONFIG_NAME ?= Default TBG

SIMPY_NUM_RUNS ?= 64
SIMPY_TIMESTEPS ?= 64
SIMPY_WINDOW ?= 12
SIMPY_HORIZON ?= 1

TRAIN_EPOCHS ?= 10
TRAIN_WINDOW ?= 12
TRAIN_HORIZON ?= 1
TRAIN_DEVICE ?= cuda

# Prefer the modern Docker Compose plugin when available, but allow overriding.
DOCKER ?= docker
DOCKER_COMPOSE ?= $(shell if command -v $(DOCKER) >/dev/null 2>&1 && $(DOCKER) compose version >/dev/null 2>&1; then echo "$(DOCKER) compose"; elif command -v docker-compose >/dev/null 2>&1; then echo "docker-compose"; else echo "$(DOCKER) compose"; fi)

# Compose V1 (the standalone docker-compose binary) is incompatible with newer
# Docker Engine releases because the Engine no longer exposes the legacy
# ContainerConfig field in its API. Detect the version early and, when we fall
# back to docker-compose, downgrade the API version so the helper keeps working.
COMPOSE_VERSION := $(shell $(DOCKER_COMPOSE) version --short 2>/dev/null)
COMPOSE_VERSION_NORMALIZED := $(patsubst v%,%,$(COMPOSE_VERSION))
COMPOSE_IS_V1 := 0
ifeq ($(firstword $(DOCKER_COMPOSE)),docker-compose)
    COMPOSE_IS_V1 := 1
else ifneq ($(COMPOSE_VERSION_NORMALIZED),)
    ifneq (,$(filter 1.%,$(COMPOSE_VERSION_NORMALIZED)))
        COMPOSE_IS_V1 := 1
    endif
endif

COMPOSE_ENV :=
ifeq ($(COMPOSE_IS_V1),1)
    COMPOSE_ENV := COMPOSE_API_VERSION=1.44 DOCKER_API_VERSION=1.44
endif

DOCKER_COMPOSE_CMD = $(strip $(COMPOSE_ENV) $(DOCKER_COMPOSE))

.PHONY: up gpu-up up-dev down ps logs seed reset-admin help init-env proxy-up proxy-down proxy-restart proxy-recreate proxy-logs proxy-url seed-default-group build-create-users db-bootstrap db-reset

# Default CPU target
up:
	@echo "\n[+] Building and starting full system (proxy, frontend, backend, db)..."; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml build --no-cache $(DOCKER_BUILD_ARGS_CPU) backend && \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml up -d proxy frontend backend db create-users && \
	if [ "$(FORCE_GPU)" = "1" ]; then \
		$(MAKE) --no-print-directory db-bootstrap; \
	fi; \
	mode_label="CPU"; \
	if [ "$(FORCE_GPU)" = "1" ]; then mode_label="GPU"; fi; \
	echo "\n[✓] Local development server started (${mode_label} mode)."; \
	echo "   URL:     http://$(HOST):8088"; \
	echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"

# GPU target
gpu-up:
	@echo "\n[+] Building and starting full system in GPU mode (proxy, frontend, gpu-backend, db)..."; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache backend && \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.gpu.yml up -d proxy frontend backend db create-users; \
	echo "\n[✓] Local development server started (GPU mode)."; \
	echo "   URL:     http://$(HOST):8088"; \
	echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"; \
	echo "   GPU:     $(shell nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"

up-dev:
	@echo "\n[+] Building and starting full system with dev overrides (proxy, frontend, backend, db)..."; \
	echo "   Build type: CPU (default)"; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml build $(DOCKER_BUILD_ARGS) backend && \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml up -d proxy frontend backend db create-users; \
	echo "\n[✓] Local development server started with dev overrides (CPU mode)."; \
	echo "   URL:     http://$(HOST):8088"; \
	echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"

up-remote:
	@echo "\n[+] Building and starting full system for remote access..."; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml up -d --build proxy frontend backend db create-users; \
	echo "\n[✓] Remote server started."; \
	echo "   URL:     http://$(REMOTE_HOST):8088"; \
	echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"; \
	echo "\n   For local development, use: make up-dev"

up-tls:
	@echo "\n[+] Building and starting full system with TLS proxy on 8443..."; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Local HTTPS server started (self-signed)."; \
	echo "   URL:     https://$(HOST):8443"; \
	echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"; \
	echo "\n   For remote HTTPS access, use: make up-remote-tls"

up-remote-tls:
	@echo "\n[+] Building and starting full system with TLS for remote access..."; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Remote HTTPS server started (self-signed)."; \
	echo "   URL:     https://$(REMOTE_HOST):8443"; \
	echo "   SystemAdmin: systemadmin@daybreak.ai / Daybreak@2025"

up-tls-only:
	@echo "\n[+] Starting TLS-only proxy (no HTTP proxy on 8088)..."; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Started. Open https://172.29.20.187:8443 in your browser (self-signed)."; \
	echo "   SystemAdmin login: systemadmin@daybreak.ai / Daybreak@2025"

rebuild-frontend:
	@echo "\n[+] Rebuilding frontend image with dev overrides..."; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml build frontend; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml up -d frontend; \
	echo "\n[✓] Frontend rebuilt and restarted."

rebuild-backend:
	@echo "\n[+] Rebuilding backend image..."; \
	echo "   Build type: $(if $(filter 1,$(FORCE_GPU)),GPU,CPU)"; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml build $(DOCKER_BUILD_ARGS) backend; \
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.dev.yml up -d backend; \
	echo "\n[✓] Backend rebuilt and restarted."

# GPU-specific targets
gpu-up gpu-up-dev:
	$(MAKE) $(subst gpu-,,$@) FORCE_GPU=1

# CPU-specific targets
cpu-up cpu-up-dev:
	$(MAKE) $(subst cpu-,,$@) FORCE_GPU=0

down:
	@echo "\n[+] Stopping and removing containers and volumes..."; \
	$(DOCKER_COMPOSE_CMD) down -v

ps:
	@$(DOCKER_COMPOSE_CMD) ps

logs:
	@$(DOCKER_COMPOSE_CMD) logs -f --tail=200

# Proxy management
proxy-up:
	@echo "\n[+] Starting proxy service..."
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.proxy.yml up -d --no-deps proxy

proxy-down:
	@echo "\n[+] Stopping proxy service..."
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.proxy.yml stop proxy

proxy-clean:
	@echo "\n[+] Removing proxy container..."
	-$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.proxy.yml rm -f proxy

proxy-restart: proxy-down proxy-up

proxy-recreate: proxy-clean
	@echo "\n[+] Recreating proxy service with a fresh container..."
	$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.proxy.yml up -d --build proxy

proxy-logs:
	@$(DOCKER_COMPOSE_CMD) -f docker-compose.yml -f docker-compose.proxy.yml logs -f --tail=200 proxy

seed:
	@echo "\n[+] Seeding default users..."; \
	$(DOCKER_COMPOSE_CMD) run --rm create-users

build-create-users:
	@echo "\n[+] Rebuilding lightweight seeding image..."; \
	pull_flag=""; \
	if [ -n "$(PULL)" ]; then pull_flag="--pull"; fi; \
	$(DOCKER_COMPOSE_CMD) build $$pull_flag create-users; \
	echo "\n[✓] create-users image refreshed."; \
	echo "    Hint: leave requirements*.txt untouched to maximise Docker build caching."

db-bootstrap:
	@echo "\n[+] Bootstrapping Daybreak defaults (config, users, training, showcase games)..."; \
	$(DOCKER_COMPOSE_CMD) exec backend python3 scripts/seed_default_group.py

db-reset:
	@echo "\n[+] Resetting games and rebuilding Daybreak training artifacts..."; \
	$(DOCKER_COMPOSE_CMD) exec backend python3 scripts/seed_default_group.py --reset-games

seed-default-group:
	@$(MAKE) --no-print-directory db-bootstrap

reset-admin:
	@echo "\n[+] Resetting superadmin password to Daybreak@2025..."; \
	$(DOCKER_COMPOSE_CMD) exec backend python scripts/reset_admin_password.py

setup-default-env:
	@echo "\n[+] Setting up default environment..."; \
	$(DOCKER_COMPOSE_CMD) exec backend python scripts/setup_default_environment.py

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
	echo "  make db-bootstrap  - create default config, users, training data, and Daybreak games"; \
	echo "  make db-reset      - delete games then rerun Daybreak bootstrap"; \
	echo "  make proxy-up      - start or restart only the proxy container"; \
	echo "  make proxy-recreate - force-rebuild the proxy container without touching deps"; \
	echo "  make proxy-logs    - tail proxy logs"; \
	echo "  make seed          - run user seeder (system administrator user)"; \
	echo "  make reset-admin   - reset system administrator password to Daybreak@2025"; \
	echo "  make proxy-url     - print URLs and login info"; \
	echo "  make init-env      - set up .env from template or host-specific file"; \
	echo ""; \
	echo "Advanced Training:"; \
	echo "  make train-setup   - create Python venv and install training deps"; \
	echo "  make train-cpu     - run local CPU training"; \
	echo "  make train-gpu     - run local GPU training"; \
	echo "  make generate-simpy-data - exec backend task to build SimPy dataset"; \
	echo "  make train-default-gpu   - exec backend task to train default model on GPU"; \
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

generate-simpy-data:
	@echo "\n[+] Generating SimPy training dataset inside backend container..."; \
	set -e; \
	force_flag=""; \
	if [ -n "$(SIMPY_FORCE)" ]; then force_flag="--force"; fi; \
	$(DOCKER_COMPOSE_CMD) exec backend python scripts/training/generate_simpy_dataset.py \
	  --config-name "$(CONFIG_NAME)" \
	  --num-runs $(SIMPY_NUM_RUNS) \
	  --timesteps $(SIMPY_TIMESTEPS) \
	  --window $(SIMPY_WINDOW) \
	  --horizon $(SIMPY_HORIZON) \
	  $$force_flag
	@echo "\n[✓] Dataset generation task completed."

train-default-gpu:
	@echo "\n[+] Training default Daybreak agent with GPU inside backend container..."; \
	set -e; \
	dataset_flag=""; \
	force_flag=""; \
	if [ -n "$(TRAIN_DATASET)" ]; then dataset_flag="--dataset $(TRAIN_DATASET)"; fi; \
	if [ -n "$(TRAIN_FORCE)" ]; then force_flag="--force"; fi; \
	$(DOCKER_COMPOSE_CMD) exec backend python scripts/training/train_gpu_default.py \
	  --config-name "$(CONFIG_NAME)" \
	  --device "$(TRAIN_DEVICE)" \
	  --epochs $(TRAIN_EPOCHS) \
	  --window $(TRAIN_WINDOW) \
	  --horizon $(TRAIN_HORIZON) \
	  $$dataset_flag $$force_flag
	@echo "\n[✓] GPU training task completed."
