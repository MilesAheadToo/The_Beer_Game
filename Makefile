SHELL := /bin/bash

# Default to localhost for local development
HOST = localhost
REMOTE_HOST = 172.29.20.187

# Default to CPU build (explicitly disable GPU)
FORCE_GPU ?= 0

# Docker build arguments
DOCKER_BUILD_ARGS = --build-arg FORCE_GPU=$(FORCE_GPU)

# Docker runtime arguments
DOCKER_RUN_ARGS = -e FORCE_GPU=$(FORCE_GPU)

.PHONY: up up-dev up-remote up-tls up-tls-only rebuild-frontend down ps logs seed reset-admin proxy-url help remote-train remote-train-dataset train-setup train-cpu

up:
	@echo "\n[+] Building and starting full system (proxy, frontend, backend, db)..."; \
	echo "   Build type: CPU (default)"; \
	docker compose build $(DOCKER_BUILD_ARGS) backend && \
	docker compose up -d proxy frontend backend db create-users; \
	echo "\n[✓] Local development server started (CPU mode)."; \
	echo "   URL:     http://$(HOST):8088"; \
	echo "   Admin:   admin@daybreak.ai / Daybreak@2025"

up-dev:
	@echo "\n[+] Building and starting full system with dev overrides (proxy, frontend, backend, db)..."; \
	echo "   Build type: CPU (default)"; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build $(DOCKER_BUILD_ARGS) backend && \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d proxy frontend backend db create-users; \
	echo "\n[✓] Local development server started with dev overrides (CPU mode)."; \
	echo "   URL:     http://$(HOST):8088"; \
	echo "   Admin:   admin@daybreak.ai / Daybreak@2025"

up-remote:
	@echo "\n[+] Building and starting full system for remote access..."; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build proxy frontend backend db create-users; \
	echo "\n[✓] Remote server started."; \
	echo "   URL:     http://$(REMOTE_HOST):8088"; \
	echo "   Admin:   admin@daybreak.ai / Daybreak@2025"; \
	echo "\n   For local development, use: make up-dev"

up-tls:
	@echo "\n[+] Building and starting full system with TLS proxy on 8443..."; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Local HTTPS server started (self-signed)."; \
	echo "   URL:     https://$(HOST):8443"; \
	echo "   Admin:   admin@daybreak.ai / Daybreak@2025"; \
	echo "\n   For remote HTTPS access, use: make up-remote-tls"

up-remote-tls:
	@echo "\n[+] Building and starting full system with TLS for remote access..."; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Remote HTTPS server started (self-signed)."; \
	echo "   URL:     https://$(REMOTE_HOST):8443"; \
	echo "   Admin:   admin@daybreak.ai / Daybreak@2025"

up-tls-only:
	@echo "\n[+] Starting TLS-only proxy (no HTTP proxy on 8088)..."; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Started. Open https://172.29.20.187:8443 in your browser (self-signed)."; \
	echo "   Admin login: admin@daybreak.ai / Daybreak@2025"

rebuild-frontend:
	@echo "\n[+] Rebuilding frontend image with dev overrides..."; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build frontend; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d frontend; \
	echo "\n[✓] Frontend rebuilt and restarted."

rebuild-backend:
	@echo "\n[+] Rebuilding backend image..."; \
	echo "   Build type: $(if $(filter true,$(GPU_ENABLED)),GPU,CPU)"; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build $(DOCKER_BUILD_ARGS) backend; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d backend; \
	echo "\n[✓] Backend rebuilt and restarted."

# GPU-specific targets
gpu-up gpu-up-dev:
	$(MAKE) $(subst gpu-,,$@) GPU_ENABLED=true

# CPU-specific targets
cpu-up cpu-up-dev:
	$(MAKE) $(subst cpu-,,$@) GPU_ENABLED=false

down:
	@echo "\n[+] Stopping and removing containers and volumes..."; \
	docker compose down -v

ps:
	@docker compose ps

logs:
	@docker compose logs -f --tail=200

seed:
	@echo "\n[+] Seeding default users..."; \
	docker compose run --rm create-users

reset-admin:
	@echo "\n[+] Resetting admin password to Daybreak@2025..."; \
	docker compose exec backend python scripts/reset_admin_password.py

proxy-url:
	@echo "Current host: $(HOST) (set with HOST=ip make ...)"; \
	echo "HTTP:  http://$(HOST):8088"; \
	echo "HTTPS: https://$(HOST):8443 (enable with: make up-tls)"; \
	echo "Login: admin@daybreak.ai / Daybreak@2025"; \
	echo "To change host: HOST=your-ip make ..."

help:
	@echo "Available targets:"; \
	echo ""; \
	echo "Local Development (CPU/GPU):"; \
	echo "  make up            - build/start HTTP proxy (8088), frontend, backend, db, seed (CPU by default)"; \
	echo "  make up FORCE_GPU=1 - enable GPU support if available"; \
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
	echo "  make seed          - run user seeder (admin and role users)"; \
	echo "  make reset-admin   - reset admin password to Daybreak@2025"; \
	echo "  make proxy-url     - print URLs and login info"; \
	echo ""; \
	echo "Advanced Training:"; \
	echo "  make train-setup   - create Python venv and install training deps"; \
	echo "  make train-cpu     - run local CPU training"; \
	echo "  make train-gpu     - run local GPU training"; \
	echo "  make remote-train  - train on remote server"; \
	echo ""; \
	echo "Environment Variables:"; \
	echo "  GPU_ENABLED=true   - Enable GPU support (e.g., make up GPU_ENABLED=true)";

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
