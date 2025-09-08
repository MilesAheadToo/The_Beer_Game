SHELL := /bin/bash

.PHONY: up up-dev up-tls up-tls-only rebuild-frontend down ps logs seed reset-admin proxy-url help

up:
	@echo "\n[+] Building and starting full system (proxy, frontend, backend, db)..."; \
	docker compose up -d --build proxy frontend backend db create-users; \
	echo "\n[✓] Started. Open http://localhost:8088 in your browser."; \
	echo "   Admin login: admin@daybreak.ai / Daybreak@2025"

up-dev:
	@echo "\n[+] Building and starting full system with dev overrides (proxy, frontend, backend, db)..."; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build proxy frontend backend db create-users; \
	echo "\n[✓] Started. Open http://localhost:8088 in your browser."; \
	echo "   Admin login: admin@daybreak.ai / Daybreak@2025"

up-tls:
	@echo "\n[+] Building and starting full system with TLS proxy on 8443..."; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Started. Open https://localhost:8443 in your browser (self-signed)."; \
	echo "   Admin login: admin@daybreak.ai / Daybreak@2025"

up-tls-only:
	@echo "\n[+] Starting TLS-only proxy (no HTTP proxy on 8088)..."; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile tls up -d --build frontend backend db proxy-tls create-users; \
	echo "\n[✓] Started. Open https://localhost:8443 in your browser (self-signed)."; \
	echo "   Admin login: admin@daybreak.ai / Daybreak@2025"

rebuild-frontend:
	@echo "\n[+] Rebuilding frontend image with dev overrides..."; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build frontend; \
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d frontend; \
	echo "\n[✓] Frontend rebuilt and restarted."

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
	@echo "HTTP:  http://localhost:8088"; \
	echo "HTTPS: https://localhost:8443 (enable with: make up-tls)"; \
	echo "Login: admin@daybreak.ai / Daybreak@2025"

help:
	@echo "Available targets:"; \
	echo "  make up            - build/start HTTP proxy (8088), frontend, backend, db, seed"; \
	echo "  make up-dev        - same as up, with dev overrides"; \
	echo "  make up-tls        - build/start TLS proxy (8443), frontend, backend, db, seed"; \
	echo "  make up-tls-only   - start TLS-only proxy (no HTTP proxy)"; \
	echo "  make rebuild-frontend - rebuild and restart only frontend (with dev overrides)"; \
	echo "  make down          - stop and remove containers and volumes"; \
	echo "  make ps            - show container status"; \
	echo "  make logs          - tail logs"; \
	echo "  make seed          - run user seeder (admin and role users)"; \
	echo "  make reset-admin   - reset admin password to Daybreak@2025"; \
	echo "  make proxy-url     - print URLs and login info"
