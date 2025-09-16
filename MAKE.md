# Make Targets Overview

This project uses a Docker-based workflow for local development, remote deployment, and
model training. The table below outlines the most common `make` targets and when to use
them.

| Target | Description |
| --- | --- |
| `make up` | Build and start the proxy, frontend, backend, database, and seed containers in CPU mode. Exposes HTTP on `http://localhost:8088`. |
| `make gpu-up` | Same as `make up`, but builds the backend with GPU support enabled (requires NVIDIA Docker). |
| `make up-dev` | Start the stack using the development overrides in `docker-compose.dev.yml`. |
| `make up-remote` | Bring up the stack for remote access using the default compose file plus dev overrides; exposes HTTP at `http://172.29.20.187:8088`. |
| `make up-tls` | Start the stack with the TLS profile enabled. Uses the self-signed certificate and listens on `https://localhost:8443`. |
| `make up-remote-tls` | Start the TLS-enabled stack for remote access on `https://172.29.20.187:8443`. |
| `make up-tls-only` | Start only the TLS proxy (no HTTP proxy on port 8088). |
| `make rebuild-frontend` | Rebuild just the frontend image and restart the container. |
| `make rebuild-backend` | Rebuild just the backend image and restart the container. Honors `FORCE_GPU`. |
| `make down` | Stop and remove all containers and named volumes. |
| `make ps` | Show the status of the running compose services. |
| `make logs` | Follow the combined logs (tail 200 lines) for all services. |
| `make seed` | Seed default users via the `create-users` service. |
| `make reset-admin` | Reset the SystemAdmin password to `Daybreak@2025`. |
| `make proxy-url` | Print the HTTP/HTTPS URLs and default credentials. |
| `make init-env` | Run the platform-specific environment setup script to generate `.env` files. |

## Docker Compose Files

The Make targets wrap a small collection of Compose files so you can mix and match stacks
for development, production, or targeted services. Combine files with Docker's `-f`
flag as needed (the base `docker-compose.yml` is used automatically when you run
`docker-compose up`).

| File | Purpose | Example usage |
| --- | --- | --- |
| `docker-compose.yml` | Core development stack providing the nginx proxy, React frontend, FastAPI backend, MariaDB database, phpMyAdmin, and the `create-users` seeding container. Acts as the base file for overrides. | `docker-compose up` or `make up` |
| `docker-compose.dev.yml` | Development overrides that pin the frontend API URLs to relative paths, expose an optional TLS proxy via the `tls` profile, and surface runtime hooks for GPU/CPU switching through `FORCE_GPU`. | `docker-compose -f docker-compose.yml -f docker-compose.dev.yml up` |
| `docker-compose.gpu.yml` | Rebuilds the backend image with `Dockerfile.gpu`, enables the NVIDIA runtime, and requests a GPU device. Layer it on top of the base file when a GPU is available. | `docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up` |
| `docker-compose.prod.yml` | Minimal production deployment with MariaDB, a Gunicorn-backed backend built from `Dockerfile.prod`, and nginx serving the pre-built frontend. Used by `deploy-prod.sh`. | `docker-compose -f docker-compose.prod.yml up -d` |
| `docker-compose.apps.yml` | Runs only the frontend and backend while expecting the `beer-game-network` and database to be provided externally—handy when pointing at a managed DB service. | `docker-compose -f docker-compose.apps.yml up` |
| `docker-compose.db.yml` | Launches a standalone MariaDB instance with tuned performance flags on host port `3307` for local development or tooling that only needs the database. | `docker-compose -f docker-compose.db.yml up` |

## GPU and CPU Controls

GPU builds can be toggled by passing `FORCE_GPU=1` to compatible targets:

```bash
make up FORCE_GPU=1
```

Helper aliases exist for convenience:

- `make gpu-up` and `make gpu-up-dev` force GPU mode.
- `make cpu-up` and `make cpu-up-dev` force CPU mode.

## Remote Training Commands

To run training jobs on a remote machine, provide the required `REMOTE` host and optional
parameters. Results are synced back to `backend/checkpoints/supply_chain_gnn.pth` by
default.

```bash
make remote-train REMOTE=user@host
make remote-train-dataset REMOTE=user@host DATASET=path/to/dataset
```

Available variables:

- `REMOTE_DIR` (default `~/beer-game`)
- `EPOCHS` (default `50`)
- `DEVICE` (default `cuda`)
- `WINDOW` (default `12`)
- `HORIZON` (default `1`)
- `NUM_RUNS` (default `64`)
- `T` (default `64`)
- `DATASET` (required for `remote-train-dataset`)
- `SAVE_LOCAL` (default `backend/checkpoints/supply_chain_gnn.pth`)

## Local Training Helpers

Set up and execute training locally from the `backend` directory:

```bash
make train-setup   # Create a Python venv and install training dependencies
make train-cpu     # Run CPU-based training
make train-gpu     # Run GPU-enabled training
```

## Environment Setup

`make init-env` runs either `scripts/setup_env.ps1` (Windows) or `scripts/setup_env.sh`
(Linux/macOS) to populate configuration files.

## Additional Tips

- Override the hostname printed in helper messages with `HOST=<ip-or-hostname>` when
  running locally (defaults to `localhost`).
- Remote helper commands use `REMOTE_HOST=172.29.20.187` for status messages.
- The `help` target prints an annotated list of all commands if you need a quick
  reminder: `make help`.
