# drl-dynamic-positioning-xai

default:
    @just --list

# --- Setup ---

# Full dev environment setup
setup: install setup-hooks submodule-init

# Install milliampere_dp in editable mode with dev deps
install:
    cd milliampere_dp && uv pip install -e ".[dev,plot]"

# Install pre-commit hooks
setup-hooks:
    pre-commit install

# Init git submodules
submodule-init:
    git submodule update --init --recursive

# --- Testing ---

# Run unit tests (host, no ROS needed)
test:
    cd milliampere_dp && python -m pytest

# Run tests with coverage
test-cov:
    cd milliampere_dp && python -m pytest --cov=milliampere_dp --cov-report=term-missing

# --- Linting ---

# Run pre-commit checks on all files
lint:
    pre-commit run --all-files

# --- Docker ---

# Generate docker-compose for local simulation
compose-local:
    cd docker && python3 generate_compose.py local

# Generate docker-compose for remote connection
compose-remote remote_ip:
    cd docker && python3 generate_compose.py remote --remote-ip {{remote_ip}}

# Build all Docker images
build:
    docker compose -f docker/docker-compose.yml build

# Start containers
up:
    docker compose -f docker/docker-compose.yml up -d

# Stop containers
down:
    docker compose -f docker/docker-compose.yml down

# --- Cleanup ---

# Remove Python cache files
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
