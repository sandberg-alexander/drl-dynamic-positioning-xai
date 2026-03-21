# drl-dynamic-positioning-xai

default:
    @just --list

# --- Setup ---

# Full dev environment setup
setup: install setup-hooks submodule-init

# Install milliampere_dp and milliampere_env in editable mode with dev deps
install:
    cd milliampere_dp && uv pip install -e ".[dev,plot]"
    uv pip install -e ros_packages/milliampere_env

# Install pre-commit hooks
setup-hooks:
    pre-commit install

# Init git submodules
submodule-init:
    git submodule update --init --recursive

# --- Testing ---

# Run milliampere_dp unit tests (host, no ROS needed)
test:
    cd milliampere_dp && python -m pytest

# Run milliampere_env tests (host, uses MockTransport, no ROS needed)
test-env:
    python -m pytest ros_packages/milliampere_env/tests/

# Run all tests
test-all: test test-env

# Run tests with coverage
test-cov:
    cd milliampere_dp && python -m pytest --cov=milliampere_dp --cov-report=term-missing

# --- Linting ---

# Run pre-commit checks on all files
lint:
    pre-commit run --all-files

# --- Evaluation (Docker) ---

# Evaluate all models in a training run (runs inside drl container)
evaluate-all run_dir episodes="5":
    docker exec drl bash -c 'source /root/catkin_ws/devel/setup.bash && python3 /app/drl_code/eval.py --dir {{run_dir}} --episodes {{episodes}} --all'

# --- Docker ---

# Generate docker-compose for local simulation
compose-local:
    cd docker && python3 generate_compose.py local

# Generate docker-compose for remote connection
compose-remote remote_ip:
    cd docker && python3 generate_compose.py remote --remote-ip {{remote_ip}}

# Build all Docker images (pass extra args like --no-cache)
build *args:
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.nvidia.yml build {{args}}

# Start containers (software rendering, works everywhere)
up:
    docker compose -f docker/docker-compose.yml up -d

# Start containers with NVIDIA GPU acceleration (requires nvidia-container-toolkit)
up-gpu:
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.nvidia.yml up -d

# Stop containers
down:
    docker compose -f docker/docker-compose.yml down

# --- Cleanup ---

# Remove Python cache files
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
