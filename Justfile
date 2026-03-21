# drl-dynamic-positioning-xai

default:
    @just --list

# --- Setup ---

# Full dev environment setup
setup: install setup-hooks submodule-init

# Install all packages in editable mode with dev deps
install:
    cd milliampere_dp && uv pip install -e ".[dev]"
    uv pip install -e ros_packages/milliampere_env
    cd drl && uv pip install -e ".[dev]"
    cd xai && uv pip install -e ".[dev]"

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

# Run milliampere_drl tests (host, no ROS needed)
test-drl:
    cd drl && python -m pytest

# Run milliampere_xai tests (host, no ROS needed)
test-xai:
    cd xai && python -m pytest

# Run all tests
test-all: test test-env test-drl test-xai

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
    docker exec drl bash -c 'source /root/catkin_ws/devel/setup.bash && drl-evaluate --dir {{run_dir}} --episodes {{episodes}} --all'

# --- Plotting ---

# Plot evaluation rewards for a training run
plot-eval run_dir *args:
    plot-eval --dir {{run_dir}} {{args}}

# Plot run data from CSV
plot-run *args:
    plot-run {{args}}

# Plot spline reference vs realised path
plot-spline *args:
    plot-spline {{args}}

# --- Docker ---

# Generate docker-compose for local simulation
compose-local:
    cd docker && python3 generate_compose.py local

# Generate docker-compose for remote connection
compose-remote remote_ip:
    cd docker && python3 generate_compose.py remote --remote-ip {{remote_ip}}

# Build all Docker images (base first, then drl/xai in parallel; pass extra args like --no-cache)
build *args:
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.nvidia.yml build base {{args}}
    docker compose -f docker/docker-compose.yml -f docker/docker-compose.nvidia.yml build drl xai {{args}}

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
