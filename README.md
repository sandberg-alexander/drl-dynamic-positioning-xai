# DRL Dynamic Positioning with Explainable AI

Deep Reinforcement Learning (PPO) for Dynamic Positioning of the milliAmpere1 autonomous ferry at NTNU, with a real-time SHAP-based Explainable AI dashboard.

- **Author:** Alexander Sandberg
- **Course:** TTK4900, Cybernetics and Robotics @ NTNU
- **Thesis:** [Deploying Trustworthy Deep Reinforcement Learning for Dynamic Positioning: A Real-time Explainable AI Approach on Real Maritime Cyber-Physical Systems](https://nva.sikt.no/registration/019a1506973c-03176847-b705-4208-9048-5069284d9c61)
- **Paper:** *Deep Reinforcement Learning for Ship Dynamic Positioning: A Live Explainable AI Approach on an Autonomous Ferry Prototype* (link TBD)

If you use this work, please cite:
```bibtex
@mastersthesis{sandberg2025drl,
  title     = {Deploying Trustworthy Deep Reinforcement Learning for Dynamic Positioning: A Real-time Explainable AI Approach on Real Maritime Cyber-Physical Systems},
  author    = {Sandberg, Alexander},
  year      = {2025},
  school    = {Norwegian University of Science and Technology (NTNU)},
  type      = {Master's thesis},
  course    = {TTK4900, Cybernetics and Robotics},
}
```

> **Note:** This is the reworked and modernised codebase. The original code as referenced in the master's thesis and paper is preserved on the [`legacy`](../../tree/legacy) branch.

## Repository structure

```
drl-dynamic-positioning-xai/
├── milliampere_dp/              # Shared domain library (pip, pure Python)
│   ├── src/milliampere_dp/
│   │   ├── vessel.py            # Physical constants, thruster geometry
│   │   ├── transforms.py        # NED/body frame transforms, ssa
│   │   ├── rewards.py           # Reward functions (Gaussian, penalties)
│   │   ├── log.py               # Structured logging
│   │   └── plotting/            # Entry-point plotting scripts
│   └── tests/                   # 69 unit tests
│
├── ros_packages/
│   ├── milliampere_env/         # Gymnasium environment (catkin)
│   │   ├── src/milliampere_env/
│   │   │   ├── milliampere_env.py   # Single config-driven env class
│   │   │   ├── config.py            # Pydantic YAML config validation
│   │   │   ├── transport.py         # RosTransport / MockTransport
│   │   │   ├── viewer.py            # Standalone pygame viewer (ROS subscriber)
│   │   │   └── wrappers.py          # Gymnasium wrappers
│   │   └── tests/               # 54 env tests (MockTransport, no ROS needed)
│   └── custom_ros_msgs/         # ROS message definitions (catkin)
│
├── drl/                         # DRL training/eval/deploy (pip)
│   ├── src/milliampere_drl/
│   │   ├── config.py            # TrainingConfig, EvalConfig, DeployConfig
│   │   ├── callbacks.py         # SaveModelCallback, HParamCallback, DPMetricsCallback
│   │   ├── tracker.py           # EvaluationTracker + CSV aggregation
│   │   ├── spline.py            # Cubic spline trajectory utilities
│   │   ├── deployer.py          # DeployMode enum + DRLDeployer state machine
│   │   ├── train.py             # drl-train entry point
│   │   ├── evaluate.py          # drl-evaluate entry point
│   │   └── deploy.py            # drl-deploy entry point (headless)
│   └── tests/                   # 37 tests
│
├── xai/                         # XAI explainer + dashboard (pip)
│   ├── src/milliampere_xai/
│   │   ├── model_wrappers.py    # Obs2ActionWrapper, Obs2ValueWrapper
│   │   ├── shap_explainer.py    # SHAP pipeline + xai-explain entry point
│   │   └── dashboard.py         # Full pygame rendering dashboard
│   └── tests/                   # 8 tests
│
├── configs/
│   ├── env/                     # Environment YAML configs
│   │   ├── dp_positive_thrust.yaml
│   │   ├── dp_waypoint.yaml
│   │   └── legacy/              # Backward-compatible configs (v4, v5, v9, v10)
│   ├── training/default.yaml    # Default training hyperparameters
│   └── deploy/default.yaml      # Default deploy config
│
├── docker/
│   ├── Dockerfile.base          # ROS Noetic + milliampere_dp + milliampere_env
│   ├── Dockerfile.drl           # DRL container (training, eval, deploy, viewer)
│   ├── Dockerfile.xai           # XAI container (SHAP dashboard)
│   ├── docker-compose.nvidia.yml
│   ├── docker-compose-template.yml
│   └── generate_compose.py      # Generates compose for local/remote mode
│
├── external/milliampere/        # Git submodule (milliAmpere1 simulator)
├── data/                        # Models, runs, XAI samples (gitignored binaries)
├── Justfile                     # Task runner (just --list for all commands)
└── .pre-commit-config.yaml      # Ruff linting + formatting
```

## Packages

| Package | Type | Purpose | ROS required? |
|---------|------|---------|---------------|
| `milliampere_dp` | pip | Shared domain code: vessel constants, transforms, rewards, plotting | No |
| `milliampere_env` | catkin | Gymnasium environment with config-driven architecture | At runtime only |
| `custom_ros_msgs` | catkin | ROS message definitions (EnvState, Mode, etc.) | Yes |
| `milliampere_drl` | pip | DRL training, evaluation, deployment | At runtime only |
| `milliampere_xai` | pip | SHAP explainer, model wrappers, rendering dashboard | At runtime only |

All pip packages use lazy ROS imports so that tests run on host without ROS installed.

## Quick start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Compose v2
- [just](https://github.com/casey/just) command runner
- [uv](https://github.com/astral-sh/uv) Python package manager (for host development)
- NVIDIA Container Toolkit (optional, for GPU rendering)

### 1. Clone and set up

```bash
git clone --recurse-submodules <repo-url>
cd drl-dynamic-positioning-xai
just setup          # install packages, hooks, submodules
```

### 2. Choose environment mode

**Local** (simulator runs in Docker alongside DRL/XAI):
```bash
just compose-local
```

**Remote** (connect to real milliAmpere1 or remote simulator over LAN):
```bash
just compose-remote <remote-ip>
```

The remote host must have ROS master running and `ROS_MASTER_URI` / `ROS_IP` set (see the [milliAmpere1 setup](#milliampere1-simulator) section).

### 3. Build and start

```bash
just build          # builds base image first, then drl + xai
just up             # software rendering (works everywhere)
# or
just up-gpu         # NVIDIA GPU acceleration
```

### 4. Enable X11 forwarding

Required for the viewer and XAI dashboard:
```bash
xhost +local:docker
```

## Usage

All commands run inside Docker containers. The `drl` container handles training, evaluation, deployment, and the debug viewer. The `xai` container handles the SHAP explanation dashboard.

### Train a DRL agent

```bash
# Default config (PPO, 500k steps, seed 42)
docker exec -it drl bash -c 'source /root/catkin_ws/devel/setup.bash && drl-train'

# Custom config
docker exec -it drl bash -c 'source /root/catkin_ws/devel/setup.bash && drl-train --config /app/configs/training/default.yaml'

# Override device
docker exec -it drl bash -c 'source /root/catkin_ws/devel/setup.bash && drl-train --device cpu'
```

Training creates a timestamped directory under `/app/models/` containing:
- `models/` - saved checkpoints (every N policy updates)
- `logs/` - TensorBoard logs
- `training_config.yaml` - snapshot of the training configuration
- `env_config.yaml` - snapshot of the environment configuration

Press `Ctrl+C` once for graceful stop (saves final model), twice to force exit.

**Weights & Biases** (optional): add `wandb_project: "my-project"` to the training config YAML, or disable with `--no-wandb`.

### Evaluate models

```bash
# Evaluate next unevaluated model (10 episodes)
docker exec -it drl bash -c 'source /root/catkin_ws/devel/setup.bash && drl-evaluate --dir /app/models/training_<timestamp>'

# Evaluate ALL models in a run (replaces the old run_evaluation.sh)
docker exec -it drl bash -c 'source /root/catkin_ws/devel/setup.bash && drl-evaluate --dir /app/models/training_<timestamp> --episodes 5 --all'

# Or via Justfile
just evaluate-all /app/models/training_<timestamp> 5
```

Evaluation is **resumable** - progress is tracked in JSON. You can stop and restart at any time.

### Deploy a DRL agent

Start the deployer first, then switch to DRL mode. This order is important — the deployer must be running before the mode switch so there is no gap without active DP control.

```bash
# 1. Start the deployer (in one terminal)
docker exec -it drl bash -c 'source /root/catkin_ws/devel/setup.bash && drl-deploy'

# Or with a custom model
docker exec -it drl bash -c 'source /root/catkin_ws/devel/setup.bash && drl-deploy --model /app/models/training_<timestamp>/models/best_model.zip --env-config /app/configs/env/dp_waypoint.yaml'

# 2. Switch to DRL mode (in another terminal, when ready)
# Option A: Open http://0.0.0.0:61200 in a browser and select DRL mode
# Option B: Command line
rosservice call /supervisor/switch_mode "mode: 'drl'"
```

The deployer runs **headless** (no display window). Test modes are controlled from the XAI dashboard keyboard (keys 0-5) or the ROS topic `/drl/mode`:

| Key | Mode | Description |
|-----|------|-------------|
| 0 | DP | Dynamic positioning (default) |
| 1 | DP_TEST | DP with waypoint sequence test |
| 2 | NORTH_TEST | Northward path following |
| 3 | SPLINE_TEST | Spline path following |
| 4 | ACTION_SAMPLE | Collect action samples for SHAP |
| 5 | VF_SAMPLE | Collect value function samples |

### Launch the debug viewer

Separate terminal (requires X11):
```bash
docker exec -it drl bash -c 'source /root/catkin_ws/devel/setup.bash && python3 -m milliampere_env.viewer'

# With GPU rendering and custom FPS
docker exec -it drl bash -c 'source /root/catkin_ws/devel/setup.bash && python3 -m milliampere_env.viewer --gpu --fps 10'
```

### Start the XAI dashboard

Requires the deployer to be running. Separate terminal (requires X11):
```bash
docker exec -it xai bash -c 'source /root/catkin_ws/devel/setup.bash && xai-explain --gpu'
```

Keyboard inputs in the dashboard window control the deployer mode (keys 0-5, same as the table above).

### Plot results (host-side)

Plotting scripts are installed as CLI entry points - no Docker needed:
```bash
# Evaluation rewards across training steps
plot-eval --dir data/models/training_<timestamp>

# Training episode rewards/lengths
plot-monitored --csv data/models/training_<timestamp>/logs/monitor.csv

# Deployment run data
plot-run --csv data/runs/sim/data_test_dp_<timestamp>.csv

# Spline reference vs realised path
plot-spline --csv data/runs/sim/data_test_spline_<timestamp>.csv
```

### Simulate wind

```bash
docker exec simulator_local bash -c 'source /workspace/devel/setup.bash && rosservice call /sim/set_wind_sigma "steady: 3.0
gust: 5.0"'
```

## Development

### Run tests (no Docker needed)

```bash
just test           # milliampere_dp (69 tests)
just test-env       # milliampere_env (54 tests, uses MockTransport)
just test-drl       # milliampere_drl (37 tests)
just test-xai       # milliampere_xai (8 tests)
just test-all       # all 168 tests
just test-cov       # milliampere_dp with coverage
```

### Lint and format

```bash
just lint           # ruff check + format via pre-commit
```

### Live editing in Docker

Source code is volume-mounted into containers. Edits to Python files in `milliampere_dp/`, `milliampere_env/`, `drl/`, `xai/`, and `configs/` take effect immediately without rebuilding. Only changes to dependencies or Dockerfiles require `just build`.

### Configuration

All environment parameters, training hyperparameters, and deployment settings are driven by YAML configs validated with [Pydantic](https://docs.pydantic.dev/). Configs are frozen and reject unknown fields:

```yaml
# configs/training/default.yaml
seed: 42
env_config: configs/env/dp_positive_thrust.yaml
n_steps: 2048
total_timesteps: 500000
policy: MlpPolicy
device: auto
save_interval: 2
```

Every training run snapshots its config into the run directory for reproducibility.

### Adding a new environment variant

Create a YAML file in `configs/env/` following the existing examples. All scripts use `MilliAmpere1-v1` with a `config_path` argument - no code changes needed.

## milliAmpere1 simulator

The simulator runs as a Docker container from the `external/milliampere` git submodule. Make sure the submodule is initialised:
```bash
just submodule-init
```

For remote deployment, the remote host must set:
```bash
export ROS_MASTER_URI=http://<remote_ip>:11311
export ROS_IP=<remote_ip>
```

## All commands

Run `just --list` for a complete list:

```
just setup          # Full dev environment setup
just install        # Install all packages in editable mode
just test-all       # Run all 168 tests
just lint           # Ruff lint + format
just compose-local  # Generate compose for local simulation
just compose-remote # Generate compose for remote connection
just build          # Build Docker images (base first, then drl/xai)
just up             # Start containers (software rendering)
just up-gpu         # Start containers (NVIDIA GPU)
just down           # Stop containers
just evaluate-all   # Evaluate all models in a training run
just plot-eval      # Plot evaluation rewards
just plot-run       # Plot deployment run data
just plot-spline    # Plot spline reference vs realised path
just clean          # Remove Python cache files
```

## License

MIT
