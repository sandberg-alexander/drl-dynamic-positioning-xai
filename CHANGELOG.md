## v0.7.0 (2026-03-23)

### Feat

- Phase 1 — extract shared rendering into milliampere_dp.rendering

## v0.6.0 (2026-03-22)

### Feat

- polish, velocity subscription, and version bump to 0.6.0

## v0.5.0 (2026-03-21)

### Feat

- Phase 5 — infrastructure, CI, type checking, and changelog

### Fix

- use .venv/bin/cz path in Justfile release recipes
- use python -m commitizen.cli in Justfile for venv compat
- add ruff exclusions for legacy and dashboard files in CI
- install ruff and pyright explicitly in CI workflow

## v0.4.0 (2026-03-21)

### Feat

- create `milliampere_drl` pip package with config-driven entry points (`drl-train`, `drl-evaluate`, `drl-deploy`), Pydantic configs, callbacks, EvaluationTracker, spline utilities, and DRLDeployer state machine — 37 tests
- create `milliampere_xai` pip package with SHAP explainer, model wrappers, and pygame dashboard — 8 tests
- promote 4 analysis scripts to `milliampere_dp` entry points (`plot-eval`, `plot-run`, `plot-spline`, `plot-monitored`)
- add optional W&B integration, seed management, and config snapshotting
- comprehensive README with full usage guide
- 168 tests passing (69 dp + 54 env + 37 drl + 8 xai)

### Fix

- fix truncation log spam in env (log once, not every step)
- fix XAI video recording (MJPG codec, capture size at record time)
- fix XAI crash on startup (wait for target_pose before rendering)

### Refactor

- move `custom_ros_msgs` to `ros_packages/`, delete `project_mAXAI/`
- update Dockerfiles to pip-install packages, update compose volume mounts

## v0.3.0 (2026-03-21)

### Feat

- add DeployMode IntEnum replacing integer mode flags 0–5
- enable ruff linting for application code in pre-commit
- add unified version numbering (0.3.0) via importlib.metadata
- parameterize all plot scripts with argparse CLI
- add `eval.py --all` flag (replaces `run_evaluation.sh`)
- add `evaluate-all` and `up-gpu` Justfile recipes
- extract NVIDIA GPU config into `docker-compose.nvidia.yml` override

### Fix

- fix spelling errors: explaination, bakground, arrowhead_widht, calulate, interupt
- fix `pygame.quit()` missing parentheses in shap_explanation.py
- fix stale env ID: migrate SHAP to MilliAmpere1-v1
- fix SHAP CUDA/CPU tensor mismatch (load model on CPU)
- fix `pygame.init()` missing before font creation
- fix video_writer referenced before assignment
- initialize `Agent.actuator_ref` in `__init__`
- fix NED display: add `epsilon_ned` to EnvState.msg and rewrite NedRender positioning logic

### Refactor

- replace magic numbers with `milliampere_dp.vessel` constants
- remove ~200 lines of dead/commented-out code
- remove pygame from deploy.py — runs headless, mode controlled via ROS topics
- merge `plot_run.py` + `plot_run2.py` into single parameterized script
- add docstrings to 10 key classes in render_explanation.py

## v0.2.0 (2026-03-16)

### Feat

- replace 14 copy-pasted env files (~11,000 lines) with single parameterized `MilliAmpereEnv` class (~430 lines) driven by validated YAML configs (Pydantic, frozen)
- add `EnvConfig` Pydantic model: validates all params at load, fails fast on typos
- add `VesselTransport` ABC with `RosTransport` (lazy rospy imports) and `MockTransport` for testing without ROS
- single gym registration: `MilliAmpere1-v1` with `config_path` kwarg
- legacy configs for v4, v5, v9, v10 preserving trained model compatibility
- standalone pygame viewer reproducing full legacy visualization (heatmap, value bars, observation text)
- 54 env tests using MockTransport, no ROS needed (115 total)

### Fix

- fix Python 3.8 compatibility, gymnasium version specifier, SB3 Monitor info key conflict

### Refactor

- modernize Docker: uv instead of pip (10–100x faster installs), NVIDIA GPU passthrough via nvidia-container-toolkit + CDI, dev volume mounts for live editing
- add `--device` flag to train.py (auto|cpu|cuda), Ctrl+C stops cleanly via SB3 callback
- archive old `gym_env` to `gym_env_legacy`

## v0.1.0 (2026-03-15)

### Feat

- create `milliampere_dp` shared domain library extracting vessel constants, coordinate transforms, reward functions, structured logging, and plotting utilities from duplicated env files — 61 unit tests
- restructure repository: move submodule to `external/milliampere`, Docker files to `docker/`
- add `.pre-commit-config.yaml` with ruff check + format
- add Justfile with dev workflows (setup, test, lint, docker)
- add MIT LICENSE
