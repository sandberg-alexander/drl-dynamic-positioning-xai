# Contributing

Thank you for your interest in contributing to this project.

## Prerequisites

- Python 3.8+ (3.10+ recommended; 3.8 required for ROS Noetic Docker images)
- [uv](https://github.com/astral-sh/uv) (package manager)
- [just](https://github.com/casey/just) (task runner)
- Docker with Compose v2 (for integration/deployment)

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd drl-dynamic-positioning-xai
just setup    # installs all packages, pre-commit hooks, and git submodules
```

## Running tests

```bash
just test-all           # all tests (no Docker needed)
just test               # milliampere_dp only
just test-env           # milliampere_env only (uses MockTransport)
just test-drl           # milliampere_drl only
just test-xai           # milliampere_xai only
just test-integration   # full DRL episode with MockTransport
just test-cov           # all packages with coverage reports
```

## Linting and type checking

```bash
just lint         # ruff check + format via pre-commit
just type-check   # pyright (0 errors expected, warnings OK for ROS imports)
```

Both run automatically in CI on every push.

## Code style

- **Formatter/linter:** [ruff](https://docs.astral.sh/ruff/) (88-char line length, enforced via pre-commit)
- **Type checker:** [pyright](https://github.com/microsoft/pyright) in basic mode
- **Type hints** on all function signatures
- **Docstrings** on all public functions (NumPy format)
- `from __future__ import annotations` in every module

## Commit conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/) enforced by [commitizen](https://commitizen-tools.github.io/commitizen/). Every commit message must follow the format:

```
type(scope): description

# Examples:
feat(env): add waypoint tracking mode
fix(drl): handle keyboard interrupt during evaluation
chore(ci): enable uv caching in GitHub Actions
```

The pre-commit hook validates commit messages automatically.

## Project structure

The repository is organized as 4 Python packages with a clean dependency graph:

```
milliampere_dp    (shared domain library, no ROS)
  └─ milliampere_env   (Gymnasium environment, catkin)
       ├─ milliampere_drl  (training/eval/deploy)
       └─ milliampere_xai  (SHAP explanations)
```

All packages use lazy ROS imports so tests run on the host without ROS installed.

## Pull request workflow

1. Create a feature branch from `rework/v2`
2. Make your changes
3. Ensure `just test-all`, `just lint`, and `just type-check` all pass
4. Push and open a PR against `rework/v2`

## Available commands

Run `just --list` for all available recipes.
