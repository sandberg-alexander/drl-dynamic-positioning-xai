# Parallel Training & Evaluation

**Status:** In progress
**Priority:** High
**Progress:** 20 / 100

---

### About project

Parallelize DRL training and evaluation for `drl-dynamic-positioning-xai` using stable-baselines3's `SubprocVecEnv` with separate ROS master instances per environment. The current single-environment setup is bottlenecked by the simulator's real-time sleep (`dt=0.1s` per step). Running N environments in parallel collects N times the transitions per wall-clock second, yielding ~3x training speedup at N=4. A stretch goal uses the simulator's existing external control mode to eliminate real-time sleep entirely, enabling faster-than-realtime simulation.

---

## Motivation

1. **Simulator is the bottleneck** -- each `step()` calls `transport.sleep(0.1)` plus simulator computation time. The PPO policy update is fast (~ms on CPU); >99% of wall-clock time is spent waiting for the simulator.
2. **PPO is designed for parallelism** -- PPO collects a rollout buffer of `n_steps * n_envs` transitions before each policy update. SB3's `SubprocVecEnv` is the standard mechanism for this. With N=4, the rollout buffer fills ~4x faster.
3. **Training takes hours** -- 500k timesteps at `dt=0.1s` = ~14 hours of wall-clock time (single env). At N=4, this drops to ~4-5 hours. With external mode (faster-than-realtime), potentially under 1 hour.
4. **Evaluation is embarrassingly parallel** -- evaluating a model across 10 episodes can run all episodes simultaneously on separate simulator instances.

---

## Current state

```
Single Docker container:
  roscore:11311
  simulator.launch (~12 ROS nodes)
  drl_train (single MilliAmpereEnv → single RosTransport)
```

- `train.py` creates one `gym.make("MilliAmpere1-v1")` instance
- `rospy.init_node("drl_train")` called in parent process before env creation
- `RosTransport` subscribes to hardcoded topics: `/navigation/pose`, `/actuator_ref_1-4`, `/sim_vessel/reset_state`
- No `SubprocVecEnv`, `DummyVecEnv`, or any vectorized env usage
- Simulator runs in real-time (or 1.5x via `sim_clock` node)

### Key files

| File | Role |
|------|------|
| `drl/src/milliampere_drl/train.py` | Training entry point (`rospy.init_node` at line 97) |
| `drl/src/milliampere_drl/evaluate.py` | Evaluation entry point |
| `drl/src/milliampere_drl/config.py` | `TrainingConfig`, `EvalConfig` (Pydantic) |
| `ros_packages/milliampere_env/src/milliampere_env/transport.py` | `VesselTransport` ABC, `RosTransport`, `MockTransport` |
| `ros_packages/milliampere_env/src/milliampere_env/milliampere_env.py` | `MilliAmpereEnv` (Gymnasium) |
| `external/milliampere/workspace/src/simulator.launch` | Simulator launch (`external_mode`, `external_mode_port` params) |
| `external/milliampere/workspace/src/sim_milliampere/scripts/external_simulator_control.py` | TCP step protocol for external mode |
| `docker/docker-compose.yml` | Current single-simulator Docker setup |

---

## Architecture decisions

### Decision 1: Separate containers per simulator (not namespace isolation, not multi-roscore-in-one-container)

Each parallel simulator runs in its own Docker container with its own roscore on port 11311 (standard). Container networking provides isolation -- no port management needed, no ROS namespace remapping, no simulator source modifications. The `drl` container's subprocesses each set `ROS_MASTER_URI=http://sim_{i}:11311`.

**Alternatives considered:**
- **Namespace isolation** (one roscore, `<group ns="sim_N">`): The milliAmpere simulator was not designed for this -- nodes use hardcoded absolute topic paths. Would require modifying the simulator source and creating elaborate remap rules. Fragile.
- **Multiple roscores in one container**: No process isolation, complex port management (11311, 11312, ...), harder cleanup. Used by MultiROS but messier than Docker's natural isolation.
- **`docker compose --scale`**: All replicas get identical config -- can't set per-instance hostnames or env vars.

**Why separate containers:**
- Docker provides natural process isolation per roscore. Each on standard port 11311.
- No changes needed to the simulator source code or launch files.
- Containers reach each other by hostname (`sim_0`, `sim_1`, ...) on a shared bridge network.
- Health checks via `depends_on: condition: service_healthy` ensure all simulators are ready before training starts.
- Clean shutdown via `docker compose down`.

### Decision 2: SubprocVecEnv with `spawn` start method

`SubprocVecEnv` spawns each env in a fresh Python subprocess. The `spawn` start method (SB3's default) creates a fresh Python interpreter per subprocess, avoiding the `fork`-related `rospy.init_node()` "already called" error.

**Key constraint:** `rospy.init_node()` must be called *inside* the env factory callable, not in the parent process. The current `train.py:97` call must move.

### Decision 3: Env factory pattern

Each env index gets a factory callable that:
1. Sets `ROS_MASTER_URI` to the correct port for this env
2. Calls `rospy.init_node(f"drl_train_{env_id}", anonymous=True)`
3. Creates and returns `gym.make("MilliAmpere1-v1", config_path=...)`

This keeps the existing `RosTransport` / `MilliAmpereEnv` code untouched -- all changes are in the factory and `train.py`.

### Decision 4: External mode for faster-than-realtime (stretch goal)

The simulator already supports an `external_mode` that accepts TCP commands on a configurable port:
- `$SIM_CLOCK,<timestamp>` -- advance simulation clock to timestamp
- `$SIM_RESET,N0,E0,H0,u0,v0,r0` -- reset vessel state

This eliminates the `dt=0.1s` real-time sleep. The simulator steps as fast as the physics engine computes (~ms per step instead of 100ms). A new `ExternalModeTransport(VesselTransport)` would implement this protocol.

### Decision 5: Backward-compatible configuration

`TrainingConfig.n_envs` defaults to `1`, preserving existing single-env behavior. `n_envs > 1` activates `SubprocVecEnv`. No breaking changes to CLI or config files.

### Resource estimates

| N (envs) | roscores | Simulator nodes | Est. RAM | Est. CPU cores |
|----------|----------|-----------------|----------|----------------|
| 1 | 1 | ~12 | ~1 GB | 2-4 |
| 2 | 2 | ~24 | ~2 GB | 4-6 |
| 4 | 4 | ~48 | ~3-4 GB | 6-10 |
| 8 | 8 | ~96 | ~6-8 GB | 12-16 |

---

## Phases

### Phase 1 -- Multi-Simulator Orchestration

Infrastructure for running N independent simulator containers in parallel.

**Architecture:** Each simulator runs in its own Docker container with its own roscore on port 11311. All containers share a bridge network (`ros_parallel_net`). The `drl` container reaches each simulator by hostname (`sim_0`, `sim_1`, ...).

**Files created/modified:**
- `docker/Dockerfile.simulator` — Pre-built simulator image (copies workspace + `catkin_make` at build time, no startup build step)
- `docker/generate_compose.py` — Extended with `parallel` mode + `--n-envs N` argument
- `docker/docker-compose-parallel.yml` — Auto-generated compose file (N sim services + drl + xai)
- `Justfile` — Added `compose-parallel`, `build-parallel`, `up-parallel`, `down-parallel` recipes

**Action items:**
- [x] Create `docker/Dockerfile.simulator`: extends `milliampere-sim:latest`, COPY workspace, `catkin_make` at build time
- [x] Extend `generate_compose.py` with `parallel` mode: generates N `sim_{i}` services programmatically (Python dict → `yaml.dump`), each with health check (`rostopic list`), `depends_on: condition: service_healthy` on drl, shared bridge network
- [x] Add Justfile recipes: `just compose-parallel [n_envs]`, `just build-parallel`, `just up-parallel`, `just down-parallel`
- [x] Add `docker/docker-compose-parallel.yml` to `.gitignore` (auto-generated)
- [x] Test: all 4 simulators reach healthy status, `rostopic list` from drl against each sim shows full topic set, `/sim_vessel/reset_state` service available on each
- [x] Test: `/navigation/pose` actively publishing on all 4 sims with independent seq numbers
- [x] Test: `drl-train` runs against sim_0 — connects to topics, completes episodes, PPO logging active
- [x] Test: 4 concurrent rospy nodes via `multiprocessing.Process` (one per sim) all receive independent pose data
- [x] Test: parallel reset service calls across all 4 sims complete independently
- [x] Verify existing `just up` / `just down` still works (regression)

**Implementation notes:**
- Simulator command uses `;` separators (not `&&`) between `source` and backgrounded `roscore &` / `roslaunch &`. With `&&`, bash groups `source ... && roscore &` as one background job, leaving `roslaunch` in a subshell without the sourced ROS environment.
- Health check wraps `source` in `bash -c '...'` because Docker runs `CMD-SHELL` with `/bin/sh` (dash), which doesn't support the `source` bashism.
- Parallel compose is generated programmatically (Python dict → `yaml.dump`) rather than from the string-replacement template, since N services need to be created dynamically.
- `drl` service keeps a default `ROS_MASTER_URI=http://sim_0:11311` for manual `docker exec` convenience; Phase 2 subprocesses will override per-env.
- `fork` multiprocessing context works for the parallel test because the parent process has not called `rospy.init_node()`. Phase 2's `SubprocVecEnv` should use `spawn` (SB3's default) to be safe.
- Tested with `multiprocessing.get_context("fork")` for validation since `spawn` can't pickle inline functions — `SubprocVecEnv` uses `cloudpickle` which handles this.

### Phase 2 -- SubprocVecEnv Training Integration

Code changes to parallelize training with `SubprocVecEnv`.

**Files modified:**
- `drl/src/milliampere_drl/train.py` — `_make_env` factory, `--n-envs` CLI, `DummyVecEnv`/`SubprocVecEnv` conditional
- `drl/src/milliampere_drl/config.py` — `n_envs: int = 1` field on `TrainingConfig`
- `drl/src/milliampere_drl/callbacks.py` — `DPMetricsCallback` fixed for VecEnv + actual info keys
- `configs/training/default.yaml` — `n_envs: 1`

**Action items:**
- [x] Create `_make_env(rank, config_path, seed, log_dir)` factory in `train.py`
  - Sets `ROS_MASTER_URI=http://sim_{rank}:11311` and `ROS_HOSTNAME=drl_container`
  - Calls `rospy.init_node()`, imports `milliampere_env`, creates env + `Monitor` — all inside subprocess
- [x] `n_envs=1`: `rospy.init_node()` in parent process, `DummyVecEnv` — identical to previous behavior
- [x] `n_envs>1`: `SubprocVecEnv` with `start_method="forkserver"`, no rospy in parent
- [x] Add `n_envs: int = Field(default=1, ge=1)` to `TrainingConfig`
- [x] Add `n_envs: 1` to `configs/training/default.yaml`
- [x] Add `--n-envs` CLI argument to `drl-train` (overrides config)
- [x] Validate `batch_size` divides `n_steps * n_envs` with warning
- [x] Fix `DPMetricsCallback`: use actual info keys (`epsilon`, `thrusters`), aggregate across all envs with `np.mean`
- [x] PPO receives `vec_env` instead of bare `env`
- [x] `finally` block calls `vec_env.close()` (properly shuts down subprocesses)
- [x] Print `n_envs` in startup banner
- [x] All 40 DRL unit tests pass
- [ ] Test N=1 regression in Docker (single sim)
- [ ] Test N=4 parallel in Docker (4 sims)

**Implementation notes:**
- `Monitor` wraps each env **inside** the factory (not `VecMonitor` outside) — avoids double-counting episode stats.
- `forkserver` start method chosen over `fork` (ROS threading deadlock risk) and `spawn` (SB3 default but slower).
- `_make_env` is a module-level function (not a closure inside `main`) so `cloudpickle` can serialize it for `SubprocVecEnv`.
- `DPMetricsCallback` was previously a no-op — the keys it checked (`position_error`, `heading_error`, `total_thrust`) don't exist in `MilliAmpereEnv._get_info()`. Now uses actual keys: `epsilon` (body-frame error) and `thrusters` (RPM setpoints).
- `total_timesteps` counts across all envs — with N=4 and 500k timesteps, training finishes in ~125k wall-clock steps per env.

### Phase 3 -- Parallel Evaluation

Apply the same SubprocVecEnv pattern to `evaluate.py`.

- [ ] Add `n_envs: int = 1` and `sim_host_prefix: str = "sim_"` to `EvalConfig`
- [ ] Create vectorized evaluation function: N envs run different episodes simultaneously
- [ ] Adapt `EvaluationTracker` to aggregate results from N concurrent episode streams
  - Episodes complete at different times -- track per-env progress
  - Resumability: save which episodes are assigned to which env
- [ ] Add `--n-envs` CLI argument to `drl-evaluate`
- [ ] Verify deterministic evaluation: `seed + episode_index` ensures reproducibility regardless of N
- [ ] Test: compare evaluation metrics at N=1 vs N=4 (should be identical given same seeds)

### Phase 4 -- Faster-Than-Realtime via External Mode (Stretch)

Use the simulator's existing `external_simulator_control.py` TCP protocol to step the clock programmatically instead of real-time sleep.

- [ ] Study `external_simulator_control.py` TCP protocol in detail:
  - Connection handshake: receive `$SIM_CONN,<current_time>\r\n`
  - Clock step: send `$SIM_CLOCK,<timestamp>\r\n`
  - Reset: send `$SIM_RESET,N0,E0,H0,u0,v0,r0\r\n`
  - Acknowledgment format
- [ ] Create `ExternalModeTransport(VesselTransport)`:
  - `__init__`: TCP connect to `external_mode_port`, call `rospy.init_node()`
  - `sleep(dt)`: send `$SIM_CLOCK,<current_time + dt>` instead of `rospy.sleep(dt)`
  - `reset_simulation()`: send `$SIM_RESET,...` via TCP (faster than ROS service call)
  - `get_pose()`, `publish_actuator_setpoints()`: still via rospy (physics state only available on ROS topics)
  - `is_shutdown()`: check TCP connection + rospy shutdown
- [ ] Update `simulator.launch` invocation to set `use_sim_time=1 external_mode=1 external_mode_port=203X`
- [ ] Parameterize port: `ExternalModeTransport(config, control_port=2030+env_id)`
- [ ] Add `use_external_mode: bool = False` to `TrainingConfig`
- [ ] Benchmark: measure simulation step time (physics compute only, no sleep)
- [ ] Combined test: N=4 envs * external mode -- verify correctness and measure total speedup
- [ ] Document trade-offs: faster training but simulator behavior may differ at non-realtime speeds (numerical integration sensitivity)

### Phase 5 -- Benchmarking & Documentation

- [ ] Benchmark wallclock time for 500k timesteps at N=1, 2, 4, 8
- [ ] Benchmark with external mode at N=1, 2, 4
- [ ] PPO hyperparameter guidance: document recommended `n_steps`, `batch_size`, `n_epochs` for different N values
- [ ] Learning curve comparison: plot single-env vs parallel training reward curves (same total timesteps)
- [ ] Document setup in README: parallel training section with resource requirements
- [x] Update Justfile with parallel workflow recipes (done in Phase 1)
- [ ] Version bump

---

## References

- [MultiROS](https://github.com/ncbdrck/multiros) -- IEEE CASE 2022, separate roscores per env with multiprocessing
- [UniROS](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473579/) -- extends MultiROS with separate Python interpreters
- [robo-gym](https://github.com/jr-robotics/robo-gym) -- client-server decoupling of RL from ROS via gRPC
- [SB3 Vectorized Environments](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- [PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) -- 37 details including multi-env scaling
