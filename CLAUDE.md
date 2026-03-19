# CLAUDE.md

## Project Overview

Bachelor thesis project: Training a drone to autonomously land in vineyard soil strips using reinforcement learning with the Genesis physics simulator (`genesis-world` v0.4.3).

Migrated from v0.3.13 — see the original repo at `../genesis/` for history and legacy prototypes.

## Repository Structure

- **prototyp_global_coordinate/** — Global coordinate-based landing with cascading PID controller. Uses PPO via rsl-rl.
- **prototyp_obstacle_avoidance/** — CNN depth-map obstacle avoidance. Extends global_coordinate with random obstacles, downward-facing depth camera, and rsl-rl v5.0.1 `CNNModel` + `share_cnn_encoders`.
- **assets/** — Shared drone URDF + meshes (`assets/robots/draugas/`). Referenced from prototypes as `../assets/`.
- **hpc/** — HPC Leipzig cluster scripts: env setup, code sync, SLURM job submission.

Each prototype is self-contained with its own `envs/`, `controllers/`, etc.

## Running (prototyp_global_coordinate)

```bash
cd prototyp_global_coordinate

# PPO training (rsl-rl, headless, 4096 envs)
python train_rl.py -B 4096 --max_iterations 401

# PPO smoke test with viewer
python train_rl.py -B 4 -v --max_iterations 5

# Evaluation
python eval_rl.py      # PPO

# Visualization
python visualize_paths.py --ckpt 100 300                  # matplotlib + screenshots
python visualize_paths.py --ckpt 300 --video              # + landing GIF with trail
python visualize_paths.py --ckpt 100 300 --no_render      # matplotlib only

# W&B variants (same behavior, logs to Weights & Biases)
python train_rl_wb.py -B 4096 --max_iterations 401
python eval_rl_wb.py                                       # auto-finds latest checkpoint
```

## Dependencies

Conda environment: `ba_v04`. Key packages: `genesis-world==0.4.3` (installs `quadrants==0.4.4`), `torch>=2.0.0`, `numpy`, `scipy`, `pyyaml`, `tensorboard`, `wandb`, `rsl-rl-lib==5.0.1`, `tensordict`. Genesis v0.4.3 auto-installs `moviepy`, `opencv-python`, `mujoco` as deps.

## Genesis API Notes (v0.4.3)

### Changes from v0.3.13

- **`get_link("main_body")`** for URDF root link reference. v0.4.3's improved URDF parser no longer wraps root as "base" — use the actual URDF link name.
- **Quadrants backend** replaces Taichi. 30%+ faster collision, improved URDF parsing.
- **Legacy URDF parser fallback** for `.dae` meshes — works but shows a warning.
- **CoACD convex decomposition** runs automatically on first build (~60s for drone mesh, 60 convex hulls). Cached for subsequent runs.

### Unchanged from v0.3.13

- **`scene.add_camera()` blocked post-build** — bypass via `scene._visualizer.add_camera(res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise, near, far, env_idx, debug)` + `cam.build()`
- **Two-phase construction**: call `gs.init(backend=gs.gpu)` once, then `env.build()` — the env constructor does NOT build the scene. `scene.build()` triggers JIT compilation; must be called before stepping.
- **No `camera.start()`/`stop()`** — call `camera.render()` directly
- `camera.render(depth=True, segmentation=True)` returns a **tuple** `(rgb, depth, segmentation, normal)`, not a dict
- **Typo in API**: `drone.set_propellels_rpm(rpms)` (propellels, not propellers)
- Custom URDF drones need explicit `propellers_link_name` and `propellers_spin`
- Conventions: right-handed Z-up, quaternions w-x-y-z, Euler angles in degrees (scipy extrinsic x-y-z), all state tensors `(n_envs, ...)` on `gs.device`.
- **Auto-reset contaminates state**: when `step()` returns `done=True`, the env has already reset — `base_pos` reflects the new spawn, not the terminal position. Record positions BEFORE `step()` or break before reading post-done state.
- **Batched camera tracking**: `trans_quat_to_T(pos, quat)` from `genesis.utils.geom` converts to `(n_envs, 4, 4)` transforms. Multiply by a fixed offset matrix for attached cameras: `camera.set_pose(transform=torch.matmul(link_T, offset_T))`. Downward-facing offset: `Rotation.from_euler('zyx', [-90, -90, 0], degrees=True)` + `T[2,3] = -0.1`.
- **`gs.morphs.Box` with `fixed=True`**: supports per-env `set_pos(new_pos, envs_idx=idx, zero_velocity=True)` for obstacle randomization. Use `collision=False` when handling collisions via distance checks.
- **BatchRenderer requires Madrona Vulkan backend** — crashes with `abort()` on Volta and older GPUs (not a Python exception). Check `torch.cuda.get_device_capability() >= (7, 5)` before attempting. Turing+ required. Use Rasterizer with `env_separate_rigid=True` as fallback.
- **Rasterizer does NOT support batched cameras** — `_is_batched=False`, rejects `set_pose(transform=(N,4,4))`. Use `env_separate_rigid=True` which renders all envs in one `render()` call, returning `(n_rendered_envs, H, W)` in local env coordinates.
- **Rasterizer `env_separate_rigid` depth bug**: `depth[i]` for i>0 always shows env 0's rigid bodies, not env i's. Workaround: swap env 0's obstacle positions to match env_i via `set_pos(envs_idx=[0])` before each render, extract `depth[0]`. `set_pos()` updates Rasterizer visuals immediately — no `scene.step()` needed.
- **`scene.envs_offset`**: `(n_envs, 3)` numpy array of per-env world-space offsets after `scene.build()`. Camera `set_pose()` in Rasterizer uses local env coordinates, not world.
- **Non-batched `camera.render(depth=True)`** returns numpy `(H, W)` arrays, not torch tensors. Wrap with `torch.as_tensor(depth, dtype=gs.tc_float, device=gs.device)`.
- **Eval reward accumulation must use GPU tensors** — `env.step()` returns rewards on `gs.device`. Accumulator tensors (`ep_reward`) must be on the same device, not CPU.

## Architecture

The RL agent outputs a target position; a cascading PID controller (position->velocity->attitude->RPM, three nested loops) tracks it and produces motor RPMs.

Key files in `prototyp_global_coordinate/`:
- `envs/coordinate_landing_env.py` — rsl-rl compatible env (reset/step/reward)
- `controllers/pid_controller.py` — CascadingPIDController
- `config/training_config.yaml` — DEPRECATED, not used. All config is in `train_rl.py`

## rsl-rl Gotchas

- `OnPolicyRunner.__init__` calls `.pop()` on config dicts, mutating them. Always pass `copy.deepcopy(train_cfg)` when creating multiple runners.
- `get_inference_policy()` returns the actor model directly — works with both `MLPModel` (flat TensorDict) and `CNNModel` (multi-key TensorDict including image obs).
- The runner accesses `cfg["algorithm"]["rnd_cfg"]` unconditionally — always include `"rnd_cfg": None` in algorithm config even when not using RND.
- **GaussianDistribution defaults to `std_type="scalar"`** which can go negative. Use `"std_type": "log"` for stability, especially with low/zero `entropy_coef`.
- **`entropy_coef` must be small** relative to per-step rewards (which are dt-scaled to ~0.05/step). Values like 0.01 cause std explosion with log-space; 0.001 works.
- **`share_cnn_encoders: True`** goes in `"algorithm"` config, NOT in actor/critic. Critic config should omit `cnn_cfg` entirely — PPO injects `actor.cnns` automatically.
- **W&B logger needs `.to_dict()` on configs** — `WandbSummaryWriter.store_config()` calls `env_cfg.to_dict()`, which fails on plain dicts. Wrap with a `DictConfig(dict)` subclass that adds `to_dict()`. See `train_rl_wb.py`.
- **Pickle + DictConfig**: `cfgs.pkl` from W&B runs contains `DictConfig` instances. Any script loading it (e.g. `eval_rl_wb.py`) must `from train_rl_wb import DictConfig` or pickle will fail with `AttributeError`.

## Observation & Action Spaces

- **Observation**: `(n_envs, 17)` — `rel_pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + last_actions(4)`, each clipped and scaled
  - Observation scales: `rel_pos * 1/15` (train_rl.py) or `1/30` (train_rl_wb.py), `lin_vel * 1/5`, `ang_vel * 1/pi`
- **Actions**: `(n_envs, 4)` float in `[-1, 1]` — `[ax, ay, az, ayaw]`
  - `target_xyz = current_pos + action[:3] * action_scales` (offset from current position)
  - `target_yaw = ayaw * 180.0` (degrees)

## Key Parameters

- Drone: custom "draugas" URDF (mass 0.714kg, thrust2weight 2.25), base hover RPM 1789.2, max RPM 2700
- Spawn: height 10m fixed, drone offset +/-5m
- Target: `train_rl.py` uses fixed (3,3,1); `train_rl_wb.py` randomizes in 10x10m square at 1m height
- Curriculum: `train_rl.py` uses 20000-step curriculum; `train_rl_wb.py` has no curriculum (performs better)
- Success: hover within 0.3m of target at <0.3 m/s for 30 consecutive steps (0.3s)
- Crash: height < 0.2m, tilt > 60°, or distance from target > 50m
- Rewards: distance penalty (-5.0), time penalty (-0.5), crash (-100), success (+200)

## Running (prototyp_obstacle_avoidance)

```bash
cd prototyp_obstacle_avoidance

# PPO training (headless, 64 envs — depth rollouts use more memory)
python train_rl.py -B 64 --max_iterations 401

# Smoke test with viewer
python train_rl.py -B 4 -v --max_iterations 5

# Evaluation
python eval_rl.py --ckpt 300
python eval_rl.py --ckpt 300 --vis

# Visualization
python visualize_paths.py --ckpt 100 300 --no_render
python visualize_paths.py --ckpt 300 --video
```

## prototyp_obstacle_avoidance Architecture

Key files (mirrors `prototyp_global_coordinate/` structure):
- `envs/obstacle_avoidance_env.py` — Env: obstacles + depth camera + TensorDict
- `debug_depth.py` — Validates depth camera rendering and CNN gradient flow
- `visualize_obstacle_setup.py` — Validates strategic obstacle placement effectiveness

**Observations**: `TensorDict({"state": (n, 17), "depth": (n, 1, 64, 64)})`. State is identical 17-dim vector; depth is forward-facing (45 deg downward tilt), normalized `clamp(depth / 20, 0, 1)`.

**Depth camera**: 45 deg downward tilt (body-frame), 90 deg FOV, 0.1m below drone center. Dual-path rendering: BatchRenderer uses `camera.attach()` + `move_to_attach()` (single call for all envs); Rasterizer uses per-env serial loop with obstacle position swapping. `use_batch_renderer` config: `"auto"` (default), `True`/`"batch"`, `False`/`"rasterizer"`.

**Model**: `CNNModel` (3 conv layers: [32,64,128], kernels [8,4,3], strides [4,2,1], batch norm, global avg pool -> 128-dim latent) + MLP [256,256]. CNN shared between actor and critic via `share_cnn_encoders`.

**Obstacles**: 8 `gs.morphs.Box(1x1x2m)`, distance-based collision (radius 0.3m), obstacle proximity penalty within safety radius (3.0m). Post-curriculum uses strategic placement with 1 guaranteed path blocker on the spawn->target line; curriculum phase uses sparse random placement.

**Rewards**: distance (-5), time (-0.5), obstacle_proximity (-10), crash (-100), obstacle_collision (-150), success (+200). Per-step rewards scaled by dt.

## Running on HPC (Leipzig cluster)

```bash
# Sync code to HPC (never run by Claude — user manages manually)
./hpc/sync_to_hpc.sh

# On HPC: first-time setup
source ~/genesis_v04/hpc/setup_env.sh

# On HPC: reload env in new session
source ~/genesis_v04/hpc/setup_env.sh --load

# Submit batch training
./hpc/submit_training.sh global_coordinate --batch 4096 --iters 401
./hpc/submit_training.sh obstacle_avoidance --batch 512 --iters 401

# Pull results back
./hpc/sync_from_hpc.sh
```

GPU partitions: `paula` (A30 24GB Ampere, 2-day limit, supports BatchRenderer), `clara` (V100 32GB Volta, 2-day, Rasterizer only), `clara-long` (V100, 10-day, Rasterizer only).
Standalone job scripts in `hpc/jobs/`. Setup details in `hpc/README.md`.

## Forbidden Actions

- **NEVER execute HPC sync/SSH commands** — do not run `sync_to_hpc.sh`, `sync_from_hpc.sh`, `submit_training.sh`, `run_interactive.sh`, or any rsync/scp/ssh targeting the HPC cluster. The user manages all HPC operations manually. Claude may *create or edit* these scripts but must never *execute* them.

## Plugins

- **`genesis-world`** plugin (`~/.claude/plugins/genesis-world/`) — Genesis physics simulator API reference. Consult when writing simulation code.
- **`rsl-rl`** plugin (`~/.claude/plugins/rsl-rl/`) — rsl-rl reinforcement learning library API reference. Consult when working with PPO training, OnPolicyRunner, actor-critic models, rollout storage, or any rsl-rl integration code.
