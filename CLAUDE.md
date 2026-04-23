# CLAUDE.md

## Project Overview

Bachelor thesis project: Training a drone to autonomously land in vineyard soil strips using reinforcement learning with the Genesis physics simulator (`genesis-world` v0.4.3).

Migrated from v0.3.13 — see the original repo at `../genesis/` for history and legacy prototypes.

## Repository Structure

- **prototyp_global_coordinate/** — Global coordinate-based landing with cascading PID controller. Uses PPO via rsl-rl.
- **prototyp_obstacle_avoidance/** — CNN depth-map obstacle avoidance. Extends global_coordinate with random obstacles, downward-facing depth camera, and rsl-rl v5.0.1 `CNNModel` + `share_cnn_encoders`.
- **prototyp_corridor_navigation/** — Forked from obstacle_avoidance env v2. Fixed-axis corridor slalom along +X with mixed-shape obstacles (2 boxes, 2 spheres, 2 cylinders, 2 pillar cylinders). Hard termination on leaving the corridor bounding box; no v1/v2 split, no strategic/vineyard placers.
- **assets/** — Shared drone URDF + meshes (`assets/robots/draugas/`). Referenced from prototypes as `../assets/`.
- **hpc/** — HPC Leipzig cluster scripts: env setup, code sync, SLURM job submission.

Each prototype is self-contained with its own `envs/`, `controllers/`, etc.

## Running (prototyp_global_coordinate)

```bash
cd prototyp_global_coordinate

# PPO training (rsl-rl + W&B, headless, 4096 envs)
python train_rl_wb.py -B 4096 --max_iterations 401

# PPO smoke test with viewer
python train_rl_wb.py -B 4 -v --max_iterations 5

# Evaluation (auto-finds latest checkpoint)
python eval_rl_wb.py

# Visualization
python visualize_paths.py --ckpt 100 300                  # matplotlib + screenshots
python visualize_paths.py --ckpt 300 --video              # + landing GIF with trail
python visualize_paths.py --ckpt 100 300 --no_render      # matplotlib only

# Record landing MP4 for HPC-back visual check
python record_landing.py --ckpt 300

# Eval with a custom run label (artifacts land in logs/{exp}/evals/{name}/)
python eval_rl_wb.py --ckpt 300 --name baseline-seed42
python record_landing.py --ckpt 300 --name baseline-seed42
```

## Dependencies

Conda environment: `ba_v04`. Key packages: `genesis-world==0.4.3` (installs `quadrants==0.4.4`), `torch>=2.0.0`, `numpy`, `scipy`, `pyyaml`, `tensorboard`, `wandb`, `rsl-rl-lib==5.0.1`, `tensordict`. Genesis v0.4.3 auto-installs `moviepy`, `opencv-python`, `mujoco` as deps.

## Genesis API Gotchas (v0.4.3)

- **`get_link("main_body")`** for URDF root link reference — use the actual URDF link name, not "base".
- **Quadrants backend** (`quadrants==0.4.4`) replaces Taichi. Env variable prefix: `QD_` (not `TI_`). One-time JIT recompilation on first run.
- **CoACD convex decomposition** runs automatically on first build (~60s for drone mesh). Cached for subsequent runs.
- **Two-phase construction**: `gs.init(backend=gs.gpu)` once, then `env.build()`. `scene.build()` triggers JIT; must be called before stepping.
- `camera.render(depth=True, segmentation=True)` returns a **tuple** `(rgb, depth, segmentation, normal)`, not a dict.
- **Typo in API**: `drone.set_propellels_rpm(rpms)` (propellels, not propellers).
- Custom URDF drones need explicit `propellers_link_name` and `propellers_spin`.
- Conventions: right-handed Z-up, quaternions w-x-y-z, Euler angles in degrees (scipy extrinsic x-y-z), all state tensors `(n_envs, ...)` on `gs.device`.
- **Auto-reset contaminates state**: when `step()` returns `done=True`, the env has already reset — `base_pos` reflects the new spawn, not the terminal position. Record positions BEFORE `step()`.
- **`gs.morphs.Box` with `fixed=True`**: supports per-env `set_pos(new_pos, envs_idx=idx, zero_velocity=True)` for obstacle randomization. Use `collision=False` when handling collisions via distance checks.
- **BatchRenderer** requires Turing+ GPU (compute capability >= 7.5). Crashes with `abort()` on Volta — check capability before use. Fallback: Rasterizer with `env_separate_rigid=True`.
- **Rasterizer** does NOT support batched cameras. Use `env_separate_rigid=True` which renders all envs in one `render()` call. Known depth bug: `depth[i]` for i>0 shows env 0's rigid bodies — workaround in `_render_depth_rasterizer()` swaps env 0 positions per render.
- **BatchRenderer requires uniform camera resolution** — all cameras in one scene must share the same `res`, else `scene.build()` raises. Multi-res recordings need separate scenes (see `prototyp_obstacle_avoidance/record_landing.py` two-pass design).
- **Post-build camera addition**: `scene.add_camera()` is `@assert_not_built`-gated. Workaround: `scene._visualizer.add_camera()` adds a pyrender-only node (Rasterizer only, not BatchRenderer-compatible). Pattern: `visualize_paths.py:_add_camera`.
- **`env_separate_rigid=True` render output is batched** `(n_envs, H, W, C)` for rgb and `(n_envs, H, W)` for depth — slice `[0]` / check `ndim==4` before feeding to cv2/PIL/numpy viz code.
- **Viewer mode (`-v`) crashes on window close**: closing the GUI mid-run raises `GenesisException` from `viewer.update()` and kills training. Dev-only — HPC must run headless.
- **v0.4.4+ latent breakage (we're pinned to 0.4.3)**: `set_quat` default flips to `relative=True` (6 call sites across env files assume absolute), and `set_propellels_rpm` was renamed to `set_propellers_rpm` with no alias. Both need explicit fixes before any `genesis-world` bump.
- **`scene.add_camera` defaults `far=20.0`** (Rasterizer near/far plane). Any scene wider than ~15 m from the camera will silently clip distant geometry. Pass `far=max(cam_to_center * 3, 100.0)` or similar for wide-field third-person views. (POV cams mirroring the CNN's depth camera should keep the small default to match the CNN's `max_depth` clamp.)

## Architecture

The RL agent outputs a target position; a cascading PID controller (position->velocity->attitude->RPM, three nested loops) tracks it and produces motor RPMs.

Key files in `prototyp_global_coordinate/`:
- `envs/coordinate_landing_env.py` — rsl-rl compatible env (reset/step/reward)
- `controllers/pid_controller.py` — CascadingPIDController

## rsl-rl Gotchas

- `OnPolicyRunner.__init__` calls `.pop()` on config dicts, mutating them. Always pass `copy.deepcopy(train_cfg)` when creating multiple runners.
- `get_inference_policy()` returns the actor model directly — works with both `MLPModel` (flat TensorDict) and `CNNModel` (multi-key TensorDict including image obs).
- The runner accesses `cfg["algorithm"]["rnd_cfg"]` unconditionally — always include `"rnd_cfg": None` in algorithm config even when not using RND.
- **GaussianDistribution defaults to `std_type="scalar"`** which can go negative. Use `"std_type": "log"` for stability, especially with low/zero `entropy_coef`.
- **`entropy_coef` must be small** relative to per-step rewards (which are dt-scaled to ~0.05/step). Values like 0.01 cause std explosion with log-space; 0.001 works.
- **`share_cnn_encoders: True`** goes in `"algorithm"` config, NOT in actor/critic. Critic config should omit `cnn_cfg` entirely — PPO injects `actor.cnns` automatically.
- **W&B logger needs `.to_dict()` on configs** — `WandbSummaryWriter.store_config()` calls `env_cfg.to_dict()`, which fails on plain dicts. Wrap with a `DictConfig(dict)` subclass that adds `to_dict()`. See `train_rl_wb.py`.
- **Pickle + DictConfig**: `cfgs.pkl` from W&B runs contains `DictConfig` instances. Any script loading it (e.g. `eval_rl_wb.py`) must `from train_rl_wb import DictConfig` or pickle will fail with `AttributeError`.
- **`cfgs.pkl` format**: `[env_cfg, obs_cfg, reward_cfg, train_cfg]` — a 4-element list, not a dict.

## Observation & Action Spaces

- **Observation**: `(n_envs, 17)` — `rel_pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + last_actions(4)`, each clipped and scaled
  - Observation scales: `rel_pos * 1/15`, `lin_vel * 0.4`, `ang_vel * 1/pi` (both prototypes, both train scripts)
- **Actions**: `(n_envs, 4)` float in `[-1, 1]` — `[ax, ay, az, ayaw]`
  - `target_xyz = current_pos + action[:3] * action_scales` (offset from current position)
  - `target_yaw = ayaw * 180.0` (degrees)

## Key Parameters

- Drone: custom "draugas" URDF (mass 0.714kg, thrust2weight 2.25), base hover RPM 1789.2, max RPM 2700
- Spawn: height 10m fixed, drone offset +/-5m
- Target: `train_rl_wb.py` randomizes in 10x10m square at 1m height
- Success: hover within 0.3m of target for the entire decision step (decimation substeps)
- Crash: height < 0.2m, tilt > 60°, or distance from target > 50m
- Rewards: distance penalty (-5.0), time penalty (-0.5), crash (-100), success (+200)

## Curriculum Coupling

`curriculum_steps` counts `env.step()` calls, not runner iterations. Formula: `curriculum_steps = target_iteration × num_steps_per_env`. Changing `num_steps_per_env` shifts the curriculum transition point.

## Eval W&B convention

Eval runs live in their own per-prototype, per-env-version W&B projects — separate from training — so eval dashboards don't mix with training noise:

- `eval-drone-continuous-v{1|2}` (global_coordinate)
- `eval-obstacle-avoidance-v{1|2}` (obstacle_avoidance)

Env version is auto-detected from `cfgs.pkl` reward keys (`"progress"` in `reward_scales` → v2). `--wandb_project` still works as an explicit override.

`eval_rl_wb.py` and `record_landing.py` both accept `--name <custom>`. All artifacts (`eval_stats.png`, `landing_ckpt_{ckpt}*.mp4`) land in `logs/{exp}/evals/{name}/`. Default `--name` is `iter{ckpt}` — reruns on the same ckpt overwrite, so use `--name` to compare protocols (e.g. `--name baseline-hard`, `--name vineyard-seed7`).

## V2 Env Pattern

- Adding a new env version: create `envs/*_v2.py`, update `train_rl_wb.py` (import, `get_cfgs(env_v2=)`, `--env-v2` flag, env class selection), update `eval_rl_wb.py` (detect from `cfgs.pkl` reward keys).
- Both prototypes support `--env-v2` via their respective `train_rl_wb.py`.

## Running (prototyp_obstacle_avoidance)

```bash
cd prototyp_obstacle_avoidance

# PPO training with W&B (headless, 256 envs, v2 rewards)
python train_rl_wb.py -B 256 --max_iterations 8001 --env-v2

# PPO training (v1 rewards)
python train_rl_wb.py -B 256 --max_iterations 8001

# Smoke test with viewer
python train_rl_wb.py -B 4 -v --max_iterations 5

# Evaluation
python eval_rl_wb.py                                        # auto-finds latest checkpoint
python eval_rl_wb.py --vis                                  # single visual episode

# Visualization
python visualize_paths.py --ckpt 100 300 --no_render
python visualize_paths.py --ckpt 300 --video

# Record landing MP4s (3rd-person + POV RGB + POV depth) for HPC-back visual check
python record_landing.py --ckpt 300 [--no-pov] [--placement strategic|vineyard] [--no-obstacles]

# Eval with a custom run label (artifacts land in logs/{exp}/evals/{name}/)
python eval_rl_wb.py --ckpt 300 --name baseline-hard
python record_landing.py --ckpt 300 --name baseline-hard
```

`train_rl_multigpu.py` is parked — currently unused, kept on disk for a possible future multi-GPU revival. Do not list it as a first-class command; its `CUDA_VISIBLE_DEVICES` remap workaround is self-documented at the top of the script.

## prototyp_obstacle_avoidance Architecture

**Env Versions**: v1 (default) uses dt-scaled distance+time penalties. v2 (`--env-v2`) uses fixed-weight progress+close rewards (no dt-scaling), keeps obstacle_proximity unchanged. `eval_rl_wb.py` auto-detects v2 via `"progress" in reward_scales` from `cfgs.pkl`.

Key files (mirrors `prototyp_global_coordinate/` structure):
- `envs/obstacle_avoidance_env.py` — Env: obstacles + depth camera + TensorDict
- `debug_depth.py` — Validates depth camera rendering and CNN gradient flow
- `visualize_obstacle_setup.py` — Validates strategic obstacle placement effectiveness

**Observations**: `TensorDict({"state": (n, 17), "depth": (n, 3, 64, 64)})`. State is identical 17-dim vector; depth is forward-facing (45 deg downward tilt), 3-frame stack, normalized `clamp(depth / 20, 0, 1)`.

**Depth camera**: 45 deg downward tilt (body-frame), 90 deg FOV, 0.1m below drone center. Dual-path rendering: BatchRenderer uses `camera.attach()` + `move_to_attach()` (single call for all envs); Rasterizer uses per-env serial loop with obstacle position swapping. `use_batch_renderer` config: `"auto"` (default), `True`/`"batch"`, `False`/`"rasterizer"`.

**Model**: `CNNModel` (3 conv layers: [32,64,128], kernels [8,4,3], strides [4,2,1], batch norm, global avg pool -> 128-dim latent) + MLP [256,256]. CNN shared between actor and critic via `share_cnn_encoders`.

**Obstacles**: 8 `gs.morphs.Box(1x1x2m)`, distance-based collision (radius 0.3m), obstacle proximity penalty within safety radius (3.0m). Post-curriculum uses strategic placement with 1 guaranteed path blocker on the spawn->target line; curriculum phase uses sparse random placement.

**Rewards (v1)**: distance (+5, delta: prev_dist − curr_dist), time (-0.5), obstacle_proximity (-6), crash (-100, includes obstacle collision), success (+200). Per-step rewards scaled by dt.

**Rewards (v2)**: progress (+5, delta-distance), close (+1, exp(-dist)), obstacle_proximity (-6), crash (-100), success (+200). No dt-scaling, no time penalty.

**Video recording (`record_landing.py`)**: two-scene design (policy scene + viz scene) because BatchRenderer forbids mixed resolutions. Pass 2 streams each rendered frame directly to `cv2.VideoWriter` (lazy-init on first frame) — never accumulate into a list. A default 60 s / decimation=300 episode is ~6000 `substep_callback` fires × full-res frames; buffering them all OOM-kills SLURM jobs with tight `--mem`.

## Running (prototyp_corridor_navigation)

```bash
cd prototyp_corridor_navigation

# PPO training with W&B (headless, 256 envs)
python train_rl_wb.py -B 256 --max_iterations 8001

# Smoke test with viewer (4 envs, 5 iterations)
python train_rl_wb.py -B 4 -v --max_iterations 5

# Smoke test with curriculum disabled (obstacles visible from step 0)
python train_rl_wb.py -B 4 -v --max_iterations 5 --curriculum-iterations 0
```

Eval/record/visualize scripts are not yet ported from `prototyp_obstacle_avoidance/`; add them once training is stable and `cfgs.pkl` format has settled.

## prototyp_corridor_navigation Architecture

Fixed-axis corridor slalom along +X. Single env class (`CorridorNavigationEnv`), no v1/v2 split. State/action/depth/observation scales identical to `prototyp_obstacle_avoidance` env v2.

**Corridor geometry (metres)**: X `[0.0, 13.0]`, Y `[-3.0, 3.0]`, Z `[0.3, 6.0]`. Spawn uniform in X `[0.5, 5.5]` × Y `[-2.5, 2.5]` at Z `5.0`. Target fixed at `(12.0, 0.0, 1.0)`. Leaving the bounding box terminates the episode (contributes to both `crash` and `out_of_corridor` reward components — the latter is logged separately for diagnosis).

**Obstacle mix (8 total)**: 2 Boxes, 2 Spheres, 2 Cylinders, 2 thin-tall pillar Cylinders. Sizes configurable via `env_cfg` keys `corridor_{box_sizes, sphere_radii, cylinder_specs, pillar_specs}`. Placed in 4 X-slices (centres `[3.5, 6.0, 8.5, 11.0]`), 2 per slice on opposite Y sides with a guaranteed ≥ 2.5 m Y-gap so the drone always has a feasible path. Shape-interleaved layout gives each slice a different shape pair.

**Shape-aware collision**: `_compute_obstacle_distances()` computes per-shape distances (box AABB / sphere radial / cylinder radial+axial) for every obstacle and dispatches via `torch.where` on `self.obstacle_shape_type`. Fully vectorised across the `n_obs` axis — no Python loop per obstacle.

**Reward scales**: `progress +0.5`, `close +0.1`, `obstacle_proximity -0.6`, `crash -10.0`, `success +20.0`, `out_of_corridor -10.0` (same magnitude as `crash`, same sign convention, logged separately).

**Curriculum**: identical to obstacle_avoidance — obstacles stay underground (`z = -100`) until `global_step >= curriculum_steps`. Corridor bounds are enforced from step 0, so the agent learns to stay in the box even during curriculum warm-up.

## Running on HPC (Leipzig cluster)

```bash
# Sync code to HPC (never run by Claude — user manages manually)
./hpc/sync_to_hpc.sh

# On HPC: first-time setup
source ~/genesis_v04/hpc/setup_env.sh

# On HPC: reload env in new session
source ~/genesis_v04/hpc/setup_env.sh --load

# Submit batch training (interactive — prompts for prototype, env version, batch size, etc.)
./hpc/submit_training.sh

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
