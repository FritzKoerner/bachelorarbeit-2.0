# Vineyard Drone Landing — RL with Genesis

Bachelor thesis project: training a quadrotor to autonomously land in vineyard
soil strips using deep reinforcement learning on the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
physics simulator (v0.4.3).

The agent outputs a **target position offset** every decision step; a cascading
PID controller (position → velocity → attitude → RPM) tracks that target and
produces the four motor RPMs. PPO is provided by [`rsl-rl`](https://github.com/leggedrobotics/rsl_rl)
v5.0.1, with W&B logging and HPC-batch training on the Leipzig University
cluster.

---

## Repository Structure

```
genesis_v04/
├── assets/                          # Shared drone URDF + meshes (draugas)
├── prototyp_global_coordinate/      # Coordinate-only landing (state vector → PPO)
├── prototyp_obstacle_avoidance/     # + depth camera + obstacles (CNN+state → PPO)
├── hpc/                             # SLURM wrappers, env setup, sync scripts
├── logs/                            # SLURM stdout/stderr from submitted jobs
├── hpc_results/                     # Pulled-back checkpoints from HPC runs
└── CLAUDE.md                        # Detailed architecture + API gotchas
```

Each `prototyp_*/` directory is self-contained with its own `envs/`,
`controllers/`, training/eval scripts, and run logs.

---

## Quick Start

### 1. Install

```bash
# One-time: create the conda env (Genesis 0.4.3, PyTorch CUDA 12.6, rsl-rl, W&B, …)
conda create -n ba_v04 python=3.13
conda activate ba_v04
pip install genesis-world==0.4.3 rsl-rl-lib==5.0.1 tensordict torch numpy scipy \
            pyyaml tensorboard wandb
```

> Genesis automatically pulls in `quadrants==0.4.4` (its physics backend),
> `moviepy`, `opencv-python`, and `mujoco` as transitive dependencies.

### 2. Smoke test (with viewer)

```bash
cd prototyp_global_coordinate
python train_rl_wb.py -B 4 -v --max_iterations 5
```

If the viewer opens and the drone twitches around, you're set. The first build
triggers a one-time CoACD convex decomposition (~60 s) and JIT compilation —
subsequent runs reuse the cache.

### 3. Train (headless)

```bash
# Coordinate-only landing (4096 envs, ~5 min on a single A30)
cd prototyp_global_coordinate
python train_rl_wb.py -B 4096 --max_iterations 401

# Obstacle avoidance with depth camera (256 envs, hours-long)
cd prototyp_obstacle_avoidance
python train_rl_wb.py -B 256 --max_iterations 8001 --env-v2
```

### 4. Evaluate

```bash
# Auto-finds the latest checkpoint in logs/<exp_name>/
python eval_rl_wb.py

# Or pick a specific iteration / single visual episode
python eval_rl_wb.py --ckpt 400
python eval_rl_wb.py --vis
```

---

## The Two Prototypes

### `prototyp_global_coordinate/`
Drone learns to fly to a randomized target position in a 10×10 m square at 1 m
altitude. Pure 17-D state observation (relative position, quaternion, velocities,
last actions). Used to validate the controller stack and reward shaping in
isolation from perception.

### `prototyp_obstacle_avoidance/`
Same task plus 8 randomly-placed obstacles and a forward-tilted (45°) depth
camera mounted on the drone. Observation is a `TensorDict({"state": (n, 17),
"depth": (n, 1, 64, 64)})`. CNN encoder (3 conv layers → 128-d latent) is
**shared between actor and critic** via rsl-rl's `share_cnn_encoders`.

Two **post-curriculum obstacle layouts** can be selected at training time
(`--placement strategic|vineyard`) or overridden at eval time:

| Strategy | Description |
|---|---|
| `strategic` | Corridor-blocker on the spawn→target line + ring of obstacles around the target |
| `vineyard`  | Two parallel rows of tall (3 m) boxes flanking the target, mimicking grape rows |

Both env versions also support a `v2` reward formulation
(`--env-v2`: progress + close rewards, no `dt`-scaling) which trains more
reliably than the v1 distance-penalty formulation.

---

## HPC Training (Leipzig cluster)

For Bachelorarbeit-scale runs (millions of env-steps), training happens on the
Leipzig HPC cluster. Three interactive wrappers do everything:

```bash
~/genesis_v04/hpc/submit_training.sh      # new training run
~/genesis_v04/hpc/continue_training.sh    # resume from a checkpoint
~/genesis_v04/hpc/submit_evaluation.sh    # eval + landing video
```

They prompt for prototype, env version, batch size, partition, time limit, etc.,
then build and `sbatch` a SLURM job script. See [`hpc/README.md`](hpc/README.md)
for the full first-time setup, syncing, and partition guide.

> ⚠️ **Renderer must match the partition.** The BatchRenderer requires
> compute capability ≥ 7.5 (paula / A30). Volta GPUs (clara / V100) fall back to
> the per-env Rasterizer path — eval on the same partition you trained on or the
> depth distribution shifts.

---

## Project-Specific Tools

| Script | Purpose |
|---|---|
| `train_rl_wb.py` | Main PPO training (W&B-logged) |
| `eval_rl_wb.py` | Stats, success rate, W&B-logged eval over many episodes |
| `record_landing.py` | Record a single landing episode as MP4 (BatchRenderer + replay scene) |
| `visualize_paths.py` | Trajectory plots / GIFs across multiple checkpoints |
| `visualize_obstacle_setup.py` | Static visualization of strategic + vineyard layouts |
| `debug_depth.py` (obstacle_avoidance) | Validate depth-camera rendering and CNN gradient flow |
| `debug_rl.py` | Per-step env+policy trace (CSV, stochastic/random/zero policies) |
| `debug_pid.py` (global_coordinate) | PID step-response test, useful for re-tuning |

---

## Key Design Notes

- **Action space**: 4-D float in `[-1, 1]` interpreted as a target offset:
  `target_xyz = current_pos + action[:3] * action_scales`, plus `target_yaw`.
- **Custom drone** (`assets/robots/draugas/`): mass 0.714 kg, thrust-to-weight
  2.25, base hover RPM 1789.2, max RPM 2700.
- **Episode**: 60 s max; success = hover within 0.3 m of target for one full
  decision step; crash = altitude < 0.2 m, tilt > 60°, or distance > 50 m.
- **Reward (v1)**: distance penalty −5, time penalty −0.5, crash −100,
  success +200. **Reward (v2)**: progress +5, close-bonus +1·exp(−d), crash
  −100, success +200, no `dt`-scaling.
- **rsl-rl pickle quirk**: `cfgs.pkl` contains `DictConfig` instances. Any
  script loading it must `from train_rl_wb import DictConfig` first.

For deeper architectural notes, Genesis API gotchas (Quadrants backend, propeller-RPM
typo, BatchRenderer constraints, etc.), and the exact reward formulas, see
**[CLAUDE.md](CLAUDE.md)**.

---

## Status

- ✅ Continuous PPO training pipeline (rsl-rl + W&B) on both prototypes
- ✅ HPC submission, monitoring, and result-pull scripts
- ✅ Vineyard placement strategy with eval-time override
- ✅ Per-substep video recording for paper figures
- 🚧 Final hyperparameter sweep on vineyard configuration
