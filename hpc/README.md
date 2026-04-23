# HPC Leipzig — Genesis Training Setup

Running RL training on the Leipzig University HPC cluster (A30/V100 GPUs).

## Prerequisites

- VPN access to Uni Leipzig network
- SSH key configured: `Host hpc` in `~/.ssh/config` pointing to `login01.sc.uni-leipzig.de`
- Both should already be set up

## First-Time Setup

1. **Sync code to HPC:**
   ```bash
   ./hpc/sync_to_hpc.sh
   ```

2. **SSH into the cluster and run setup:**
   ```bash
   ssh hpc
   source ~/genesis_v04/hpc/setup_env.sh
   ```
   This creates the `ba_v04` conda env with PyTorch (cu126), Genesis v0.4.3, rsl-rl, W&B, etc.

3. **Create a workspace** (optional, recommended for large training runs):
   ```bash
   ws_allocate genesis-training 30    # 30-day renewable, up to 5TB on /work
   ```

## Syncing Code

```bash
# Push local code to HPC (excludes .git, wandb, logs, __pycache__)
./hpc/sync_to_hpc.sh

# Preview what would sync
./hpc/sync_to_hpc.sh --dry-run

# Pull training results (logs/checkpoints) back
./hpc/sync_from_hpc.sh

# Pull specific prototype only
./hpc/sync_from_hpc.sh global_coordinate
```

## Running Training

### Interactive (smoke test)

```bash
# Request a 1-hour GPU session
./hpc/run_interactive.sh              # default: clara, v100, 1h
./hpc/run_interactive.sh paula a30 2  # paula partition, a30, 2h

# Once allocated:
source ~/genesis_v04/hpc/setup_env.sh --load
cd ~/genesis_v04/prototyp_global_coordinate
python train_rl_wb.py -B 4 --max_iterations 5
```

### Batch Job (full training)

**Option A: Standalone job scripts** (edit parameters directly):
```bash
ssh hpc
sbatch ~/genesis_v04/hpc/jobs/train_global_coord.job
sbatch ~/genesis_v04/hpc/jobs/train_obstacle_avoidance.job
sbatch ~/genesis_v04/hpc/jobs/train_corridor_navigation.job
```

**Option B: Interactive submission wrapper** (prompts for prototype, env version, batch size, partition, etc.):
```bash
ssh hpc
~/genesis_v04/hpc/submit_training.sh

# Continue an existing run from the latest (or chosen) checkpoint
~/genesis_v04/hpc/continue_training.sh

# Submit an evaluation / landing-video job for a trained checkpoint
~/genesis_v04/hpc/submit_evaluation.sh
```

## Monitoring

```bash
# Check job status
ssh hpc squeue -u fk67rahe

# Detailed job info after completion
ssh hpc reportseff <jobid>

# Check GPU availability
ssh hpc sc_nodes_gap --gpu

# Check storage quotas
ssh hpc sc_quotas

# TensorBoard via SSH port forwarding
ssh -L 6006:localhost:6006 hpc
# Then on HPC:
source ~/genesis_v04/hpc/setup_env.sh --load
tensorboard --logdir ~/genesis_v04/prototyp_global_coordinate/logs --port 6006
# Open http://localhost:6006 locally

# W&B dashboard (no port forwarding needed)
# Runs logged to wandb.ai automatically
```

## Retrieving Results

```bash
# All prototypes
./hpc/sync_from_hpc.sh

# Specific prototype
./hpc/sync_from_hpc.sh global_coordinate

# Dry run
./hpc/sync_from_hpc.sh --dry-run
```

Results land in `hpc_results/` at repo root.

## Useful Commands

| Command | Description |
|---------|-------------|
| `squeue -u $USER` | List your running/pending jobs |
| `scancel <jobid>` | Cancel a job |
| `scancel -u $USER` | Cancel all your jobs |
| `sinfo -p paula` | Show partition status |
| `sc_nodes_gap --gpu` | GPU availability across partitions |
| `sc_quotas` | Storage quota usage |
| `reportseff <jobid>` | Resource usage report after job ends |
| `ws_list` | List your workspaces |
| `ws_allocate <name> <days>` | Create workspace on /work |
| `ws_extend <name> <days>` | Extend workspace expiry |

## GPU Partitions

| Partition | GPUs | VRAM | Time Limit |
|-----------|------|------|------------|
| `paula` | 12 nodes x 8x A30 | 24 GB | 2 days |
| `clara` | V100 | 32 GB | 2 days |
| `clara-long` | V100 | 32 GB | 10 days |

## Batch Size Guidelines

| Prototype | GPU | Recommended `-B` |
|-----------|-----|-------------------|
| global_coordinate | A30 (24GB) | 4096 |
| global_coordinate | V100 (32GB) | 4096 |
| obstacle_avoidance | A30 (24GB) | 512 |
| obstacle_avoidance | V100 (32GB) | 512-1024 |
| corridor_navigation | A30 (24GB) | 256 |
| corridor_navigation | V100 (32GB) | 256-512 |
