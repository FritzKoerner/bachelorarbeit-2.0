"""Multi-GPU distributed training for obstacle avoidance.

Launch with torchrun:
    torchrun --nproc_per_node=2 train_rl_multigpu.py -B 32 --max_iterations 401

Each GPU runs its own Genesis scene + env. rsl-rl handles gradient
all-reduce across processes via NCCL. Effective batch = num_envs × num_gpus.
"""

# --- Multi-GPU device isolation (BEFORE any CUDA init) ---
# Genesis's Quadrants backend allocates internal buffers on cuda:0 regardless
# of torch.cuda.set_device(). Fix: restrict each process to see only its
# assigned GPU via CUDA_VISIBLE_DEVICES so cuda:0 maps to the correct
# physical GPU per rank.
import os as _os

_world_size = int(_os.environ.get("WORLD_SIZE", "1"))
if _world_size > 1:
    _local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
    # Respect SLURM's GPU allocation (e.g. CUDA_VISIBLE_DEVICES="2,3")
    _visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _visible:
        _gpu_list = [g.strip() for g in _visible.split(",")]
        _os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_list[_local_rank]
    else:
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)
    # Each process now sees 1 GPU as cuda:0.
    # Override LOCAL_RANK so rsl-rl's device validation (device == cuda:LOCAL_RANK) passes.
    _os.environ["LOCAL_RANK"] = "0"

import argparse
import copy
import os
import pickle
import shutil

from rsl_rl.runners import OnPolicyRunner

import torch
import genesis as gs

from envs.obstacle_avoidance_env import ObstacleAvoidanceEnv


class DictConfig(dict):
    """Thin dict wrapper so rsl-rl's WandbSummaryWriter.store_config() works."""

    def to_dict(self):
        return dict(self)


def get_train_cfg(exp_name, max_iterations):
    return {
        "num_steps_per_env": 120,  # ~1 episode (30s / 0.25s decision_dt)
        "save_interval": 100,
        "max_iterations": max_iterations,

        # W&B logging
        "logger": "wandb",
        "wandb_project": "obstacle-avoidance",
        "experiment_name": exp_name,

        # Algorithm
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": None,
            "entropy_coef": 0.001,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "fixed",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "share_cnn_encoders": True,
            "rnd_cfg": None,
        },

        # Actor (CNN + MLP)
        "actor": {
            "class_name": "CNNModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
            "cnn_cfg": {
                "depth": {
                    "output_channels": [32, 64, 128],
                    "kernel_size": [8, 4, 3],
                    "stride": [4, 2, 1],
                    "norm": "none",
                    "activation": "elu",
                    "global_pool": "avg",
                    "flatten": True,
                },
            },
        },

        # Critic (CNN shared from actor + separate MLP)
        "critic": {
            "class_name": "CNNModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
            "obs_normalization": True,
        },

        # Observation routing
        "obs_groups": {
            "actor":  ["state", "depth"],
            "critic": ["state", "depth"],
        },
    }


def get_cfgs():
    env_cfg = DictConfig({
        "num_actions": 4,
        "episode_length_s": 60.0,
        "decimation": 300,             # PID runs at 100 Hz, RL decides every 300 steps (3 s)
        "action_scales": [1.0, 1.0, 1.0],
        # Drone spawn
        "spawn_offset": 5.0,
        "spawn_height_min": 10.0,
        "spawn_height_max": 10.0,
        # Target
        "target_x_range": [3.0, 3.0],
        "target_y_range": [3.0, 3.0],
        "target_z_range": [1.0, 1.0],
        # Obstacle curriculum: no obstacles for first N steps, then strategic placement
        "curriculum_steps": 3840000,
        "curriculum_n_obstacles": 5,
        # Success: within radius of target for the entire decision step
        "hover_radius": 0.3,
        # Obstacles
        "num_obstacles": 8,
        "obstacle_size": [1.0, 1.0, 2.0],
        "collision_radius": 0.3,
        "safety_radius": 1.0,
        # Post-curriculum strategic placement
        "n_corridor_obstacles": 3,
        "n_ring_obstacles": 4,
        "ring_radius_range": [1.5, 3.5],
        "corridor_lateral_offset": 2.0,
        "post_curriculum_range": 5.0,
        # Depth camera
        "render_interval": 1,
        "max_depth": 20.0,
        # Visualisation
        "visualize_target": False,
        "env_spacing": 40.0,
        # Renderer
        "use_batch_renderer": "auto",
        # PID
        "pid_params": {
            "base_rpm": 1789.2,
            "max_rpm": 2700.0,
            "max_tilt": 30.0,
            "max_vel_xy": 5.0,
            "max_vel_z": 3.0,
            "pid_params_pos_x": [1.0, 0.0, 0.7],
            "pid_params_pos_y": [1.0, 0.0, 0.7],
            "pid_params_pos_z": [1.5, 0.0, 1.0],
            "pid_params_vel_x": [16.0, 0.0, 8.0],
            "pid_params_vel_y": [16.0, 0.0, 8.0],
            "pid_params_vel_z": [100.0, 2.0, 10.0],
            "pid_params_roll":  [6.0, 0.0, 3.0],
            "pid_params_pitch": [6.0, 0.0, 3.0],
            "pid_params_yaw":   [0.5, 0.0, 0.8],
        },
    })

    obs_cfg = {
        "num_state_obs": 17,
        "depth_res": 64,
        "depth_stack_size": 1,
        "obs_scales": {
            "rel_pos": 1 / 15.0,
            "lin_vel": 0.4,
            "ang_vel": 1 / 3.14159,
        },
    }

    reward_cfg = {
        "reward_scales": {
            "distance":           -5.0,
            "time":               -0.5,
            "obstacle_proximity": -6.0,
            "crash":            -100.0,
            "success":           200.0,
        },
    }

    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="obstacle-avoidance-multigpu")
    parser.add_argument("-B", "--num_envs", type=int, default=128)
    parser.add_argument("--max_iterations", type=int, default=401)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    # RANK is preserved (not overridden); LOCAL_RANK was remapped to 0 above
    global_rank = int(os.environ.get("RANK", "0"))
    is_rank0 = global_rank == 0

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Only rank 0 manages log directory and config dump
    if is_rank0:
        if args.resume is None:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )
    else:
        os.makedirs(log_dir, exist_ok=True)
        train_cfg["logger"] = "tensorboard"

    env = ObstacleAvoidanceEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
    )
    env.build()

    # All processes see cuda:0 (each mapped to a different physical GPU).
    # rsl-rl reads WORLD_SIZE/RANK and auto-configures NCCL gradient all-reduce.
    runner = OnPolicyRunner(
        env,
        copy.deepcopy(train_cfg),
        log_dir,
        device="cuda:0",
    )
    if args.resume is not None:
        runner.load(args.resume)
        if is_rank0:
            print(f"Resumed from {args.resume} at iteration {runner.current_learning_iteration}")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# 2-GPU training (32 envs per GPU = 64 total)
torchrun --nproc_per_node=2 train_rl_multigpu.py -B 32 --max_iterations 401

# 4-GPU training on HPC (paula partition, A30s)
torchrun --nproc_per_node=4 train_rl_multigpu.py -B 16 --max_iterations 401

# Resume from checkpoint
torchrun --nproc_per_node=2 train_rl_multigpu.py --resume logs/obstacle-avoidance-multigpu/model_200.pt
"""
