import argparse
import copy
import os
import pickle
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from envs.obstacle_avoidance_env import ObstacleAvoidanceEnv
from envs.obstacle_avoidance_env_v2 import ObstacleAvoidanceEnvV2


class DictConfig(dict):
    """Thin dict wrapper so rsl-rl's WandbSummaryWriter.store_config() works.

    store_config tries env_cfg.to_dict() first, then dataclasses.asdict() —
    both fail on a plain dict.  This wrapper adds the missing to_dict().
    """

    def to_dict(self):
        return dict(self)


def get_train_cfg(exp_name, max_iterations, adaptive_lr=False, desired_kl=0.02, learning_rate=0.001):
    schedule = "adaptive" if adaptive_lr else "fixed"
    kl_target = desired_kl if adaptive_lr else None
    return {
        # Runner-level
        "num_steps_per_env": 20,
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
            "desired_kl": kl_target,
            "entropy_coef": 0.001,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": learning_rate,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": schedule,
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "share_cnn_encoders": True,
            "rnd_cfg": None,
        },

        # Actor (CNN + MLP)
        "actor": {
            "class_name": "CNNModel",
            "hidden_dims": [64, 64, 64],
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
                    "norm": "batch",
                    "activation": "elu",
                    "global_pool": "max",
                    "flatten": True,
                },
            },
        },

        # Critic (CNN shared from actor + separate MLP)
        "critic": {
            "class_name": "CNNModel",
            "hidden_dims": [64, 64, 64],
            "activation": "elu",
            "obs_normalization": True,
            # No cnn_cfg — PPO injects actor.cnns via share_cnn_encoders
        },

        # Observation routing
        "obs_groups": {
            "actor":  ["state", "depth"],
            "critic": ["state", "depth"],
        },
    }


def get_cfgs(env_v2=False):
    env_cfg = DictConfig({
        "num_actions": 4,
        "episode_length_s": 60.0,
        "decimation": 300,             # PID runs at 100 Hz, RL decides every 300 steps (3 s)
        "action_scales": [1.0, 1.0, 1.0],
        # Drone spawn
        "spawn_offset": 5.0,
        "spawn_height_min": 10.0,
        "spawn_height_max": 10.0,
        # Target (randomized across 10×10m area)
        "target_x_range": [-5.0, 5.0],
        "target_y_range": [-5.0, 5.0],
        "target_z_range": [1.0, 1.0],
        # Obstacle curriculum disabled — obstacles present from step 0
        "curriculum_steps": 0,
        "curriculum_n_obstacles": 5,
        # Success: within radius of target for the entire decision step
        "hover_radius": 0.3,
        # Obstacles
        "num_obstacles": 8,
        "obstacle_size": [1.0, 1.0, 2.0],
        "collision_radius": 0.3,
        "safety_radius": 3.0,
        # Post-curriculum placement strategy: "strategic" | "vineyard"
        "placement_strategy": "strategic",
        # Post-curriculum strategic placement
        "n_corridor_obstacles": 3,
        "n_ring_obstacles": 4,
        "ring_radius_range": [1.5, 3.5],
        "corridor_lateral_offset": 2.0,
        "post_curriculum_range": 5.0,
        # Post-curriculum vineyard placement (3 m-tall rows around the target)
        "vineyard_n_rows": 2,
        "vineyard_row_spacing": 4.0,
        "vineyard_within_row_spacing": 2.0,
        "vineyard_jitter": 0.15,
        "vineyard_height": 3.0,
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

    if env_v2:
        reward_cfg = {
            "reward_scales": {
                "progress":           0.5,
                "close":              0.1,
                "obstacle_proximity": -0.6,
                "crash":             -10.0,
                "success":            20.0,
            },
        }
    else:
        reward_cfg = {
            "reward_scales": {
                "distance":           -0.5,
                "time":               -0.05,
                "obstacle_proximity": -0.6,
                "crash":             -10.0,
                "success":            20.0,
            },
        }

    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="obstacle-avoidance")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=8001)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. logs/obstacle-avoidance/model_300.pt)")
    parser.add_argument("--env-v2", action="store_true", help="Use V2 env (progress+close rewards, no dt-scaling)")
    parser.add_argument("--placement", choices=["strategic", "vineyard"], default="strategic",
                        help="Post-curriculum obstacle placement strategy")
    parser.add_argument("--adaptive-lr", action="store_true",
                        help="Enable rsl-rl adaptive KL-based LR schedule (default: fixed LR)")
    parser.add_argument("--desired-kl", type=float, default=0.01,
                        help="Target KL divergence for adaptive schedule (default: 0.01)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="PPO learning rate (default: 0.001)")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs(env_v2=args.env_v2)
    train_cfg = get_train_cfg(
        args.exp_name, args.max_iterations,
        adaptive_lr=args.adaptive_lr, desired_kl=args.desired_kl,
        learning_rate=args.learning_rate,
    )
    if args.env_v2:
        train_cfg["wandb_project"] = "obstacle-avoidance-v2"

    if args.resume is None:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    env_cfg["placement_strategy"] = args.placement

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    EnvClass = ObstacleAvoidanceEnvV2 if args.env_v2 else ObstacleAvoidanceEnv
    env = EnvClass(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )
    env.build()

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    if args.resume is not None:
        runner.load(args.resume)
        # Sync env curriculum clock with runner progress. Without this, global_step
        # resets to 0 on every resume and the env never exits curriculum when HPC
        # sessions are shorter than curriculum_steps / num_steps_per_env iterations.
        env.global_step = runner.current_learning_iteration * train_cfg["num_steps_per_env"]
        print(f"Resumed from {args.resume} at iteration {runner.current_learning_iteration} "
              f"(env.global_step={env.global_step}, curriculum_steps={env_cfg.get('curriculum_steps', 0)})")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# Training with W&B logging (headless, 256 envs)
python train_rl_wb.py -B 256 --max_iterations 8001

# V2 env (progress+close rewards, no dt-scaling)
python train_rl_wb.py -B 256 --max_iterations 8001 --env-v2

# Smoke test with viewer (4 envs, 5 iterations)
python train_rl_wb.py -B 4 -v --max_iterations 5

# First-time W&B setup:
#   pip install wandb
#   wandb login
"""
