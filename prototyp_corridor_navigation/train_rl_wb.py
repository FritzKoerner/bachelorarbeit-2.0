import argparse
import copy
import os
import pickle
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from envs.corridor_navigation_env import CorridorNavigationEnv


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
        "wandb_project": "corridor-navigation",
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


def get_cfgs():
    env_cfg = DictConfig({
        "num_actions": 4,
        "episode_length_s": 60.0,
        "decimation": 300,             # PID runs at 100 Hz, RL decides every 300 steps (3 s)
        "action_scales": [1.0, 1.0, 1.0],
        # Length of the obstacles-hidden warm-up phase, in env steps.
        # 6000 = 300 iterations * num_steps_per_env(20). This is the authoritative
        # default — the --curriculum-iterations CLI flag only overrides it when
        # explicitly passed.
        "curriculum_steps": 6000,
        # Success: within radius of target for the entire decision step
        "hover_radius": 0.3,
        # Obstacles — 8 total, mix of 2 boxes, 2 spheres, 2 cylinders, 2 pillars
        "collision_radius": 0.3,
        "safety_radius": 3.0,
        "corridor_box_sizes":      [(2.5, 2.5, 3.0), (2.8, 2.2, 2.5)],
        "corridor_sphere_radii":   [1.2, 1.4],
        "corridor_cylinder_specs": [(1.0, 3.0), (1.1, 2.5)],
        "corridor_pillar_specs":   [(0.5, 4.5), (0.6, 4.0)],
        "corridor_pair_gap_min": 1.0,
        "corridor_pair_gap_max": 2.0,
        "corridor_first_obstacle_offset_min": 0.0,
        "corridor_first_obstacle_offset_max": 3.0,
        # Corridor geometry (all metres). Corridor is fixed-axis along +X.
        "corridor_x_range": [0.0, 32.0],
        "corridor_y_range": [-4.0, 4.0],
        "corridor_z_range": [0.3, 6.0],
        "corridor_spawn_x_range": [0.5, 2.5],
        "corridor_spawn_y_range": [-2.5, 2.5],
        "corridor_spawn_z": 5.0,
        "corridor_target_pos": [30.0, 0.0, 1.0],
        "corridor_slice_centres": [10.0, 14.0, 18.0, 22.0],
        "corridor_min_gap": 2.0,
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
            "rel_pos": 1 / 30.0,   # corridor is ~30 m end-to-end now (was 1/15.0)
            "lin_vel": 0.4,
            "ang_vel": 1 / 3.14159,
        },
    }

    reward_cfg = {
        "reward_scales": {
            "progress":           0.5,
            "close":              0.1,
            "obstacle_proximity": -0.6,
            "crash":             -10.0,
            "success":            20.0,
            "out_of_corridor":   -10.0,
        },
    }

    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="corridor-navigation")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=8001)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. logs/corridor-navigation/model_300.pt)")
    parser.add_argument("--curriculum-iterations", type=int, default=None,
                        help="Override curriculum length (iterations). When omitted, "
                             "env_cfg['curriculum_steps'] from get_cfgs() is used as-is. "
                             "Converted to env steps via num_steps_per_env. Set 0 to "
                             "disable curriculum. Corridor bounds are enforced from step 0 "
                             "regardless.")
    parser.add_argument("--adaptive-lr", action="store_true",
                        help="Enable rsl-rl adaptive KL-based LR schedule (default: fixed LR)")
    parser.add_argument("--desired-kl", type=float, default=0.01,
                        help="Target KL divergence for adaptive schedule (default: 0.01)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="PPO learning rate (default: 0.001)")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(
        args.exp_name, args.max_iterations,
        adaptive_lr=args.adaptive_lr, desired_kl=args.desired_kl,
        learning_rate=args.learning_rate,
    )

    if args.resume is None:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    if args.curriculum_iterations is not None:
        env_cfg["curriculum_steps"] = (
            args.curriculum_iterations * train_cfg["num_steps_per_env"]
        )

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = CorridorNavigationEnv(
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

# Smoke test with viewer (4 envs, 5 iterations)
python train_rl_wb.py -B 4 -v --max_iterations 5

# Smoke test with curriculum disabled (obstacles visible from step 0)
python train_rl_wb.py -B 4 -v --max_iterations 5 --curriculum-iterations 0
"""
