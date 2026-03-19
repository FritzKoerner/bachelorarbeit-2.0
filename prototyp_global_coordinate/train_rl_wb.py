import argparse
import copy
import os
import pickle
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from envs.coordinate_landing_env import CoordinateLandingEnv


class DictConfig(dict):
    """Thin dict wrapper so rsl-rl's WandbSummaryWriter.store_config() works.

    store_config tries env_cfg.to_dict() first, then dataclasses.asdict() —
    both fail on a plain dict.  This wrapper adds the missing to_dict().
    """

    def to_dict(self):
        return dict(self)


def get_train_cfg(exp_name, max_iterations):
    return {
        # Runner-level
        "num_steps_per_env": 100,
        "save_interval": 100,

        # ── W&B logging ──────────────────────────────────────────────
        "logger": "wandb",
        "wandb_project": "drone-landing",
        # Optional: set WANDB_USERNAME env var for team/entity routing
        # ─────────────────────────────────────────────────────────────

        # Algorithm
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.001,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            "rnd_cfg": None,
        },

        # Actor (stochastic policy)
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        },

        # Critic (value function)
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
        },

        # Observation routing
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
    }


def get_cfgs():
    env_cfg = DictConfig({
        "num_actions": 4,
        "episode_length_s": 30.0,
        # Action scaling: maps [-1,1] → offset from current drone position (metres).
        "action_scales": [3.0, 3.0, 3.0],
        # Drone spawn randomisation
        "spawn_offset": 5.0,        # drone x/y in [-5, +5] m
        "spawn_height_min": 10.0,
        "spawn_height_max": 10.0,
        # Target randomisation (10×10 m square at 1 m height)
        "target_x_range": [-5.0, 5.0],
        "target_y_range": [-5.0, 5.0],
        "target_z_range": [1.0, 1.0],
        # Curriculum: first curriculum_steps env steps → target within
        # curriculum_radius of drone spawn.  curriculum_steps = N_iters × num_steps_per_env
        "curriculum_steps": 0,        # disabled — full target range from start
        "curriculum_radius": 1.0,    # (unused when curriculum_steps=0)
        # Success: hover within radius at low velocity for N consecutive steps
        "hover_radius": 0.3,             # metres
        "success_vel_threshold": 0.3,    # m/s  (body frame)
        "hover_steps": 30,               # 0.3 s at 100 Hz
        # Visualisation
        "visualize_target": False,
        "env_spacing": 40.0,
        # PID controller parameters  [kp, ki, kd]
        "pid_params": {
            # Motor limits — URDF thrust2weight=2.25 → max_rpm = 1789.2 * sqrt(2.25) ≈ 2684
            "base_rpm": 1789.2,
            "max_rpm": 2700.0,
            # Cascade output limits
            "max_tilt": 30.0,       # degrees — max roll/pitch the velocity loop can command
            "max_vel_xy": 5.0,      # m/s — max horizontal velocity the position loop can command
            "max_vel_z": 3.0,       # m/s — max vertical velocity the position loop can command
            # Position loop → desired velocity
            "pid_params_pos_x": [1.0, 0.0, 0.7],
            "pid_params_pos_y": [1.0, 0.0, 0.7],
            "pid_params_pos_z": [1.5, 0.0, 1.0],
            # Velocity loop → desired attitude / thrust
            "pid_params_vel_x": [16.0, 0.0, 8.0],
            "pid_params_vel_y": [16.0, 0.0, 8.0],
            "pid_params_vel_z": [100.0, 2.0, 10.0],
            # Attitude loop → mixer corrections
            "pid_params_roll":  [6.0, 0.0, 3.0],
            "pid_params_pitch": [6.0, 0.0, 3.0],
            "pid_params_yaw":   [1.0, 0.0, 0.2],
        },
    })

    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos":    1 / 30.0,
            "lin_vel":    1 / 5.0,
            "ang_vel":    1 / 3.14159,
        },
    }

    reward_cfg = {
        "reward_scales": {
            "distance":  -5.0,
            "time":     -0.5,
            "crash":   -100.0,
            "success":  200.0,
        },
    }

    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-landing")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=401)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = CoordinateLandingEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )
    env.build()

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# Training with W&B logging (headless, 4096 envs)
python train_rl_wb.py -B 4096 --max_iterations 301

# Smoke test with viewer (4 envs, 5 iterations)
python train_rl_wb.py -B 4 -v --max_iterations 5

# First-time W&B setup:
#   pip install wandb
#   wandb login
"""
