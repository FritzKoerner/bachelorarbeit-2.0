import argparse
import copy
import os
import pickle
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from envs.coordinate_landing_discrete_env import CoordinateLandingDiscreteEnv
from envs.coordinate_landing_discrete_env_v2 import CoordinateLandingDiscreteEnvV2
from modules.multi_categorical import MultiCategoricalDistribution


class DictConfig(dict):
    """Thin dict wrapper so rsl-rl's WandbSummaryWriter.store_config() works."""

    def to_dict(self):
        return dict(self)


def get_train_cfg(exp_name, max_iterations):
    return {
        # Runner-level
        "num_steps_per_env": 30,
        "save_interval": 100,

        # W&B logging
        "logger": "wandb",
        "wandb_project": "drone-discrete",

        # Algorithm
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,       # higher than continuous — encourage exploration of discrete choices
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

        # Actor (stochastic policy — MultiCategorical instead of Gaussian)
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "distribution_cfg": {
                "class_name": MultiCategoricalDistribution,
                "num_choices": 3,
            },
        },

        # Critic (value function — no distribution)
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


def get_cfgs(env_v2=False):
    env_cfg = DictConfig({
        "num_actions": 4,
        "decimation": 100,            # PID at 100 Hz, RL at 1 Hz
        "episode_length_s": 60.0,
        # Action scaling: discrete {-1,+1} * scale = step size in metres
        "action_scales": [1.0, 1.0, 1.0],
        # Drone spawn
        "spawn_offset": 5.0,
        "spawn_height_min": 10.0,
        "spawn_height_max": 10.0,
        # Target
        "target_x_range": [-5.0, 5.0],
        "target_y_range": [-5.0, 5.0],
        "target_z_range": [1.0, 1.0],
        # Curriculum
        "curriculum_steps": 0,
        "curriculum_radius": 1.0,
        # Success
        "hover_radius": 0.3,
        # Visualisation
        "visualize_target": False,
        "env_spacing": 40.0,
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
        "num_obs": 17,
        "obs_scales": {
            "rel_pos":    1 / 15.0,
            "lin_vel":    0.4,
            "ang_vel":    1 / 3.14159,
        },
    }

    if env_v2:
        reward_cfg = {
            "reward_scales": {
                "progress":  5.0,
                "close":     1.0,
                "crash":   -100.0,
                "success":  200.0,
            },
        }
    else:
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
    parser.add_argument("-e", "--exp_name", type=str, default="drone-landing-discrete")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=401)
    parser.add_argument("--env-v2", action="store_true", help="Use V2 env (progress+close rewards)")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs(env_v2=args.env_v2)
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    if args.env_v2:
        train_cfg["wandb_project"] = "drone-discrete-v2"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    EnvClass = CoordinateLandingDiscreteEnvV2 if args.env_v2 else CoordinateLandingDiscreteEnv
    env = EnvClass(
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
python train_rl_discrete_wb.py -B 4096 --max_iterations 401

# Smoke test with viewer
python train_rl_discrete_wb.py -B 4 -v --max_iterations 5
"""
