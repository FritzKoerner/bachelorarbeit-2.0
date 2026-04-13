"""Train discrete-action drone landing with Stable-Baselines3 PPO.

Uses the 9-action simple discrete env (halt, ±x, ±y, ±z, ±yaw) wrapped
in an SB3 VecEnv adapter.  SB3 natively handles Discrete action spaces —
no custom distribution module needed.

Usage:
    python train_rl_sb3_discrete.py -B 4096 --max_iterations 401
    python train_rl_sb3_discrete.py -B 4096 --max_iterations 401 --env-v2   # V2 rewards
    python train_rl_sb3_discrete.py -B 4 -v --max_iterations 5              # smoke test
"""

import argparse
import os
import shutil

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.logger import configure
import torch

import genesis as gs

from envs.coordinate_landing_simple_discrete_env import CoordinateLandingSimpleDiscreteEnv
from envs.coordinate_landing_simple_discrete_env_v2 import CoordinateLandingSimpleDiscreteEnvV2
from envs.sb3_vec_env import GenesisSB3VecEnv


# ---------------------------------------------------------------------------
# W&B callback (optional — only used when --wandb is passed)
# ---------------------------------------------------------------------------

class WandbStepCallback(BaseCallback):
    """Logs per-update metrics to W&B using env.extras populated by the Genesis env."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        genesis_env = self.training_env.genesis_env
        extras = getattr(genesis_env, "extras", {})
        ep = extras.get("episode", {})
        if ep:
            for k, v in ep.items():
                wandb.log({f"episode/{k}": v}, step=self.num_timesteps)


# ---------------------------------------------------------------------------
# Environment config (mirrors train_rl_simple_discrete_wb.py)
# ---------------------------------------------------------------------------

def get_env_cfgs(env_v2=False):
    env_cfg = {
        "num_actions": 4,       # internal PID buffers; env.build() overrides to 1
        "decimation": 100,
        "episode_length_s": 30.0,
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
    }

    obs_cfg = {
        "num_obs": 17,
        "obs_scales": {
            "rel_pos":  1 / 15.0,
            "lin_vel":  0.4,
            "ang_vel":  1 / 3.14159,
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
                "distance": -5.0,
                "time":     -0.5,
                "crash":   -100.0,
                "success":  200.0,
            },
        }

    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-landing-sb3-discrete")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=401)
    parser.add_argument("--env-v2", action="store_true", help="Use V2 env (progress+close rewards)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging", default=True)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg = get_env_cfgs(env_v2=args.env_v2)

    import pickle
    pickle.dump(
        {"env_cfg": env_cfg, "obs_cfg": obs_cfg, "reward_cfg": reward_cfg,
         "env_v2": args.env_v2},
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    if args.vis:
        env_cfg["visualize_target"] = True

    # --- Build Genesis env and wrap for SB3 ---
    EnvClass = CoordinateLandingSimpleDiscreteEnvV2 if args.env_v2 else CoordinateLandingSimpleDiscreteEnv
    genesis_env = EnvClass(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=args.vis,
    )
    genesis_env.build()

    env = GenesisSB3VecEnv(
        genesis_env,
        action_space=gymnasium.spaces.Discrete(9),
    )

    # --- PPO hyperparameters (mapped from rsl-rl config) ---
    n_steps = 5                     # num_steps_per_env in rsl-rl
    buffer_size = n_steps * args.num_envs
    num_minibatches = 4             # same as rsl-rl
    batch_size = buffer_size // num_minibatches

    total_timesteps = args.max_iterations * buffer_size

    # --- W&B init (must happen BEFORE model creation so sync_tensorboard
    #     patches TensorBoard before SB3 creates its SummaryWriter) ---
    wandb_run = None
    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wb_project = "drone-sb3-discrete-v2" if args.env_v2 else "drone-sb3-discrete"
        wandb_run = wandb.init(
            project=wb_project,
            name=args.exp_name,
            config={
                "env_cfg": env_cfg,
                "obs_cfg": obs_cfg,
                "reward_cfg": reward_cfg,
                "num_envs": args.num_envs,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "max_iterations": args.max_iterations,
                "env_version": "v2" if args.env_v2 else "v1",
            },
            sync_tensorboard=True,
        )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        # Core PPO
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=5,                 # num_learning_epochs in rsl-rl
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        # Entropy — SB3 Categorical handles this natively
        ent_coef=0.01,
        vf_coef=1.0,
        max_grad_norm=1.0,
        # Adaptive KL (SB3 uses early stopping, not LR adaption like rsl-rl)
        target_kl=0.01,
        # Network architecture (match rsl-rl [128, 128] + tanh)
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=torch.nn.Tanh,
        ),
        device="cuda",
        verbose=1,
        tensorboard_log=f"runs/{wandb_run.id}" if wandb_run else log_dir,
    )

    # --- Callbacks ---
    callbacks = [
        CheckpointCallback(
            save_freq=max(100 * buffer_size // args.num_envs, 1),
            save_path=f"{log_dir}/checkpoints",
            name_prefix="ppo",
        ),
    ]

    if wandb_run:
        callbacks.append(WandbCallback(
            model_save_path=f"{log_dir}/checkpoints",
            verbose=2,
        ))
        callbacks.append(WandbStepCallback())

    # --- Train ---
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(f"{log_dir}/final_model")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
