"""SB3 VecEnv wrapper for batched Genesis environments.

Bridges the rsl-rl-style Genesis env interface to Stable-Baselines3's VecEnv API.
Handles torch↔numpy conversion and terminal observation capture.
"""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


class GenesisSB3VecEnv(VecEnv):
    """Wraps a batched Genesis env (rsl-rl interface) as an SB3 VecEnv."""

    def __init__(self, genesis_env, action_space: spaces.Space):
        obs_dim = genesis_env.num_obs
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        super().__init__(
            num_envs=genesis_env.num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.genesis_env = genesis_env
        self._actions = None

    def reset(self):
        obs_td = self.genesis_env.reset()
        return obs_td["policy"].cpu().numpy()

    def step_async(self, actions):
        # actions: np.ndarray — (n_envs,) for Discrete, (n_envs, D) for Box/MultiDiscrete
        self._actions = torch.as_tensor(
            actions, device=self.genesis_env.device, dtype=torch.float32,
        )
        # Discrete → (n_envs, 1) to match env expectation
        if self._actions.dim() == 1:
            self._actions = self._actions.unsqueeze(-1)

    def step_wait(self):
        obs_td, rewards, dones, extras = self.genesis_env.step(self._actions)

        obs_np = obs_td["policy"].cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy().astype(bool)

        # Build info dicts — only populate for done envs to avoid unnecessary CPU transfers
        infos = [{} for _ in range(self.num_envs)]
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_indices.numel() > 0:
            terminal_obs_np = self.genesis_env.terminal_obs_buf[done_indices].cpu().numpy()
            time_outs = extras.get("time_outs")
            for j, i in enumerate(done_indices.cpu().tolist()):
                infos[i]["terminal_observation"] = terminal_obs_np[j]
                if time_outs is not None and time_outs[i]:
                    infos[i]["TimeLimit.truncated"] = True

        return obs_np, rewards_np, dones_np, infos

    def close(self):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return getattr(self.genesis_env, method_name)(*method_args, **method_kwargs)

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.genesis_env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.genesis_env, attr_name, value)

    def seed(self, seed=None):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
