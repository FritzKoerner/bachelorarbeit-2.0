"""Simple 9-action discrete variant of CoordinateLandingEnvV2.

Identical action mapping as coordinate_landing_simple_discrete_env.py
but inheriting V2's reward structure (progress + close, no dt-scaling).
"""

import torch

from envs.coordinate_landing_env_v2 import CoordinateLandingEnvV2
from envs.coordinate_landing_simple_discrete_env import _ACTION_TABLE


class CoordinateLandingSimpleDiscreteEnvV2(CoordinateLandingEnvV2):

    def build(self):
        super().build()
        self.num_actions = 1
        self._action_lut = torch.tensor(
            _ACTION_TABLE, device=self.device, dtype=torch.float32,
        )

    def step(self, actions):
        idx = actions[:, 0].long()
        continuous = self._action_lut[idx]  # (n_envs, 4)
        return super().step(continuous)
