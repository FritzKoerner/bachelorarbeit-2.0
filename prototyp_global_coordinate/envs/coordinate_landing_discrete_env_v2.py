"""4x3 ternary discrete variant of CoordinateLandingEnvV2.

Identical action mapping as coordinate_landing_discrete_env.py
but inheriting V2's reward structure (progress + close, no dt-scaling).
"""

import torch

from envs.coordinate_landing_env_v2 import CoordinateLandingEnvV2
from envs.coordinate_landing_discrete_env import _ACTION_MAP


class CoordinateLandingDiscreteEnvV2(CoordinateLandingEnvV2):

    def step(self, actions):
        idx = actions.long()
        lut = torch.tensor(_ACTION_MAP, device=actions.device, dtype=torch.float32)
        continuous = lut[idx]  # (n_envs, 4) -> {0, -1, +1}
        continuous[:, 3] *= 0.5
        return super().step(continuous)
