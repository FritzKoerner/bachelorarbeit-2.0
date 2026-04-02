"""
CoordinateLandingDiscreteEnv: Discrete action variant of CoordinateLandingEnv.

Inherits all physics, rewards, observations, and reset logic from the continuous
environment.  The only change: the RL agent outputs 4 ternary discrete actions
instead of 4 continuous floats.

Action space: 4-element integer vector, each in {0, 1, 2}
    [0] lateral X:  0 = stay,  1 = backward (-x),  2 = forward  (+x)
    [1] lateral Y:  0 = stay,  1 = right    (-y),  2 = left      (+y)
    [2] vertical:   0 = stay,  1 = down     (-z),  2 = up        (+z)
    [3] yaw:        0 = stay,  1 = right   (-90),  2 = left     (+90)

Mapping: discrete {0,1,2} -> continuous {0, -1, +1}, then passed to parent step().
Yaw is scaled by 0.5 so parent's `action*180` gives {0, -90, +90} degrees.
"""

# Lookup table: discrete action index -> continuous value
#   0 = stay (0.0),  1 = negative (-1.0),  2 = positive (+1.0)
import torch

from envs.coordinate_landing_env import CoordinateLandingEnv

# Lookup table: discrete action index -> continuous value
#   0 = stay (0.0),  1 = negative (-1.0),  2 = positive (+1.0)
_ACTION_MAP = [0.0, -1.0, 1.0]


class CoordinateLandingDiscreteEnv(CoordinateLandingEnv):

    def step(self, actions):
        # actions: (n_envs, 4) integer tensor in {0, 1, 2}
        idx = actions.long()
        lut = torch.tensor(_ACTION_MAP, device=actions.device, dtype=torch.float32)
        continuous = lut[idx]  # (n_envs, 4) -> {0, -1, +1}
        # Scale yaw: +/-1 * 180 = +/-180, but we want +/-90
        continuous[:, 3] *= 0.5
        return super().step(continuous)
