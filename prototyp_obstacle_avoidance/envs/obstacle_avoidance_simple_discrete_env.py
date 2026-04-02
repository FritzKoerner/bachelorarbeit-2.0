"""
ObstacleAvoidanceSimpleDiscreteEnv: Single-categorical discrete action variant.

The agent picks exactly ONE of 9 actions per step (no simultaneous movement):

    0 = halt          → PID target = current position (hover in place)
    1 = forward (+x)
    2 = backward (-x)
    3 = left (+y)
    4 = right (-y)
    5 = up (+z)
    6 = down (-z)
    7 = turn-left (+90 yaw)
    8 = turn-right (-90 yaw)

Each action maps to a 4D continuous vector passed to the parent env's PID controller.
Yaw entries are ±0.5 so parent's `action * 180` gives ±90 degrees.
"""

import torch

from envs.obstacle_avoidance_env import ObstacleAvoidanceEnv

#                         ax     ay     az    ayaw
_ACTION_TABLE = [
    [ 0.0,   0.0,   0.0,  0.0 ],   # 0: halt
    [ 1.0,   0.0,   0.0,  0.0 ],   # 1: forward  (+x)
    [-1.0,   0.0,   0.0,  0.0 ],   # 2: backward (-x)
    [ 0.0,   1.0,   0.0,  0.0 ],   # 3: left     (+y)
    [ 0.0,  -1.0,   0.0,  0.0 ],   # 4: right    (-y)
    [ 0.0,   0.0,   1.0,  0.0 ],   # 5: up       (+z)
    [ 0.0,   0.0,  -1.0,  0.0 ],   # 6: down     (-z)
    [ 0.0,   0.0,   0.0,  0.5 ],   # 7: turn-left  (+90°)
    [ 0.0,   0.0,   0.0, -0.5 ],   # 8: turn-right (-90°)
]


class ObstacleAvoidanceSimpleDiscreteEnv(ObstacleAvoidanceEnv):

    def build(self):
        super().build()
        self.num_actions = 1
        self._action_lut = torch.tensor(
            _ACTION_TABLE, device=self.device, dtype=torch.float32,
        )

    def step(self, actions):
        # actions: (n_envs, 1) integer tensor in {0..8}
        idx = actions[:, 0].long()
        continuous = self._action_lut[idx]  # (n_envs, 4)
        return super().step(continuous)
