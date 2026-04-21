"""
CoordinateLandingEnvV2: reward-reworked variant of CoordinateLandingEnv.

Changes from v1:
  - No dt-scaling of rewards.  Weights are fixed regardless of decimation.
  - Delta-distance "progress" reward replaces absolute distance penalty.
  - Exponential "close" reward for strong final-approach attraction.
  - No time penalty (progress + close already discourage idling).

Reward components:
  progress  — positive when drone moves toward target (bounded by drone speed)
  close     — exp(-dist): peaks at target, strong gradient within ~2 m
  crash     — one-time penalty on crash
  success   — one-time bonus on successful hover

Everything else (obs, actions, PID, reset, termination) is identical to v1.
"""

import math
import copy

import torch
from tensordict import TensorDict
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

from controllers.pid_controller import CascadingPIDController


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class CoordinateLandingEnvV2:
    def __init__(self, num_envs: int, env_cfg: dict, obs_cfg: dict,
                 reward_cfg: dict, show_viewer: bool = False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]           # 17
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]   # 4
        self.device = gs.device

        self.dt = 0.01  # 100 Hz physics
        self.decimation = env_cfg.get("decimation", 1)  # RL decision every N physics steps
        self.decision_dt = self.dt * self.decimation      # effective RL timestep
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.decision_dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.cfg = env_cfg  # Logger expects env.cfg
        self.show_viewer = show_viewer

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        self._built = False

    # ------------------------------------------------------------------
    # Two-phase construction: call build() after gs.init()
    # ------------------------------------------------------------------

    def build(self):
        """Create Genesis scene, drone, PID controller, and all tensor buffers."""
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(7.0, 0.0, 5.0),
                camera_lookat=(3.0, 3.0, 3.0),
                camera_fov=60,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(10, self.num_envs)))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=self.show_viewer,
        )

        scene.add_entity(gs.morphs.Plane())

        if self.env_cfg.get("visualize_target", False):
            self.target_vis = scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.15,
                    fixed=True,
                    collision=False,
                    batch_fixed_verts=True,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5))
                ),
            )
        else:
            self.target_vis = None

        self.drone = scene.add_entity(
            gs.morphs.Drone(
                file="../assets/robots/draugas/draugas_genesis.urdf",
                pos=(0, 0, 3.0),
                euler=(0, 0, 0),
                propellers_link_name=["prop0_link", "prop1_link", "prop2_link", "prop3_link"],
                propellers_spin=[1, -1, 1, -1],
            )
        )

        env_spacing = self.env_cfg.get("env_spacing", 40.0)
        scene.build(n_envs=self.num_envs, env_spacing=(env_spacing, env_spacing))
        self.scene = scene

        # Initial orientation (identity quat [w,x,y,z])
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # Cascading PID controller (vectorized)
        self.controller = CascadingPIDController(
            drone=self.drone,
            dt=self.dt,
            base_rpm=self.env_cfg["pid_params"]["base_rpm"],
            max_rpm=self.env_cfg["pid_params"]["max_rpm"],
            pid_params=self.env_cfg["pid_params"],
            n_envs=self.num_envs,
            device=gs.device,
        )

        # No dt-scaling: reward weights are used as-is from reward_cfg.
        # This keeps reward magnitudes stable across different decimation values.

        self.reward_functions = {
            name: getattr(self, "_reward_" + name)
            for name in self.reward_scales
        }
        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
            for name in self.reward_scales
        }

        # Observation / reward / reset buffers
        self.obs_buf           = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.terminal_obs_buf  = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf           = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf         = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        # Action buffers
        self.actions      = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # State buffers
        self.base_pos       = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.last_base_pos  = torch.zeros_like(self.base_pos)
        self.base_quat      = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_lin_vel   = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel   = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        # Target and relative position
        self.target_pos  = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.rel_pos     = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)

        # Unstable buffer: once a drone becomes unstable in an episode,
        # it stays frozen at frozen_pos (no gravity, no movement) until reset.
        self.unstable_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        self.frozen_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)

        self.global_step = 0  # counts calls to step(); used for curriculum

        self.extras = {}

        # Debug / monitoring accumulators (logged via extras["episode"])
        self.action_saturated_count = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.action_sum    = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.obs_clipped_count = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self._built = True

        # Reset all envs so the runner sees valid state from get_observations().
        # The v5.x runner never calls reset() explicitly.
        self.reset()

    # ------------------------------------------------------------------
    # rsl-rl interface
    # ------------------------------------------------------------------

    def step(self, actions):
        self.actions = torch.clip(actions, -1.0, 1.0)

        # Accumulate debug metrics
        self.action_sum += self.actions
        self.action_saturated_count += (torch.abs(self.actions) > 0.99).any(dim=1).float()

        # Map RL actions → small offsets from the current drone position.
        # The PID target is always nearby, preventing aggressive chasing.
        scales = self.env_cfg["action_scales"]
        target_x   = self.base_pos[:, 0] + self.actions[:, 0] * scales[0]
        target_y   = self.base_pos[:, 1] + self.actions[:, 1] * scales[1]
        target_z   = self.base_pos[:, 2] + self.actions[:, 2] * scales[2]
        target_yaw = self.actions[:, 3] * 180.0  # absolute yaw (degrees)
        target_pos = torch.stack([target_x, target_y, target_z], dim=-1)

        if self.target_vis is not None:
            self.target_vis.set_pos(self.target_pos, zero_velocity=True)

        # Run physics N times per RL decision (decimation).
        # PID tracks the same target each substep; state is read after the last one.
        # With high decimation, early random actions can cause extreme states mid-loop.
        # Kill motors for unstable envs to prevent NaN accelerations in the solver;
        # the post-loop crash check will reset them normally.
        inside_target_all_substeps = torch.ones(self.num_envs, device=gs.device, dtype=torch.bool)
        for _ in range(self.decimation):
            rpms = self.controller.update(target_pos, target_yaw)
            curr_pos = self.drone.get_pos()
            curr_att = quat_to_xyz(self.drone.get_quat(), rpy=True, degrees=True)
            newly_unstable = (
                (curr_pos[:, 2] < 0.2)
                | (torch.abs(curr_att[:, 0]) > 60.0)
                | (torch.abs(curr_att[:, 1]) > 60.0)
                | (torch.norm(self.drone.get_vel(), dim=1) > 20.0)
            ) & ~self.unstable_buf
            # Latch: record frozen position for newly unstable drones
            if newly_unstable.any():
                self.frozen_pos[newly_unstable] = curr_pos[newly_unstable].clone()
                self.unstable_buf |= newly_unstable
            # Freeze all unstable drones (newly + previously) at their frozen position
            if self.unstable_buf.any():
                unstable_idx = self.unstable_buf.nonzero(as_tuple=False).reshape(-1)
                rpms[self.unstable_buf] = 0.0
                self.drone.set_pos(
                    self.frozen_pos[unstable_idx],
                    zero_velocity=True,
                    envs_idx=unstable_idx,
                )
            self.drone.set_propellels_rpm(rpms)
            self.scene.step()

            # Track whether drone stays inside target area throughout decision step
            substep_dist = torch.norm(self.target_pos - self.drone.get_pos(), dim=1)
            inside_target_all_substeps &= (substep_dist < self.env_cfg["hover_radius"])

        # Update state buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:]      = self.drone.get_pos()
        self.rel_pos          = self.target_pos - self.base_pos
        self.last_rel_pos     = self.target_pos - self.last_base_pos
        self.base_quat[:]     = self.drone.get_quat()

        base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True, degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # Termination
        self.crash_condition = (
            (self.base_pos[:, 2] < 0.2)
            | (torch.abs(base_euler[:, 0]) > 60.0)
            | (torch.abs(base_euler[:, 1]) > 60.0)
            | (torch.norm(self.rel_pos, dim=1) > 50.0)
        )
        # Success: drone stayed within hover_radius for the entire decision step
        self.success_condition = inside_target_all_substeps
        timeout_condition = self.episode_length_buf > self.max_episode_length

        self.penalty_condition = self.crash_condition | timeout_condition

        self.reset_buf = timeout_condition | self.crash_condition | self.success_condition

        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, dtype=gs.tc_float)
        self.extras["time_outs"][timeout_condition] = 1.0

        self.global_step += 1

        # Compute rewards BEFORE reset so they reflect the terminal state,
        # not the new episode's initial state.
        self.rew_buf[:] = 0.0
        self.reward_components = {}
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.reward_components[name] = rew
            self.episode_sums[name] += rew

        # Snapshot terminal observations for SB3 wrapper (before reset overwrites state).
        self.terminal_obs_buf[:] = self._compute_obs()

        # Auto-reset done envs AFTER rewards but BEFORE observations,
        # so obs reflects the new episode for rsl-rl.
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # Compute observations (post-reset for done envs)
        self.obs_buf = self._compute_obs()
        self.last_actions[:] = self.actions[:]

        # Detect obs clipping (rel_pos dims 0:3, lin_vel dims 7:10, ang_vel dims 10:13)
        clipped = (self.obs_buf[:, :3].abs().gt(0.99).any(dim=1)
                   | self.obs_buf[:, 7:13].abs().gt(0.99).any(dim=1))
        self.obs_clipped_count += clipped.float()

        obs = self.get_observations()
        return obs, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        n = len(envs_idx)

        # Randomize drone spawn
        offset = self.env_cfg["spawn_offset"]
        sx = gs_rand_float(-offset, offset, (n,), gs.device)
        sy = gs_rand_float(-offset, offset, (n,), gs.device)
        sz = gs_rand_float(
            self.env_cfg["spawn_height_min"],
            self.env_cfg["spawn_height_max"],
            (n,), gs.device,
        )
        spawn_pos = torch.stack([sx, sy, sz], dim=-1)

        # Randomize target position (curriculum: close to drone early in training)
        curriculum_steps = self.env_cfg.get("curriculum_steps", 0)
        if self.global_step < curriculum_steps:
            r = self.env_cfg.get("curriculum_radius", 1.0)
            tx = sx + gs_rand_float(-r, r, (n,), gs.device)
            ty = sy + gs_rand_float(-r, r, (n,), gs.device)
            tz = sz + gs_rand_float(-r, r, (n,), gs.device)
            tz = torch.clamp(tz, min=0.2)  # keep above ground
        else:
            tx = gs_rand_float(*self.env_cfg["target_x_range"], (n,), gs.device)
            ty = gs_rand_float(*self.env_cfg["target_y_range"], (n,), gs.device)
            tz = gs_rand_float(*self.env_cfg["target_z_range"], (n,), gs.device)
        self.target_pos[envs_idx] = torch.stack([tx, ty, tz], dim=-1)

        self.base_pos[envs_idx]      = spawn_pos
        self.last_base_pos[envs_idx] = spawn_pos
        self.base_quat[envs_idx]     = self.base_init_quat.reshape(1, -1)

        self.drone.set_pos(spawn_pos, zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # Reset relative positions
        self.rel_pos      = self.target_pos - self.base_pos
        self.last_rel_pos = self.target_pos - self.last_base_pos

        # Reset PID state for these envs
        self.controller.reset_idx(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx]      = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True
        self.unstable_buf[envs_idx]       = False

        # Log episode stats
        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # Debug / monitoring metrics
        # ep_len = self.episode_length_buf[envs_idx].float().clamp(min=1)
        # mean_ep_len = ep_len.mean()
        # self.extras["episode"]["action_saturation_rate"] = (
        #     self.action_saturated_count[envs_idx].mean() / mean_ep_len
        # ).item()
        # self.extras["episode"]["obs_clip_rate"] = (
        #     self.obs_clipped_count[envs_idx].mean() / mean_ep_len
        # ).item()
        # action_names = ["ax", "ay", "az", "ayaw"]
        # action_mean = self.action_sum[envs_idx] / mean_ep_len
        # for i, name in enumerate(action_names):
        #     self.extras["episode"][f"action_mean_{name}"] = action_mean[:, i].mean().item()

        # Reset accumulators
        # self.action_saturated_count[envs_idx] = 0.0
        # self.action_sum[envs_idx] = 0.0
        # self.obs_clipped_count[envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.get_observations()

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _compute_obs(self):
        s = self.obs_scales
        return torch.cat(
            [
                torch.clip(self.rel_pos     * s["rel_pos"],    -1, 1),   # 3
                self.base_quat,                                            # 4
                torch.clip(self.base_lin_vel * s["lin_vel"],   -1, 1),   # 3
                torch.clip(self.base_ang_vel * s["ang_vel"],   -1, 1),   # 3
                self.last_actions,                                         # 4
            ],
            dim=-1,
        )  # total: 17

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _reward_progress(self):
        """Reward for moving closer to target. Positive when approaching."""
        prev_dist = torch.norm(self.last_rel_pos, dim=1)
        curr_dist = torch.norm(self.rel_pos, dim=1)
        return prev_dist - curr_dist

    def _reward_close(self):
        """Exponential reward that peaks at the target.
        Gives strong gradient within ~2 m for precise final approach."""
        return torch.exp(-torch.norm(self.rel_pos, dim=1))

    def _reward_crash(self):
        """Penalty on crash."""
        rew = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        rew[self.crash_condition] = 1.0
        return rew

    def _reward_success(self):
        """Reward on success."""
        rew = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        rew[self.success_condition] = 1.0
        return rew
