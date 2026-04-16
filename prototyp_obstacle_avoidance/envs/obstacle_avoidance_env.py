"""
ObstacleAvoidanceEnv: rsl-rl v5.x compatible drone environment with obstacle avoidance.

Extends CoordinateLandingEnv with:
  - Random obstacle boxes between drone spawn area and target
  - Forward-facing depth camera (CNN input)
  - AABB obstacle collision detection
  - Obstacle proximity penalty reward
  - Frame-stacked TensorDict observations: {"state": (n, 17), "depth": (n, stack, 64, 64)}

Constructor signature (rsl-rl style):
    env = ObstacleAvoidanceEnv(num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer)
    gs.init(...)
    env.build()
"""

import math
import copy

import numpy as np
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


def _can_use_batch_renderer():
    """Check if Madrona BatchRenderer is available.

    Requires gs_madrona importable, CUDA backend, and compute capability >= 7.0
    (Volta/Turing+). Older GPUs (e.g. GTX 1050, Pascal sm_61) lack required
    Vulkan features and Madrona will abort() the process — unrecoverable from Python.
    """
    try:
        import gs_madrona  # noqa: F401
    except ImportError:
        return False
    if gs.backend != gs.cuda:
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (
            7,
            5,
        )  # Turing+ required (V100/Volta 7.0 lacks buffer_device_address)
    except Exception:
        return False


class ObstacleAvoidanceEnv:
    def __init__(
        self,
        num_envs: int,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        show_viewer: bool = False,
    ):
        self.num_envs = num_envs
        self.num_state_obs = obs_cfg["num_state_obs"]  # 17
        self.num_obs = self.num_state_obs  # runner reads shapes from TensorDict
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]  # 4
        self.device = gs.device

        self.dt = 0.01  # 100 Hz
        self.decimation = env_cfg.get(
            "decimation", 1
        )  # RL decision every N physics steps
        self.decision_dt = self.dt * self.decimation  # effective RL timestep
        self.max_episode_length = math.ceil(
            env_cfg["episode_length_s"] / self.decision_dt
        )

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.cfg = env_cfg
        self.show_viewer = show_viewer

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

        # Obstacle params
        self.num_obstacles = env_cfg.get("num_obstacles", 5)
        self.obstacle_size = env_cfg.get("obstacle_size", [1.0, 1.0, 2.0])
        self.collision_radius = env_cfg.get("collision_radius", 0.8)
        self.safety_radius = env_cfg.get("safety_radius", 3.0)

        # Depth camera params
        self.depth_res = obs_cfg.get("depth_res", 64)
        self.render_interval = env_cfg.get("render_interval", 2)
        self.max_depth = env_cfg.get("max_depth", 20.0)
        self.depth_stack_size = obs_cfg.get("depth_stack_size", 3)
        self._use_batch_renderer_cfg = env_cfg.get("use_batch_renderer", "auto")
        self.obstacle_half_extents = torch.tensor(
            [s / 2.0 for s in self.obstacle_size], device=gs.device
        )

        self._built = False

    # ------------------------------------------------------------------
    # Two-phase construction
    # ------------------------------------------------------------------

    def build(self, pre_build_cameras=None):
        """Create Genesis scene with obstacles, depth camera, drone, PID, and buffers."""
        # Determine renderer: BatchRenderer (parallel Vulkan) or Rasterizer (per-env loop)
        cfg_val = self._use_batch_renderer_cfg
        if cfg_val == "auto":
            use_batch = _can_use_batch_renderer()
        elif cfg_val is True or cfg_val == "batch":
            use_batch = True
        else:
            use_batch = False
        self._using_batch_renderer = use_batch
        renderer_name = "BatchRenderer" if use_batch else "Rasterizer"
        print(f"[ObstacleAvoidanceEnv] Renderer: {renderer_name} (cfg={cfg_val!r})")

        scene_kwargs = dict(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(7.0, 0.0, 5.0),
                camera_lookat=(3.0, 3.0, 3.0),
                camera_fov=60,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=self.show_viewer,
        )

        if use_batch:
            scene_kwargs["renderer"] = gs.renderers.BatchRenderer()
        else:
            scene_kwargs["vis_options"] = gs.options.VisOptions(
                rendered_envs_idx=list(range(self.num_envs)),
                env_separate_rigid=True,
            )

        scene = gs.Scene(**scene_kwargs)

        scene.add_entity(gs.morphs.Plane())

        # Target visualization sphere
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

        # Obstacle boxes
        ox, oy, oz = self.obstacle_size
        self.obstacles = []
        for _ in range(self.num_obstacles):
            obs_entity = scene.add_entity(
                morph=gs.morphs.Box(
                    size=(ox, oy, oz),
                    pos=(0.0, 0.0, oz / 2.0),  # base on ground
                    fixed=True,
                    collision=False,  # collision handled via distance check
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(0.8, 0.3, 0.2))
                ),
            )
            self.obstacles.append(obs_entity)

        # Forward-facing depth camera (pose set dynamically in _render_depth)
        self.depth_camera = scene.add_camera(
            res=(self.depth_res, self.depth_res),
            pos=(0, 0, 5),
            lookat=(0, 0, 0),
            fov=90,
            GUI=False,
        )

        # Drone
        self.drone = scene.add_entity(
            gs.morphs.Drone(
                file="../assets/robots/draugas/draugas_genesis.urdf",
                pos=(0, 0, 3.0),
                euler=(0, 0, 0),
                propellers_link_name=[
                    "prop0_link",
                    "prop1_link",
                    "prop2_link",
                    "prop3_link",
                ],
                propellers_spin=[1, -1, 1, -1],
            )
        )

        # Extra cameras (e.g. recording camera) — must be added before scene.build()
        # so the BatchRenderer allocates buffers for them.
        self.extra_cameras = []
        if pre_build_cameras:
            for cam_cfg in pre_build_cameras:
                self.extra_cameras.append(scene.add_camera(**cam_cfg))

        env_spacing = self.env_cfg.get("env_spacing", 40.0)
        scene.build(n_envs=self.num_envs, env_spacing=(env_spacing, env_spacing))
        self.scene = scene

        # Attach depth camera to drone body for BatchRenderer (batched move_to_attach)
        if self._using_batch_renderer:
            base_link = self.drone.get_link("main_body")
            offset_T = np.eye(4)
            # Base rotation: cam -Z → body +X, cam +Y → body +Z (horizontal forward)
            # Then tilt 45° downward: compose with R_x(-45°) in camera frame
            # Result: look direction = body [1, 0, -1]/√2 (forward + 45° down)
            c = np.cos(np.radians(45))  # √2/2
            offset_T[:3, :3] = np.array(
                [
                    [0, c, -c],
                    [-1, 0, 0],
                    [0, c, c],
                ],
                dtype=np.float64,
            )
            offset_T[2, 3] = -0.1  # 0.1m below drone center
            self.depth_camera.attach(base_link, offset_T)

        if not self._using_batch_renderer and self.num_envs > 16:
            print(
                f"[NOTE] Per-env depth rendering with {self.num_envs} envs "
                f"may be slow. Use -B 4 for local dev."
            )

        # Drone base link
        self.drone_base_link = self.drone.get_link("main_body")

        # Initial orientation
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # PID controller
        self.controller = CascadingPIDController(
            drone=self.drone,
            dt=self.dt,
            base_rpm=self.env_cfg["pid_params"]["base_rpm"],
            max_rpm=self.env_cfg["pid_params"]["max_rpm"],
            pid_params=self.env_cfg["pid_params"],
            n_envs=self.num_envs,
            device=gs.device,
        )

        # Reward scaling: per-step rewards *= decision_dt (time-independent magnitude)
        per_step_rewards = {
            "time",
            "distance",
            "obstacle_proximity",
            "distance_flat",
        }
        for name in self.reward_scales:
            if name in per_step_rewards:
                self.reward_scales[name] *= self.decision_dt

        self.reward_functions = {
            name: getattr(self, "_reward_" + name) for name in self.reward_scales
        }
        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
            for name in self.reward_scales
        }

        # Observation buffers
        self.state_buf = torch.zeros(
            (self.num_envs, self.num_state_obs), device=gs.device, dtype=gs.tc_float
        )
        self.depth_buf = torch.zeros(
            (self.num_envs, self.depth_stack_size, self.depth_res, self.depth_res),
            device=gs.device,
            dtype=gs.tc_float,
        )
        # Last known forward direction (Rasterizer fallback only — not needed with attach())
        if not self._using_batch_renderer:
            self._last_forward = torch.zeros(
                (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
            )
            self._last_forward[:, 0] = 1.0  # default: +X

        # Reward / reset buffers
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )

        # Action buffers
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)

        # State buffers
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

        # Target
        self.target_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.rel_pos = torch.zeros_like(self.base_pos)
        self.last_rel_pos = torch.zeros_like(self.base_pos)

        # Obstacle positions: (n_envs, n_obstacles, 3) — center of each box
        self.obstacle_positions = torch.zeros(
            (self.num_envs, self.num_obstacles, 3), device=gs.device, dtype=gs.tc_float
        )

        # Obstacle collision / proximity
        self.obstacle_collision = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.bool
        )
        self.min_obstacle_dist = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )

        # Unstable buffer: once a drone becomes unstable in an episode,
        # it stays frozen at frozen_pos (no gravity, no movement) until reset.
        self.unstable_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.bool
        )
        self.frozen_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

        self.global_step = 0
        self.extras = {}

        # Debug / monitoring accumulators (logged via extras["episode"])
        self.action_saturated_count = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self.action_sum = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.obs_clipped_count = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )

        self._built = True

        self.reset()

    # ------------------------------------------------------------------
    # Obstacle placement strategies
    # ------------------------------------------------------------------

    def _place_obstacles_random(self, envs_idx, spawn_pos, n, oz_val):
        """Curriculum phase: sparse random obstacles in a wide area."""
        ox_range = self.env_cfg.get("obstacle_x_range", [-8.0, 12.0])
        oy_range = self.env_cfg.get("obstacle_y_range", [-8.0, 12.0])
        curriculum_n = self.env_cfg.get("curriculum_n_obstacles", self.num_obstacles)

        for i, obs_entity in enumerate(self.obstacles):
            if i < curriculum_n:
                new_x = gs_rand_float(*ox_range, (n,), gs.device)
                new_y = gs_rand_float(*oy_range, (n,), gs.device)
                new_z = torch.full((n,), oz_val, device=gs.device)
            else:
                # Move inactive obstacles underground
                new_x = torch.zeros(n, device=gs.device)
                new_y = torch.zeros(n, device=gs.device)
                new_z = torch.full((n,), -100.0, device=gs.device)

            new_pos = torch.stack([new_x, new_y, new_z], dim=-1)
            obs_entity.set_pos(new_pos, envs_idx=envs_idx, zero_velocity=True)
            self.obstacle_positions[envs_idx, i] = new_pos

    def _place_obstacles_strategic(self, envs_idx, spawn_pos, n, oz_val):
        """Post-curriculum: dense placement with guaranteed path blocker.

        Layout:
        - 1 guaranteed blocker on the direct spawn→target XY path
        - (n_corridor - 1) additional corridor obstacles with wider lateral spread
        - n_ring obstacles in a ring around the target
        - remaining obstacles randomly near the target
        """
        target = self.target_pos[envs_idx]  # (n, 3)

        # XY direction from spawn to target
        direction_xy = target[:, :2] - spawn_pos[:, :2]  # (n, 2)
        path_length = torch.norm(direction_xy, dim=1, keepdim=True)  # (n, 1)
        dir_norm = direction_xy / (path_length + 1e-6)  # (n, 2)
        perp = torch.stack([-dir_norm[:, 1], dir_norm[:, 0]], dim=-1)  # (n, 2)

        n_corridor = self.env_cfg.get("n_corridor_obstacles", 3)
        n_ring = self.env_cfg.get("n_ring_obstacles", 4)
        ring_r_min, ring_r_max = self.env_cfg.get("ring_radius_range", [1.5, 3.5])
        lateral_max = self.env_cfg.get("corridor_lateral_offset", 2.0)

        obs_idx = 0

        # --- Guaranteed blocker: near-zero lateral offset ---
        t = gs_rand_float(0.4, 0.6, (n, 1), gs.device)
        pos_on_path = spawn_pos[:, :2] + t * direction_xy
        offset = gs_rand_float(-0.3, 0.3, (n, 1), gs.device)
        pos_xy = pos_on_path + offset * perp
        self._set_obstacle(obs_idx, pos_xy, oz_val, n, envs_idx)
        obs_idx += 1

        # --- Additional corridor obstacles: spread along path ---
        for i in range(1, n_corridor):
            t_lo = 0.15 + (i - 1) * 0.25
            t_hi = t_lo + 0.25
            t = gs_rand_float(t_lo, t_hi, (n, 1), gs.device)
            pos_on_path = spawn_pos[:, :2] + t * direction_xy
            offset = gs_rand_float(-lateral_max, lateral_max, (n, 1), gs.device)
            pos_xy = pos_on_path + offset * perp
            self._set_obstacle(obs_idx, pos_xy, oz_val, n, envs_idx)
            obs_idx += 1

        # --- Ring obstacles: around the target ---
        for _ in range(n_ring):
            if obs_idx >= self.num_obstacles:
                break
            angle = gs_rand_float(0, 2 * math.pi, (n,), gs.device)
            radius = gs_rand_float(ring_r_min, ring_r_max, (n,), gs.device)
            rx = target[:, 0] + torch.cos(angle) * radius
            ry = target[:, 1] + torch.sin(angle) * radius
            pos_xy = torch.stack([rx, ry], dim=-1)
            self._set_obstacle(obs_idx, pos_xy, oz_val, n, envs_idx)
            obs_idx += 1

        # --- Remaining: random near target ---
        near_range = self.env_cfg.get("post_curriculum_range", 5.0)
        while obs_idx < self.num_obstacles:
            rx = target[:, 0] + gs_rand_float(-near_range, near_range, (n,), gs.device)
            ry = target[:, 1] + gs_rand_float(-near_range, near_range, (n,), gs.device)
            pos_xy = torch.stack([rx, ry], dim=-1)
            self._set_obstacle(obs_idx, pos_xy, oz_val, n, envs_idx)
            obs_idx += 1

    def _set_obstacle(self, idx, pos_xy, oz_val, n, envs_idx):
        """Helper: set obstacle position from XY coordinates."""
        new_pos = torch.stack(
            [
                pos_xy[:, 0],
                pos_xy[:, 1],
                torch.full((n,), oz_val, device=gs.device),
            ],
            dim=-1,
        )
        self.obstacles[idx].set_pos(new_pos, envs_idx=envs_idx, zero_velocity=True)
        self.obstacle_positions[envs_idx, idx] = new_pos

    # ------------------------------------------------------------------
    # Depth rendering
    # ------------------------------------------------------------------

    def _render_depth(self):
        """Dispatch to BatchRenderer or Rasterizer depth rendering path."""
        if self._using_batch_renderer:
            self._render_depth_batch()
        else:
            self._render_depth_rasterizer()

    def _render_depth_batch(self):
        """Render depth for all envs in one BatchRenderer call via attach()."""
        self.depth_camera.move_to_attach()
        _, depth_raw, _, _ = self.depth_camera.render(depth=True, segmentation=False)
        # depth_raw: (n_envs, H, W) tensor from BatchRenderer
        depth = torch.as_tensor(depth_raw, dtype=gs.tc_float, device=gs.device)
        new_depth = torch.clamp(depth / self.max_depth, 0.0, 1.0)  # (n_envs, H, W)

        if self.depth_stack_size > 1:
            self.depth_buf[:, :-1] = self.depth_buf[:, 1:].clone()
        self.depth_buf[:, -1] = new_depth

    def _render_depth_rasterizer(self):
        """Render forward-facing depth per env with Rasterizer (serial loop).

        Camera looks along the drone's body X-axis projected to the XY
        plane (horizontal forward), tilted 45° downward.

        Rasterizer limitation: env_separate_rigid only renders env 0's
        rigid bodies in depth output. Workaround: for each env_i, swap
        env 0's obstacle positions to match env_i, render, extract
        depth[0]. set_pos() updates rasterizer visuals immediately
        (no scene.step() needed).
        """
        forward_body = torch.tensor([[1.0, 0.0, 0.0]], device=gs.device)
        env_0_idx = torch.tensor([0], device=gs.device)

        for env_i in range(self.num_envs):
            # Forward direction from drone quaternion (body X -> world)
            fwd = transform_by_quat(forward_body, self.base_quat[env_i : env_i + 1])[
                0
            ]  # (3,)
            # Project to XY plane for horizontal look direction
            fwd_xy = fwd.clone()
            fwd_xy[2] = 0.0
            fwd_len = torch.norm(fwd_xy)
            if fwd_len < 1e-4:
                # Drone looking straight up/down — use last known heading
                fwd_xy = self._last_forward[env_i]
            else:
                fwd_xy = fwd_xy / fwd_len
                self._last_forward[env_i] = fwd_xy

            cam_pos = self.base_pos[env_i].clone()
            cam_pos[2] -= 0.1  # slightly below drone centre
            # 45° downward tilt: equal forward + down components
            cam_lookat = (
                cam_pos + fwd_xy + torch.tensor([0.0, 0.0, -1.0], device=gs.device)
            )

            # Swap env 0's obstacles to env_i's positions for rendering
            if env_i > 0:
                for j, obs_entity in enumerate(self.obstacles):
                    obs_entity.set_pos(
                        self.obstacle_positions[env_i, j].unsqueeze(0),
                        envs_idx=env_0_idx,
                        zero_velocity=True,
                    )

            self.depth_camera.set_pose(pos=cam_pos, lookat=cam_lookat)
            _, depth_raw, _, _ = self.depth_camera.render(
                depth=True, segmentation=False
            )
            depth = torch.as_tensor(depth_raw, dtype=gs.tc_float, device=gs.device)
            # Always extract depth[0] — Rasterizer only renders env 0's rigid bodies
            new_depth = torch.clamp(depth[0] / self.max_depth, 0.0, 1.0)

            # Frame stacking: shift history left, insert new frame
            if self.depth_stack_size > 1:
                self.depth_buf[env_i, :-1] = self.depth_buf[env_i, 1:].clone()
            self.depth_buf[env_i, -1] = new_depth

        # Restore env 0's original obstacle positions
        if self.num_envs > 1:
            for j, obs_entity in enumerate(self.obstacles):
                obs_entity.set_pos(
                    self.obstacle_positions[0, j].unsqueeze(0),
                    envs_idx=env_0_idx,
                    zero_velocity=True,
                )

    # ------------------------------------------------------------------
    # rsl-rl interface
    # ------------------------------------------------------------------

    def step(self, actions, substep_callback=None):
        self.actions = torch.clip(actions, -1.0, 1.0)

        # Accumulate debug metrics
        self.action_sum += self.actions
        self.action_saturated_count += (
            (torch.abs(self.actions) > 0.99).any(dim=1).float()
        )

        scales = self.env_cfg["action_scales"]
        target_x = self.base_pos[:, 0] + self.actions[:, 0] * scales[0]
        target_y = self.base_pos[:, 1] + self.actions[:, 1] * scales[1]
        target_z = self.base_pos[:, 2] + self.actions[:, 2] * scales[2]
        target_yaw = self.actions[:, 3] * 180.0
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
            if substep_callback is not None:
                substep_callback()

            # Track whether drone stays inside target area throughout decision step
            substep_dist = torch.norm(self.target_pos - self.drone.get_pos(), dim=1)
            inside_target_all_substeps &= (substep_dist < self.env_cfg["hover_radius"])

        # Update state
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.target_pos - self.base_pos
        self.last_rel_pos = self.target_pos - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()

        base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # AABB obstacle distance check (distance to box surface, 0 when inside)
        # base_pos: (n_envs, 3), obstacle_positions: (n_envs, n_obs, 3)
        diff = (
            self.base_pos.unsqueeze(1) - self.obstacle_positions
        )  # (n_envs, n_obs, 3)
        clamped = torch.clamp(torch.abs(diff) - self.obstacle_half_extents, min=0.0)
        obs_dists = torch.norm(clamped, dim=-1)  # (n_envs, n_obs)
        self.min_obstacle_dist = obs_dists.min(dim=1).values
        self.obstacle_collision = self.min_obstacle_dist < self.collision_radius

        # Termination
        self.crash_condition = (
            (self.base_pos[:, 2] < 0.2)
            | (torch.abs(base_euler[:, 0]) > 60.0)
            | (torch.abs(base_euler[:, 1]) > 60.0)
            | (torch.norm(self.rel_pos, dim=1) > 50.0)
            | self.obstacle_collision
        )
        # Success: drone stayed within hover_radius for the entire decision step
        self.success_condition = inside_target_all_substeps
        timeout_condition = self.episode_length_buf > self.max_episode_length

        self.reset_buf = (
            timeout_condition | self.crash_condition | self.success_condition
        )

        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, dtype=gs.tc_float)
        self.extras["time_outs"][timeout_condition] = 1.0

        self.global_step += 1

        # Rewards BEFORE reset
        self.rew_buf[:] = 0.0
        self.reward_components = {}
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.reward_components[name] = rew
            self.episode_sums[name] += rew

        # Auto-reset AFTER rewards, BEFORE obs
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # Observations (post-reset)
        obs = self._compute_obs()
        self.last_actions[:] = self.actions[:]

        # Detect obs clipping (any scaled dim at ±1 boundary)
        clipped = self.state_buf[:, :3].abs().gt(0.99).any(dim=1) | self.state_buf[
            :, 7:13
        ].abs().gt(0.99).any(dim=1)
        self.obs_clipped_count += clipped.float()

        return obs, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        n = len(envs_idx)

        # Randomize target (always at configured ranges)
        tx = gs_rand_float(*self.env_cfg["target_x_range"], (n,), gs.device)
        ty = gs_rand_float(*self.env_cfg["target_y_range"], (n,), gs.device)
        tz = gs_rand_float(*self.env_cfg["target_z_range"], (n,), gs.device)
        self.target_pos[envs_idx] = torch.stack([tx, ty, tz], dim=-1)

        # Randomize drone spawn
        offset = self.env_cfg["spawn_offset"]
        sx = gs_rand_float(-offset, offset, (n,), gs.device)
        sy = gs_rand_float(-offset, offset, (n,), gs.device)
        sz = gs_rand_float(
            self.env_cfg["spawn_height_min"],
            self.env_cfg["spawn_height_max"],
            (n,),
            gs.device,
        )
        spawn_pos = torch.stack([sx, sy, sz], dim=-1)

        # Randomize obstacles (curriculum controls density only)
        curriculum_steps = self.env_cfg.get("curriculum_steps", 0)
        oz_val = self.obstacle_size[2] / 2.0  # half-height so base sits on ground

        if self.global_step < curriculum_steps:
            # Phase 1: no obstacles — learn full-distance navigation first
            # Move entities underground so depth camera sees clear space
            underground = torch.zeros((n, 3), device=gs.device, dtype=gs.tc_float)
            underground[:, 2] = -100.0
            for i, obs_entity in enumerate(self.obstacles):
                obs_entity.set_pos(underground, envs_idx=envs_idx, zero_velocity=True)
                self.obstacle_positions[envs_idx, i] = underground
        else:
            # Phase 2: strategic obstacle placement with guaranteed path blocker
            self._place_obstacles_strategic(envs_idx, spawn_pos, n, oz_val)

        # Set drone state
        self.base_pos[envs_idx] = spawn_pos
        self.last_base_pos[envs_idx] = spawn_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        self.drone.set_pos(spawn_pos, zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(
            self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.rel_pos = self.target_pos - self.base_pos
        self.last_rel_pos = self.target_pos - self.last_base_pos

        self.controller.reset_idx(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.unstable_buf[envs_idx] = False

        # Clear depth history for reset envs (new episode, new obstacle layout)
        self.depth_buf[envs_idx] = 0.0
        if hasattr(self, "_last_forward"):
            self._last_forward[envs_idx, 0] = 1.0
            self._last_forward[envs_idx, 1] = 0.0
            self._last_forward[envs_idx, 2] = 0.0

        # Log episode stats
        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # Debug / monitoring metrics
        ep_len = self.episode_length_buf[envs_idx].float().clamp(min=1)
        mean_ep_len = ep_len.mean()
        self.extras["episode"]["terminal_dist"] = (
            torch.norm(self.rel_pos[envs_idx], dim=1).mean().item()
        )
        self.extras["episode"]["action_saturation_rate"] = (
            self.action_saturated_count[envs_idx].mean() / mean_ep_len
        ).item()
        self.extras["episode"]["obs_clip_rate"] = (
            self.obs_clipped_count[envs_idx].mean() / mean_ep_len
        ).item()
        action_names = ["ax", "ay", "az", "ayaw"]
        action_mean = self.action_sum[envs_idx] / mean_ep_len
        for i, name in enumerate(action_names):
            self.extras["episode"][f"action_mean_{name}"] = (
                action_mean[:, i].mean().item()
            )

        # Reset accumulators
        self.action_saturated_count[envs_idx] = 0.0
        self.action_sum[envs_idx] = 0.0
        self.obs_clipped_count[envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        # Sync rasterizer with new obstacle/drone positions from reset_idx
        self.scene.step()
        return self.get_observations()

    def get_observations(self) -> TensorDict:
        return self._compute_obs()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _compute_obs(self) -> TensorDict:
        s = self.obs_scales

        # State vector (17-dim, identical to CoordinateLandingEnv)
        self.state_buf[:] = torch.cat(
            [
                torch.clip(self.rel_pos * s["rel_pos"], -1, 1),  # 3
                self.base_quat,  # 4
                torch.clip(self.base_lin_vel * s["lin_vel"], -1, 1),  # 3
                torch.clip(self.base_ang_vel * s["ang_vel"], -1, 1),  # 3
                self.last_actions,  # 4
            ],
            dim=-1,
        )  # total: 17

        # Depth image — render every render_interval steps, reuse cached buffer otherwise
        if self.global_step % self.render_interval == 0:
            self._render_depth()

        return TensorDict(
            {
                "state": self.state_buf.clone(),
                "depth": self.depth_buf.clone(),
            },
            batch_size=[self.num_envs],
        )

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _reward_distance(self):
        """Penalty proportional to distance from target."""
        return torch.norm(self.rel_pos, dim=1)

    def _reward_distance_flat(self):
        """Flat bonus/penalty: +1 if drone moved closer to target, -1 if further."""
        prev_dist = torch.norm(self.last_rel_pos, dim=1)
        curr_dist = torch.norm(self.rel_pos, dim=1)
        return torch.sign(prev_dist - curr_dist)

    def _reward_time(self):
        return torch.ones(self.num_envs, device=gs.device, dtype=gs.tc_float)

    def _reward_obstacle_proximity(self):
        """Penalty when within safety_radius of any obstacle.
        Linearly increases as drone gets closer: max(0, 1 - dist/safety_radius)."""
        proximity = torch.clamp(
            1.0 - self.min_obstacle_dist / self.safety_radius, min=0.0
        )
        return proximity

    def _reward_crash(self):
        """Penalty on any crash (ground, tilt, out-of-bounds, or obstacle collision)."""
        rew = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        rew[self.crash_condition] = 1.0
        return rew

    def _reward_success(self):
        """Reward on success."""
        rew = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        rew[self.success_condition] = 1.0
        return rew
