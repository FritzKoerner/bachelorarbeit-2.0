"""
CorridorNavigationEnv: forked from ObstacleAvoidanceEnvV2.

Fixed-axis corridor slalom along +X.  The drone spawns at one end,
navigates past mixed-shape obstacles packed into 4 X-slices, and must
hover at a fixed target at the far end.  Leaving the corridor bounding
box terminates the episode (hard wall — no flying around/over/under).

Differences from ObstacleAvoidanceEnvV2:
  * Obstacle set is fixed at 8 entities with a baked-in shape mix:
    2 Boxes, 2 Spheres, 2 Cylinders, 2 tall thin pillar cylinders.
  * Collision / proximity distance is shape-aware (box AABB, sphere
    radial, cylinder radial+axial) — dispatched via a per-obstacle
    `shape_type` tensor, still fully vectorised across the n_obs axis.
  * Single placement strategy (`_place_obstacles_corridor`) — strategic
    and vineyard placers are gone.
  * Target is fixed at env_cfg["corridor_target_pos"]; spawn is uniform
    in a configurable spawn box.
  * `out_of_corridor` termination + same-sign reward component so
    corridor escapes surface separately from physics crashes in W&B.
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
    """Check if Madrona BatchRenderer is available (Turing+ required)."""
    try:
        import gs_madrona  # noqa: F401
    except ImportError:
        return False
    if gs.backend != gs.cuda:
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) >= (7, 5)
    except Exception:
        return False


# Shape-type integer codes used in self.obstacle_shape_type.
_SHAPE_BOX = 0
_SHAPE_SPHERE = 1
_SHAPE_CYLINDER = 2


class CorridorNavigationEnv:
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
        self.num_obs = self.num_state_obs
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]  # 4
        self.device = gs.device

        self.dt = 0.01  # 100 Hz
        self.decimation = env_cfg.get("decimation", 1)
        self.decision_dt = self.dt * self.decimation
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

        # Obstacle spec lists — each has length 2, total 8 obstacles.
        self.box_sizes = [tuple(s) for s in env_cfg["corridor_box_sizes"]]
        self.sphere_radii_cfg = list(env_cfg["corridor_sphere_radii"])
        self.cylinder_specs_cfg = [tuple(s) for s in env_cfg["corridor_cylinder_specs"]]
        self.pillar_specs_cfg = [tuple(s) for s in env_cfg["corridor_pillar_specs"]]

        # Shape-interleaved obstacle layout:
        # [Box, Sphere, Cylinder, Pillar, Box, Sphere, Cylinder, Pillar]
        # Pairing into 4 slices (indices (0,1), (2,3), (4,5), (6,7)) gives
        # each corridor slice two different shape types.
        assert (
            len(self.box_sizes) == 2
            and len(self.sphere_radii_cfg) == 2
            and len(self.cylinder_specs_cfg) == 2
            and len(self.pillar_specs_cfg) == 2
        ), "Corridor env expects exactly 2 of each shape type"
        self._obstacle_specs = []
        for i in range(2):
            bx = self.box_sizes[i]
            self._obstacle_specs.append(
                dict(morph="box", size=bx, shape_type=_SHAPE_BOX, z_rest=bx[2] / 2.0,
                     is_pillar=False, is_wall=False)
            )
            sr = float(self.sphere_radii_cfg[i])
            # Sphere sits on corridor floor (z = 0.3), not the global ground,
            # so its centre is lifted by 0.3 to avoid half-burying small radii.
            self._obstacle_specs.append(
                dict(morph="sphere", radius=sr, shape_type=_SHAPE_SPHERE, z_rest=sr + 0.3,
                     is_pillar=False, is_wall=False)
            )
            cr, ch = self.cylinder_specs_cfg[i]
            self._obstacle_specs.append(
                dict(
                    morph="cylinder",
                    radius=float(cr),
                    height=float(ch),
                    shape_type=_SHAPE_CYLINDER,
                    z_rest=float(ch) / 2.0,
                    is_pillar=False,
                    is_wall=False,
                )
            )
            pr, ph = self.pillar_specs_cfg[i]
            self._obstacle_specs.append(
                dict(
                    morph="cylinder",
                    radius=float(pr),
                    height=float(ph),
                    shape_type=_SHAPE_CYLINDER,
                    z_rest=float(ph) / 2.0,
                    is_pillar=True,
                    is_wall=False,
                )
            )
        self.num_obstacles = len(self._obstacle_specs)  # 8

        self.collision_radius = env_cfg.get("collision_radius", 0.3)
        self.safety_radius = env_cfg.get("safety_radius", 3.0)

        # Corridor geometry
        self.corridor_x_range = list(env_cfg["corridor_x_range"])
        self.corridor_y_range = list(env_cfg["corridor_y_range"])
        self.corridor_z_range = list(env_cfg["corridor_z_range"])
        self.spawn_x_range = list(env_cfg["corridor_spawn_x_range"])
        self.spawn_y_range = list(env_cfg["corridor_spawn_y_range"])
        self.spawn_z = float(env_cfg["corridor_spawn_z"])
        self.target_pos_val = list(env_cfg["corridor_target_pos"])
        self.slice_centres = list(env_cfg["corridor_slice_centres"])
        self.min_gap = float(env_cfg.get("corridor_min_gap", 2.0))  # noqa: F841 - documented constraint; guaranteed by Y-sampling ranges

        # Spawn->target line for line-anchored obstacle placement.
        spawn_cx = 0.5 * (self.spawn_x_range[0] + self.spawn_x_range[1])
        spawn_cy = 0.5 * (self.spawn_y_range[0] + self.spawn_y_range[1])
        self.line_start = torch.tensor(
            [spawn_cx, spawn_cy, self.spawn_z],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.line_end = torch.tensor(
            self.target_pos_val, device=gs.device, dtype=gs.tc_float
        )
        self.pair_gap_min = float(env_cfg["corridor_pair_gap_min"])
        self.pair_gap_max = float(env_cfg["corridor_pair_gap_max"])
        self.first_offset_min = float(env_cfg["corridor_first_obstacle_offset_min"])
        self.first_offset_max = float(env_cfg["corridor_first_obstacle_offset_max"])

        # Per-slice pillar flag — when True, pair placement is locked to Y-axis
        # because pillars (height ~4 m) cannot fit vertical detours in a 5.7 m corridor.
        n_slices = len(self.slice_centres)
        self._pair_has_pillar = [
            bool(self._obstacle_specs[2 * s].get("is_pillar", False)
                 or self._obstacle_specs[2 * s + 1].get("is_pillar", False))
            for s in range(n_slices)
        ]

        # Corridor boundary walls + ceiling as physical obstacles.
        # Inside face of each wall coincides with the corridor bounding box, so
        # "crashing on a wall" replaces the abstract "crossed y=±4 / z=6" check
        # as the primary boundary signal. Walls always occupy indices
        # [num_obs_random, ..., num_obs_random+2], so _place_obstacles_corridor
        # (which only touches 2*slice_i and 2*slice_i+1) leaves them untouched.
        y_min, y_max = self.corridor_y_range
        z_min, z_max = self.corridor_z_range
        x_min, x_max = self.corridor_x_range
        wall_thickness = 0.2
        wall_x_len = x_max - x_min
        wall_z_len = z_max - z_min
        wall_x_centre = 0.5 * (x_min + x_max)
        wall_z_centre = 0.5 * (z_min + z_max)
        ceiling_y_len = y_max - y_min

        self._obstacle_specs.append(dict(  # left wall
            morph="box",
            size=(wall_x_len, wall_thickness, wall_z_len),
            shape_type=_SHAPE_BOX,
            z_rest=wall_z_centre,
            is_pillar=False, is_wall=True,
            fixed_pos=(wall_x_centre, y_min - wall_thickness / 2.0, wall_z_centre),
        ))
        self._obstacle_specs.append(dict(  # right wall
            morph="box",
            size=(wall_x_len, wall_thickness, wall_z_len),
            shape_type=_SHAPE_BOX,
            z_rest=wall_z_centre,
            is_pillar=False, is_wall=True,
            fixed_pos=(wall_x_centre, y_max + wall_thickness / 2.0, wall_z_centre),
        ))
        self._obstacle_specs.append(dict(  # ceiling
            morph="box",
            size=(wall_x_len, ceiling_y_len, wall_thickness),
            shape_type=_SHAPE_BOX,
            z_rest=z_max + wall_thickness / 2.0,
            is_pillar=False, is_wall=True,
            fixed_pos=(wall_x_centre, 0.5 * (y_min + y_max), z_max + wall_thickness / 2.0),
        ))
        self.num_obstacles = len(self._obstacle_specs)  # 8 random + 3 walls = 11

        # Depth camera params
        self.depth_res = obs_cfg.get("depth_res", 64)
        self.render_interval = env_cfg.get("render_interval", 2)
        self.max_depth = env_cfg.get("max_depth", 20.0)
        self.depth_stack_size = obs_cfg.get("depth_stack_size", 1)
        self._use_batch_renderer_cfg = env_cfg.get("use_batch_renderer", "auto")

        self._built = False

    # ------------------------------------------------------------------
    # Two-phase construction
    # ------------------------------------------------------------------

    def build(self, pre_build_cameras=None):
        """Create Genesis scene with mixed-shape obstacles, depth camera, drone, PID, and buffers."""
        cfg_val = self._use_batch_renderer_cfg
        if cfg_val == "auto":
            use_batch = _can_use_batch_renderer()
        elif cfg_val is True or cfg_val == "batch":
            use_batch = True
        else:
            use_batch = False
        self._using_batch_renderer = use_batch
        renderer_name = "BatchRenderer" if use_batch else "Rasterizer"
        print(f"[CorridorNavigationEnv] Renderer: {renderer_name} (cfg={cfg_val!r})")

        scene_kwargs = dict(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                # Diagonal top-corner view so the full corridor is visible in -v mode.
                camera_pos=(-4.0, -6.0, 8.0),
                camera_lookat=(16.0, 0.0, 2.0),
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

        # Mixed-shape obstacles (8 random) + boundary walls/ceiling (3 fixed) = 11 total.
        # Walls are constructed at their fixed positions and never moved (neither
        # by _place_obstacles_corridor nor by the curriculum underground trick),
        # so they stay visible to the depth camera from step 0.
        self.obstacles = []
        for s in self._obstacle_specs:
            is_wall = s.get("is_wall", False)
            init_pos = s.get("fixed_pos", (0.0, 0.0, s["z_rest"]))
            color = (0.45, 0.45, 0.48) if is_wall else (0.8, 0.3, 0.2)
            if s["morph"] == "box":
                morph = gs.morphs.Box(
                    size=s["size"],
                    pos=init_pos,
                    fixed=True,
                    collision=False,
                )
            elif s["morph"] == "sphere":
                morph = gs.morphs.Sphere(
                    radius=s["radius"],
                    pos=init_pos,
                    fixed=True,
                    collision=False,
                )
            else:  # cylinder (includes pillars)
                morph = gs.morphs.Cylinder(
                    radius=s["radius"],
                    height=s["height"],
                    pos=init_pos,
                    fixed=True,
                    collision=False,
                )
            obs_entity = scene.add_entity(
                morph=morph,
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=color)
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

        # Extra cameras (e.g. recording) — must be added before scene.build().
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
            c = np.cos(np.radians(45))  # sqrt(2)/2
            offset_T[:3, :3] = np.array(
                [
                    [0, c, -c],
                    [-1, 0, 0],
                    [0, c, c],
                ],
                dtype=np.float64,
            )
            offset_T[2, 3] = -0.1
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

        # Per-obstacle shape tensors for vectorised collision dispatch.
        self.obstacle_shape_type = torch.tensor(
            [s["shape_type"] for s in self._obstacle_specs],
            device=gs.device,
            dtype=torch.long,
        )
        self.obstacle_box_half_extents = torch.zeros(
            (self.num_obstacles, 3), device=gs.device, dtype=gs.tc_float
        )
        self.obstacle_sphere_radii = torch.zeros(
            (self.num_obstacles,), device=gs.device, dtype=gs.tc_float
        )
        self.obstacle_cyl_radii = torch.zeros(
            (self.num_obstacles,), device=gs.device, dtype=gs.tc_float
        )
        self.obstacle_cyl_half_heights = torch.zeros(
            (self.num_obstacles,), device=gs.device, dtype=gs.tc_float
        )
        self.obstacle_z_rest = torch.zeros(
            (self.num_obstacles,), device=gs.device, dtype=gs.tc_float
        )
        for i, s in enumerate(self._obstacle_specs):
            self.obstacle_z_rest[i] = s["z_rest"]
            if s["shape_type"] == _SHAPE_BOX:
                sx, sy, sz = s["size"]
                self.obstacle_box_half_extents[i] = torch.tensor(
                    [sx / 2.0, sy / 2.0, sz / 2.0], device=gs.device
                )
            elif s["shape_type"] == _SHAPE_SPHERE:
                self.obstacle_sphere_radii[i] = s["radius"]
            else:  # cylinder
                self.obstacle_cyl_radii[i] = s["radius"]
                self.obstacle_cyl_half_heights[i] = s["height"] / 2.0

        # Reward machinery
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
        if not self._using_batch_renderer:
            self._last_forward = torch.zeros(
                (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
            )
            self._last_forward[:, 0] = 1.0  # default: +X (corridor long axis)

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

        # Obstacle positions: (n_envs, n_obstacles, 3) -- centre of each obstacle.
        self.obstacle_positions = torch.zeros(
            (self.num_envs, self.num_obstacles, 3),
            device=gs.device,
            dtype=gs.tc_float,
        )
        # Walls are fixed in all envs; pre-populate their rows so proximity /
        # crash distance checks see them from step 0 (including during the
        # curriculum phase when random obstacles are hidden underground).
        for i, s in enumerate(self._obstacle_specs):
            if s.get("is_wall", False):
                fp = torch.tensor(s["fixed_pos"], device=gs.device, dtype=gs.tc_float)
                self.obstacle_positions[:, i, :] = fp

        # Obstacle collision / proximity
        self.obstacle_collision = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.bool
        )
        self.min_obstacle_dist = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )

        # Out-of-corridor buffer (one-hot per env, True for the step the drone
        # crosses the bounding box and is about to be reset).
        self.out_of_corridor_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.bool
        )

        # Unstable buffer
        self.unstable_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.bool
        )
        self.frozen_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )

        self.global_step = 0
        self.extras = {}

        # Debug accumulators
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
    # Obstacle placement
    # ------------------------------------------------------------------

    def _place_obstacles_corridor(self, envs_idx, n):
        """Place 8 obstacles in 4 X-slices as pair-straddling slaloms.

        Per slice, per env:
          axis ~ Uniform{Y, Z}        (forced to Y when the pair contains a pillar)
          side ~ Uniform{-1, +1}
          d    ~ Uniform[first_offset_min, first_offset_max]
          gap  ~ Uniform[pair_gap_min, pair_gap_max]

        Obstacle A sits at `line_along(X) + side * d` along the chosen axis.
        Obstacle B sits on the other side of A with exactly `gap` metres
        between their facing surfaces, computed via

            B_along = A_along - side * (A_half + gap + B_half).

        On the non-chosen perpendicular axis both obstacles sit on the
        spawn->target line at that X. No corridor clamping is applied:
        obstacles may poke through the bounding-box walls; the drone's
        out-of-corridor termination still fires if the drone itself leaves.
        """
        x_jitter_mag = 0.2

        line_start = self.line_start
        line_end = self.line_end
        line_len_x = line_end[0] - line_start[0]

        for slice_i, x_centre in enumerate(self.slice_centres):
            idx_a = 2 * slice_i
            idx_b = 2 * slice_i + 1

            # Independent X jitter for A and B so the pair doesn't look glued.
            x_a = torch.full((n,), x_centre, device=gs.device, dtype=gs.tc_float) \
                + gs_rand_float(-x_jitter_mag, x_jitter_mag, (n,), gs.device)
            x_b = torch.full((n,), x_centre, device=gs.device, dtype=gs.tc_float) \
                + gs_rand_float(-x_jitter_mag, x_jitter_mag, (n,), gs.device)

            # Spawn->target line Y/Z at this X: reference for both the
            # non-chosen axis (obstacles pass through the line) and the
            # chosen axis (zero-offset origin for the `side * d` offset).
            t = (x_a - line_start[0]) / line_len_x
            y_line = line_start[1] + t * (line_end[1] - line_start[1])
            z_line = line_start[2] + t * (line_end[2] - line_start[2])

            # Axis choice: pillars can't fit vertical detours in a ~5.7 m corridor.
            if self._pair_has_pillar[slice_i]:
                axis = torch.zeros((n,), device=gs.device, dtype=torch.long)
            else:
                axis = torch.randint(0, 2, (n,), device=gs.device)

            sign = torch.where(
                torch.randint(0, 2, (n,), device=gs.device) == 0,
                torch.full((n,), -1.0, device=gs.device, dtype=gs.tc_float),
                torch.full((n,), 1.0, device=gs.device, dtype=gs.tc_float),
            )
            d = gs_rand_float(self.first_offset_min, self.first_offset_max, (n,), gs.device)
            gap = gs_rand_float(self.pair_gap_min, self.pair_gap_max, (n,), gs.device)

            y_half_a = self._obstacle_y_half(idx_a)
            y_half_b = self._obstacle_y_half(idx_b)
            z_half_a = self._obstacle_z_half(idx_a)
            z_half_b = self._obstacle_z_half(idx_b)

            # A: line + side * d along the chosen axis; line on the other axis.
            y_a = torch.where(axis == 0, y_line + sign * d, y_line)
            z_a = torch.where(axis == 1, z_line + sign * d, z_line)

            # B: A - side * (A_half + gap + B_half) along the chosen axis.
            delta_y_ab = sign * (y_half_a + gap + y_half_b)
            delta_z_ab = sign * (z_half_a + gap + z_half_b)
            y_b = torch.where(axis == 0, y_a - delta_y_ab, y_line)
            z_b = torch.where(axis == 1, z_a - delta_z_ab, z_line)

            pos_a = torch.stack([x_a, y_a, z_a], dim=-1)
            pos_b = torch.stack([x_b, y_b, z_b], dim=-1)

            self.obstacles[idx_a].set_pos(pos_a, envs_idx=envs_idx, zero_velocity=True)
            self.obstacles[idx_b].set_pos(pos_b, envs_idx=envs_idx, zero_velocity=True)
            self.obstacle_positions[envs_idx, idx_a] = pos_a
            self.obstacle_positions[envs_idx, idx_b] = pos_b

    # ------------------------------------------------------------------
    # Depth rendering (identical to ObstacleAvoidanceEnvV2)
    # ------------------------------------------------------------------

    def _render_depth(self):
        if self._using_batch_renderer:
            self._render_depth_batch()
        else:
            self._render_depth_rasterizer()

    def _render_depth_batch(self):
        self.depth_camera.move_to_attach()
        _, depth_raw, _, _ = self.depth_camera.render(depth=True, segmentation=False)
        depth = torch.as_tensor(depth_raw, dtype=gs.tc_float, device=gs.device)
        new_depth = torch.clamp(depth / self.max_depth, 0.0, 1.0)

        if self.depth_stack_size > 1:
            self.depth_buf[:, :-1] = self.depth_buf[:, 1:].clone()
        self.depth_buf[:, -1] = new_depth

    def _render_depth_rasterizer(self):
        forward_body = torch.tensor([[1.0, 0.0, 0.0]], device=gs.device)
        env_0_idx = torch.tensor([0], device=gs.device)

        for env_i in range(self.num_envs):
            fwd = transform_by_quat(forward_body, self.base_quat[env_i : env_i + 1])[0]
            fwd_xy = fwd.clone()
            fwd_xy[2] = 0.0
            fwd_len = torch.norm(fwd_xy)
            if fwd_len < 1e-4:
                fwd_xy = self._last_forward[env_i]
            else:
                fwd_xy = fwd_xy / fwd_len
                self._last_forward[env_i] = fwd_xy

            cam_pos = self.base_pos[env_i].clone()
            cam_pos[2] -= 0.1
            cam_lookat = (
                cam_pos + fwd_xy + torch.tensor([0.0, 0.0, -1.0], device=gs.device)
            )

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
            new_depth = torch.clamp(depth[0] / self.max_depth, 0.0, 1.0)

            if self.depth_stack_size > 1:
                self.depth_buf[env_i, :-1] = self.depth_buf[env_i, 1:].clone()
            self.depth_buf[env_i, -1] = new_depth

        if self.num_envs > 1:
            for j, obs_entity in enumerate(self.obstacles):
                obs_entity.set_pos(
                    self.obstacle_positions[0, j].unsqueeze(0),
                    envs_idx=env_0_idx,
                    zero_velocity=True,
                )

    # ------------------------------------------------------------------
    # Shape-aware collision dispatch
    # ------------------------------------------------------------------

    def _obstacle_y_half(self, idx: int) -> float:
        """Half-extent along Y for obstacle ``idx`` (for corridor clamping)."""
        t = int(self.obstacle_shape_type[idx].item())
        if t == _SHAPE_BOX:
            return float(self.obstacle_box_half_extents[idx, 1])
        if t == _SHAPE_SPHERE:
            return float(self.obstacle_sphere_radii[idx])
        # vertical cylinder: Y-extent = radius
        return float(self.obstacle_cyl_radii[idx])

    def _obstacle_z_half(self, idx: int) -> float:
        """Half-extent along Z for obstacle ``idx`` (for corridor clamping)."""
        t = int(self.obstacle_shape_type[idx].item())
        if t == _SHAPE_BOX:
            return float(self.obstacle_box_half_extents[idx, 2])
        if t == _SHAPE_SPHERE:
            return float(self.obstacle_sphere_radii[idx])
        return float(self.obstacle_cyl_half_heights[idx])

    def _compute_obstacle_distances(self):
        """Min distance from drone to each obstacle surface, dispatched by shape.

        Returns
        -------
        obs_dists : (n_envs, n_obstacles) tensor
            Distance from drone centre to nearest point on each obstacle.
            0 when drone is inside the obstacle.
        """
        diff = self.base_pos.unsqueeze(1) - self.obstacle_positions  # (n_envs, n_obs, 3)
        abs_diff = torch.abs(diff)

        # Box: AABB distance using per-obstacle half-extents.
        box_dist = torch.norm(
            torch.clamp(
                abs_diff - self.obstacle_box_half_extents.unsqueeze(0), min=0.0
            ),
            dim=-1,
        )

        # Sphere: max(||diff|| - radius, 0).
        sphere_dist = torch.clamp(
            torch.norm(diff, dim=-1) - self.obstacle_sphere_radii.unsqueeze(0),
            min=0.0,
        )

        # Cylinder (vertical axis): radial + axial clamps combined.
        xy_dist = torch.norm(diff[..., :2], dim=-1)
        radial = torch.clamp(
            xy_dist - self.obstacle_cyl_radii.unsqueeze(0), min=0.0
        )
        axial = torch.clamp(
            abs_diff[..., 2] - self.obstacle_cyl_half_heights.unsqueeze(0), min=0.0
        )
        cyl_dist = torch.sqrt(radial ** 2 + axial ** 2)

        shape_type_b = self.obstacle_shape_type.unsqueeze(0)  # (1, n_obs)
        obs_dists = torch.where(
            shape_type_b == _SHAPE_BOX,
            box_dist,
            torch.where(shape_type_b == _SHAPE_SPHERE, sphere_dist, cyl_dist),
        )
        return obs_dists

    # ------------------------------------------------------------------
    # rsl-rl interface
    # ------------------------------------------------------------------

    def step(self, actions, substep_callback=None):
        self.actions = torch.clip(actions, -1.0, 1.0)

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

        inside_target_all_substeps = torch.ones(
            self.num_envs, device=gs.device, dtype=torch.bool
        )
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
            if newly_unstable.any():
                self.frozen_pos[newly_unstable] = curr_pos[newly_unstable].clone()
                self.unstable_buf |= newly_unstable
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

            substep_dist = torch.norm(self.target_pos - self.drone.get_pos(), dim=1)
            inside_target_all_substeps &= substep_dist < self.env_cfg["hover_radius"]

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

        # Obstacle proximity / collision with shape-aware distance.
        obs_dists = self._compute_obstacle_distances()
        self.min_obstacle_dist = obs_dists.min(dim=1).values
        self.obstacle_collision = self.min_obstacle_dist < self.collision_radius

        # Out-of-corridor: any axis outside the bounding box.
        cx_min, cx_max = self.corridor_x_range
        cy_min, cy_max = self.corridor_y_range
        cz_min, cz_max = self.corridor_z_range
        self.out_of_corridor_buf = (
            (self.base_pos[:, 0] < cx_min)
            | (self.base_pos[:, 0] > cx_max)
            | (self.base_pos[:, 1] < cy_min)
            | (self.base_pos[:, 1] > cy_max)
            | (self.base_pos[:, 2] < cz_min)
            | (self.base_pos[:, 2] > cz_max)
        )

        # Termination
        self.crash_condition = (
            (self.base_pos[:, 2] < 0.2)
            | (torch.abs(base_euler[:, 0]) > 60.0)
            | (torch.abs(base_euler[:, 1]) > 60.0)
            | (torch.norm(self.rel_pos, dim=1) > 50.0)
            | self.obstacle_collision
            | self.out_of_corridor_buf
        )
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

        # Detect obs clipping
        clipped = self.state_buf[:, :3].abs().gt(0.99).any(dim=1) | self.state_buf[
            :, 7:13
        ].abs().gt(0.99).any(dim=1)
        self.obs_clipped_count += clipped.float()

        return obs, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        n = len(envs_idx)

        # Target: fixed corridor target (NOT randomised).
        target_pos_t = torch.tensor(
            self.target_pos_val, device=gs.device, dtype=gs.tc_float
        )
        self.target_pos[envs_idx] = target_pos_t.unsqueeze(0).expand(n, -1).clone()

        # Spawn: uniform in spawn box, Z fixed.
        sx = gs_rand_float(*self.spawn_x_range, (n,), gs.device)
        sy = gs_rand_float(*self.spawn_y_range, (n,), gs.device)
        sz = torch.full(
            (n,), self.spawn_z, device=gs.device, dtype=gs.tc_float
        )
        spawn_pos = torch.stack([sx, sy, sz], dim=-1)

        # Curriculum: random obstacles stay underground until curriculum_steps,
        # but walls/ceiling remain in place so the drone learns corridor
        # boundaries visually from step 0.
        curriculum_steps = self.env_cfg.get("curriculum_steps", 0)
        if self.global_step < curriculum_steps:
            underground = torch.zeros((n, 3), device=gs.device, dtype=gs.tc_float)
            underground[:, 2] = -100.0
            for i, obs_entity in enumerate(self.obstacles):
                if self._obstacle_specs[i].get("is_wall", False):
                    continue
                obs_entity.set_pos(underground, envs_idx=envs_idx, zero_velocity=True)
                self.obstacle_positions[envs_idx, i] = underground
        else:
            self._place_obstacles_corridor(envs_idx, n)

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

        ep_len = self.episode_length_buf[envs_idx].float().clamp(min=1)
        mean_ep_len = ep_len.mean()
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

        self.action_saturated_count[envs_idx] = 0.0
        self.action_sum[envs_idx] = 0.0
        self.obs_clipped_count[envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        self.scene.step()
        return self.get_observations()

    def get_observations(self) -> TensorDict:
        return self._compute_obs()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _compute_obs(self) -> TensorDict:
        s = self.obs_scales

        self.state_buf[:] = torch.cat(
            [
                torch.clip(self.rel_pos * s["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * s["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * s["ang_vel"], -1, 1),
                self.last_actions,
            ],
            dim=-1,
        )

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

    def _reward_progress(self):
        """Positive when drone moves closer to target (delta-distance)."""
        prev_dist = torch.norm(self.last_rel_pos, dim=1)
        curr_dist = torch.norm(self.rel_pos, dim=1)
        return prev_dist - curr_dist

    def _reward_close(self):
        """Exponential attraction within ~2 m of target."""
        return torch.exp(-torch.norm(self.rel_pos, dim=1))

    def _reward_obstacle_proximity(self):
        """Linear penalty within safety_radius of any obstacle."""
        return torch.clamp(
            1.0 - self.min_obstacle_dist / self.safety_radius, min=0.0
        )

    def _reward_crash(self):
        """One on any crash (ground/tilt/oob/obstacle/corridor-escape)."""
        rew = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        rew[self.crash_condition] = 1.0
        return rew

    def _reward_success(self):
        """One on successful hover at target."""
        rew = torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)
        rew[self.success_condition] = 1.0
        return rew

    def _reward_out_of_corridor(self):
        """One on corridor-escape. Logged separately so W&B surfaces
        escapes vs. physics/obstacle crashes as distinct failure modes,
        even though both also contribute to the shared `crash` signal."""
        return self.out_of_corridor_buf.float()
