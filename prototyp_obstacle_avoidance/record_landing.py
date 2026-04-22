#!/usr/bin/env python3
"""Record a single landing episode as MP4 video for visual confirmation on HPC.

Runs headless (no viewer, no W&B). Output: a single MP4 file you can scp back.
Obstacles and target sphere are visible in the video.

Two-pass, two-scene design:
  Pass 1: run the policy in the real env (BatchRenderer + depth@64x64, unchanged)
          and record per-substep drone pose into a plain trajectory struct.
  Pass 2: tear down the inference scene, then build a minimal viz-only scene
          with just a 960x540 recording camera (no depth camera), and replay
          the trajectory via drone.set_pos/set_quat each substep.

Why two scenes? BatchRenderer requires all cameras in a scene to share the
same resolution. A 64x64 depth cam (required for the policy) cannot coexist
with a 960x540 video cam in one scene. Option B preserves exact training
observations while allowing a high-res video.

Usage:
    python record_landing.py --hpc my_run              # latest checkpoint
    python record_landing.py --hpc my_run --ckpt 400   # specific checkpoint
    python record_landing.py --ckpt 300                # local logs/
    python record_landing.py --hpc list                # list available HPC runs
"""

import argparse
import copy
import gc
import math
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from envs.obstacle_avoidance_env import ObstacleAvoidanceEnv
from envs.obstacle_avoidance_env_v2 import ObstacleAvoidanceEnvV2
from eval_rl_wb import find_latest_checkpoint, resolve_hpc_log_dir, apply_placement_override
from train_rl_wb import DictConfig  # needed to unpickle cfgs.pkl
from visualize_paths import _build_path_mesh


# ---------------------------------------------------------------------------
# Pass 1 -- run policy in the real env and record trajectory
# ---------------------------------------------------------------------------

def _pass1_record_trajectory(EnvClass, env_cfg, obs_cfg, reward_cfg, train_cfg,
                             log_dir, ckpt, seed, no_obstacles=False):
    """Run one episode with the policy; return a plain-data trajectory record.

    The inference env is unchanged from training -- same BatchRenderer, same
    64x64 depth camera, same attach()-based positioning. No recording camera
    is added here, so the resolution-uniformity constraint is trivially
    satisfied.
    """
    cfg = copy.deepcopy(env_cfg)
    cfg["visualize_target"] = True
    # Default: strategic obstacles. --no-obstacles parks them underground (Phase 1).
    cfg["curriculum_steps"] = float("inf") if no_obstacles else 0

    render_env = EnvClass(
        num_envs=1, env_cfg=cfg, obs_cfg=obs_cfg,
        reward_cfg=copy.deepcopy(reward_cfg), show_viewer=False,
    )
    render_env.build()

    runner = OnPolicyRunner(render_env, copy.deepcopy(train_cfg), log_dir,
                            device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    torch.manual_seed(seed)
    obs = render_env.reset()

    # Statics -- recorded once immediately after reset.
    target_pos = render_env.target_pos[0].cpu().numpy().copy()
    obstacle_positions = render_env.obstacle_positions[0].cpu().numpy().copy()
    start_pos = render_env.drone.get_pos()[0].cpu().numpy().copy()
    start_quat = render_env.drone.get_quat()[0].cpu().numpy().copy()

    positions = [start_pos]
    quaternions = [start_quat]

    def record_substep():
        positions.append(render_env.drone.get_pos()[0].cpu().numpy().copy())
        quaternions.append(render_env.drone.get_quat()[0].cpu().numpy().copy())

    outcome = "timeout"
    max_steps = render_env.max_episode_length
    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = render_env.step(
                actions, substep_callback=record_substep,
            )
            if dones[0]:
                if render_env.success_condition[0].item():
                    outcome = "success"
                elif render_env.obstacle_collision[0].item():
                    outcome = "obstacle_collision"
                elif render_env.crash_condition[0].item():
                    outcome = "crash"
                break

    positions = np.array(positions)
    quaternions = np.array(quaternions)
    final_dist = float(np.linalg.norm(positions[-1] - target_pos))

    record = dict(
        positions=positions,
        quaternions=quaternions,
        obstacle_positions=obstacle_positions,
        target_pos=target_pos,
        obstacle_size=tuple(cfg.get("obstacle_size", [1.0, 1.0, 2.0])),
        dt=render_env.dt,
        outcome=outcome,
        final_dist=final_dist,
        max_depth=float(cfg.get("max_depth", 20.0)),
    )

    # Tear down inference scene before returning. Callers still do a gc pass.
    del policy
    del runner
    del render_env
    return record


# ---------------------------------------------------------------------------
# Pass 2 -- replay trajectory in a viz-only scene with a single high-res cam
# ---------------------------------------------------------------------------

def _compute_camera_framing(positions, target_pos, obstacle_positions, fov=90):
    """Frame the camera so the full trajectory + obstacles + target are visible."""
    all_points = np.vstack([
        positions, target_pos.reshape(1, 3), obstacle_positions,
    ])
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = float(np.linalg.norm(bbox_max - bbox_min))

    padding = 1.4
    dist = (extent * padding) / (2.0 * math.tan(math.radians(fov / 2.0)))
    dist = max(dist, 2.0)

    cam_pos = (
        float(center[0] + dist * 0.15),
        float(center[1] - dist * 0.75),
        float(center[2] + dist * 0.55),
    )
    lookat = tuple(float(c) for c in center)
    return cam_pos, lookat


def _pov_camera_pose(drone_pos, drone_quat_wxyz):
    """Mirror the live env's POV-cam geometry: 0.1 m below the drone body,
    looking horizontally forward + 45 deg downward.

    Returns (pos, lookat) as numpy arrays in world coords.
    """
    qw, qx, qy, qz = drone_quat_wxyz
    fwd_world = R.from_quat([qx, qy, qz, qw]).apply(np.array([1.0, 0.0, 0.0]))
    fwd_xy = fwd_world.copy()
    fwd_xy[2] = 0.0
    norm = np.linalg.norm(fwd_xy)
    if norm < 1e-4:
        fwd_xy = np.array([1.0, 0.0, 0.0])
    else:
        fwd_xy /= norm

    pos = np.asarray(drone_pos, dtype=np.float64).copy()
    pos[2] -= 0.1
    lookat = pos + fwd_xy + np.array([0.0, 0.0, -1.0])
    return pos, lookat


def _depth_to_viridis_uint8(depth_m, max_depth):
    """Match the env's CNN-input normalization, then apply viridis.

    depth_m is the raw metric depth from camera.render(depth=True).
    Uses the same `clamp(depth/max_depth, 0, 1)` mapping as the live env so
    the POV depth video shows what the CNN sees, just at higher resolution.
    """
    if isinstance(depth_m, torch.Tensor):
        depth_m = depth_m.cpu().numpy()
    if depth_m.ndim == 3:
        depth_m = depth_m[0]  # (1, H, W) -> (H, W) when env_separate_rigid
    depth_norm = np.clip(depth_m / max_depth, 0.0, 1.0)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
    # COLORMAP_VIRIDIS yields a BGR uint8 image suitable for cv2.VideoWriter.
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)


def _pass2_render_video(record, out_path, res=(1920, 1080), render_every=1,
                        pov_rgb_path=None, pov_depth_path=None,
                        pov_res=(480, 480), pov_fov=90):
    """Build a minimal viz scene, replay the recorded trajectory, encode MP4.

    No physics stepping, no policy, no PID. The drone is moved kinematically
    each substep via set_pos/set_quat. Pass 2 uses the Rasterizer (set_pos
    updates visuals directly), which has no uniform-resolution constraint --
    so we can co-host the high-res third-person cam and a body-attached POV
    cam in the same scene.

    If pov_rgb_path / pov_depth_path are provided, also writes drone-POV
    videos rendered with the same geometry as the live env's depth camera
    (45 deg downtilt, 0.1 m below body center).
    """
    positions = record["positions"]
    quaternions = record["quaternions"]
    obstacle_positions = record["obstacle_positions"]
    target_pos = record["target_pos"]
    ox, oy, oz = record["obstacle_size"]
    dt = record["dt"]
    max_depth = record.get("max_depth", 20.0)

    fov = 90
    cam_pos, lookat = _compute_camera_framing(
        positions, target_pos, obstacle_positions, fov=fov,
    )
    # Genesis Rasterizer default is far=20, which clips obstacles on the far
    # side of the bbox when the camera is offset ~10-20 m from center. Pin
    # far to (3x cam-to-center distance, min 100 m) so nothing is culled.
    cam_center_dist = float(np.linalg.norm(
        np.asarray(cam_pos) - np.asarray(lookat)
    ))
    rec_far = max(cam_center_dist * 3.0, 100.0)

    # Minimal scene: no collision, no gravity -- we never step physics.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=(0, 0, 0)),
        viewer_options=gs.options.ViewerOptions(max_FPS=60),
        rigid_options=gs.options.RigidOptions(dt=dt, enable_collision=False),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
            env_separate_rigid=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    # Target sphere (pos set after build).
    target_vis = scene.add_entity(
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

    # Obstacle boxes at recorded positions (static within episode).
    obstacle_entities = []
    for opos in obstacle_positions:
        obs_entity = scene.add_entity(
            morph=gs.morphs.Box(
                size=(ox, oy, oz),
                pos=(float(opos[0]), float(opos[1]), float(opos[2])),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.8, 0.3, 0.2))
            ),
        )
        obstacle_entities.append(obs_entity)

    drone = scene.add_entity(
        gs.morphs.Drone(
            file="../assets/robots/draugas/draugas_genesis.urdf",
            pos=(float(positions[0][0]), float(positions[0][1]), float(positions[0][2])),
            euler=(0, 0, 0),
            propellers_link_name=[
                "prop0_link", "prop1_link", "prop2_link", "prop3_link",
            ],
            propellers_spin=[1, -1, 1, -1],
        )
    )

    rec_cam = scene.add_camera(
        res=res, pos=cam_pos, lookat=lookat, fov=fov, GUI=False,
        far=rec_far,
    )

    record_pov = pov_rgb_path is not None or pov_depth_path is not None
    if record_pov:
        # Initial pose is fine -- it gets overwritten before every render.
        pov_cam = scene.add_camera(
            res=pov_res,
            pos=(float(positions[0][0]), float(positions[0][1]), float(positions[0][2])),
            lookat=(float(positions[0][0]) + 1.0, float(positions[0][1]), 0.0),
            fov=pov_fov,
            GUI=False,
        )

    scene.build(n_envs=1, env_spacing=(40.0, 40.0))

    device = gs.device
    env0 = torch.tensor([0], device=device)

    # Pin target sphere at its recorded position.
    target_vis.set_pos(
        torch.tensor([target_pos], device=device, dtype=gs.tc_float),
        envs_idx=env0, zero_velocity=True,
    )

    ctx = scene.visualizer.context

    # Clamp raised to 120 so render_every=1 gives real-time 100 fps playback
    # (dt=0.01 -> physics_fps=100). The old 50-cap made episodes play 2x slow.
    physics_fps = 1.0 / dt
    fps = min(120.0, max(10.0, physics_fps / render_every))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Stream frames directly to the encoders: a 60 s episode with decimation=300
    # is ~6000 substeps, which at 1920x1080x3 uint8 would otherwise pin ~37 GB
    # of frames in RAM (plus ~8 GB for POV RGB+depth at 480x480). That is what
    # killed HPC jobs via the SLURM --mem oom_kill. Writers are lazily opened
    # on the first frame so we can keep using actual rendered shape (which
    # differs from `res` when env_separate_rigid returns batched arrays).
    writer = None
    pov_rgb_writer = None
    pov_depth_writer = None

    num_frames = 0
    trail_positions = []
    last_drawn_idx = 0
    trail_interval = max(1, 50 // render_every)

    n_frames = len(positions)
    for t in range(n_frames):
        # Skip frames per render_every, but always capture the last one.
        if t % render_every != 0 and t != n_frames - 1:
            continue

        drone.set_pos(
            torch.tensor([positions[t]], device=device, dtype=gs.tc_float),
            envs_idx=env0, zero_velocity=True,
        )
        drone.set_quat(
            torch.tensor([quaternions[t]], device=device, dtype=gs.tc_float),
            envs_idx=env0, zero_velocity=True,
        )
        # Articulated robots don't auto-sync to the rasterizer on set_pos
        # alone -- mirrors the env's reset() trick at obstacle_avoidance_env_v2.py:808.
        # With gravity=0 and zero RPM, physics is a no-op; the drone stays put.
        scene.step()

        # POV render BEFORE the trail debug-mesh is updated so the FPV cam
        # never sees the agent's own past path.
        if record_pov:
            pov_pos, pov_lookat = _pov_camera_pose(positions[t], quaternions[t])
            pov_cam.set_pose(pos=tuple(pov_pos), lookat=tuple(pov_lookat))
            pov_rgb, pov_depth, _, _ = pov_cam.render(depth=True)
            if isinstance(pov_rgb, torch.Tensor):
                pov_rgb = pov_rgb.cpu().numpy()
            if pov_rgb.ndim == 4:
                pov_rgb = pov_rgb[0]

            # POV frames are rotated 90 deg CW so "up" in the video matches
            # the drone's body-up axis; the 45 deg downward look otherwise
            # produces a sideways-feeling FPV. Rotation swaps width/height
            # in the encoder's expected dims.
            if pov_rgb_path is not None:
                rot_rgb = cv2.rotate(pov_rgb, cv2.ROTATE_90_CLOCKWISE)
                if pov_rgb_writer is None:
                    rh, rw = rot_rgb.shape[:2]
                    pov_rgb_writer = cv2.VideoWriter(
                        pov_rgb_path, fourcc, fps, (rw, rh),
                    )
                pov_rgb_writer.write(cv2.cvtColor(rot_rgb, cv2.COLOR_RGB2BGR))

            if pov_depth_path is not None:
                # _depth_to_viridis_uint8 already returns BGR uint8.
                depth_bgr = _depth_to_viridis_uint8(pov_depth, max_depth)
                rot_depth = cv2.rotate(depth_bgr, cv2.ROTATE_90_CLOCKWISE)
                if pov_depth_writer is None:
                    rh, rw = rot_depth.shape[:2]
                    pov_depth_writer = cv2.VideoWriter(
                        pov_depth_path, fourcc, fps, (rw, rh),
                    )
                pov_depth_writer.write(rot_depth)

        # Incremental trail update (only affects the third-person view).
        trail_positions.append(positions[t])
        n = len(trail_positions)
        if n > last_drawn_idx + trail_interval:
            new_pts = np.array(trail_positions[last_drawn_idx:n])
            mesh = _build_path_mesh(
                new_pts, radius=0.015, color=(0.0, 1.0, 0.0, 0.8),
            )
            if mesh is not None:
                ctx.draw_debug_mesh(mesh)
            last_drawn_idx = n - 1

        rgb, _, _, _ = rec_cam.render()
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        # env_separate_rigid may return a batched (1, H, W, C) array.
        if rgb.ndim == 4:
            rgb = rgb[0]

        if writer is None:
            h, w = rgb.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        num_frames += 1

    # Final trail segment (visualized only on the third-person cam; does not
    # retroactively affect already-written frames, kept for parity with prior
    # behavior of flushing the tail to the debug context).
    n = len(trail_positions)
    if n > last_drawn_idx + 1:
        new_pts = np.array(trail_positions[last_drawn_idx:n])
        mesh = _build_path_mesh(
            new_pts, radius=0.015, color=(0.0, 1.0, 0.0, 0.8),
        )
        if mesh is not None:
            ctx.draw_debug_mesh(mesh)

    ctx.clear_debug_objects()

    if writer is not None:
        writer.release()
    if pov_rgb_writer is not None:
        pov_rgb_writer.release()
    if pov_depth_writer is not None:
        pov_depth_writer.release()

    return num_frames


# ---------------------------------------------------------------------------
# Orchestrator -- preserves the old record_landing() signature
# ---------------------------------------------------------------------------

def record_landing(EnvClass, env_cfg, obs_cfg, reward_cfg, train_cfg,
                   log_dir, eval_dir, ckpt, seed=42, render_every=1,
                   res=(1920, 1080), no_obstacles=False, record_pov=True,
                   pov_res=(480, 480)):
    """Record one landing episode and save as MP4(s).

    Pass 1 runs the policy in the real env to record the trajectory.
    Pass 2 replays that trajectory in a minimal scene and renders:
      - third-person video (always)
      - drone-POV RGB + viridis-depth videos (when record_pov=True)

    ``log_dir`` is where the checkpoint lives; ``eval_dir`` is where MP4s
    are written (usually ``{log_dir}/evals/{run_name}/``).

    Returns (out_path, outcome, final_dist, num_frames). The POV side outputs
    are written next to out_path with `_pov_rgb` / `_pov_depth` suffixes.
    """
    record = _pass1_record_trajectory(
        EnvClass, env_cfg, obs_cfg, reward_cfg, train_cfg,
        log_dir, ckpt, seed, no_obstacles=no_obstacles,
    )

    # Force teardown of the inference scene before building the viz scene.
    # Required so Madrona/Vulkan buffers from Pass 1 are released in time.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_path = os.path.join(eval_dir, f"landing_ckpt_{ckpt}.mp4")
    pov_rgb_path = os.path.join(eval_dir, f"landing_ckpt_{ckpt}_pov_rgb.mp4") if record_pov else None
    pov_depth_path = os.path.join(eval_dir, f"landing_ckpt_{ckpt}_pov_depth.mp4") if record_pov else None

    num_frames = _pass2_render_video(
        record, out_path, res=res, render_every=render_every,
        pov_rgb_path=pov_rgb_path, pov_depth_path=pov_depth_path,
        pov_res=pov_res,
    )
    if num_frames == 0:
        return None, record["outcome"], record["final_dist"], 0
    return out_path, record["outcome"], record["final_dist"], num_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record a single landing episode as MP4 video")
    parser.add_argument("-e", "--exp_name", type=str, default="obstacle-avoidance")
    parser.add_argument("--hpc", nargs="?", const="list", default=None,
                        help="HPC run name (no value = list available runs)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Direct path to log dir (overrides --exp_name)")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint iteration (default: latest)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-every", type=int, default=1,
                        help="Capture every Nth physics substep (default: 1 = all)")
    parser.add_argument("--res", type=str, default="1920x1080",
                        help="Third-person video resolution WxH (default: 1920x1080)")
    parser.add_argument("--pov-res", type=str, default="480x480",
                        help="Drone-POV video resolution WxH (default: 480x480)")
    parser.add_argument("--no-pov", action="store_true",
                        help="Skip drone-POV RGB+depth videos (default: also record them)")
    parser.add_argument("--no-obstacles", action="store_true",
                        help="Hide all obstacles (curriculum Phase 1). Default is strategic placement.")
    parser.add_argument("--placement", choices=["strategic", "vineyard"], default=None,
                        help="Override placement strategy regardless of how the model was trained. "
                             "Default: use whatever cfgs.pkl specifies.")
    parser.add_argument("--name", type=str, default=None,
                        help="Name for this eval run. Videos land in "
                             "logs/{exp}/evals/{name}/landing_ckpt_{ckpt}*.mp4. "
                             "Default: iter{ckpt}.")
    args = parser.parse_args()

    def _parse_wxh(s, label):
        try:
            w_str, h_str = s.lower().split("x")
            return (int(w_str), int(h_str))
        except ValueError:
            print(f"Invalid --{label} '{s}'; expected format WxH (e.g. 1920x1080)")
            sys.exit(1)

    res = _parse_wxh(args.res, "res")
    pov_res = _parse_wxh(args.pov_res, "pov-res")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # Resolve log directory
    if args.hpc is not None:
        log_dir = resolve_hpc_log_dir(args.hpc)
    elif args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = f"logs/{args.exp_name}"

    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Find checkpoint
    if args.ckpt is not None:
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        if not os.path.isfile(resume_path):
            print(f"Checkpoint not found: {resume_path}")
            sys.exit(1)
        ckpt_iter = args.ckpt
    else:
        resume_path, ckpt_iter = find_latest_checkpoint(log_dir)

    # Detect v2 env from saved reward keys
    is_v2 = "progress" in reward_cfg.get("reward_scales", {})
    EnvClass = ObstacleAvoidanceEnvV2 if is_v2 else ObstacleAvoidanceEnv

    # Eval run identity and artifact directory.
    run_name = args.name or f"iter{ckpt_iter}"
    eval_dir = os.path.join(log_dir, "evals", run_name)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Recording landing video ...")
    print(f"  Checkpoint : {resume_path}  (iteration {ckpt_iter})")
    print(f"  Env        : {'v2' if is_v2 else 'v1'}")
    print(f"  Run name   : {run_name}")
    print(f"  Seed       : {args.seed}")
    print(f"  Render     : every {args.render_every} substep(s)")
    print(f"  3rd-person : {res[0]}x{res[1]}")
    if not args.no_pov:
        print(f"  POV        : {pov_res[0]}x{pov_res[1]} (RGB + depth)")

    # Eval overrides
    reward_cfg["reward_scales"] = {}
    apply_placement_override(env_cfg, args.placement)

    result = record_landing(
        EnvClass, env_cfg, obs_cfg, reward_cfg, train_cfg,
        log_dir, eval_dir, ckpt_iter,
        seed=args.seed, render_every=args.render_every,
        res=res, no_obstacles=args.no_obstacles,
        record_pov=not args.no_pov, pov_res=pov_res,
    )

    if result[0] is None:
        print("Recording failed.")
        sys.exit(1)

    out_path, outcome, final_dist, num_frames = result
    print(f"\n{'='*48}")
    print(f"  Outcome         : {outcome}")
    print(f"  Final dist      : {final_dist:.3f} m")
    print(f"  Frames          : {num_frames}")
    print(f"  3rd-person MP4  : {out_path}")
    if not args.no_pov:
        base = out_path[:-len(".mp4")]
        print(f"  POV RGB MP4     : {base}_pov_rgb.mp4")
        print(f"  POV depth MP4   : {base}_pov_depth.mp4")
    print(f"{'='*48}")


if __name__ == "__main__":
    main()
