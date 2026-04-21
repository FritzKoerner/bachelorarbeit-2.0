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


def _pass2_render_video(record, out_path, res=(1920, 1080), render_every=1):
    """Build a minimal viz scene, replay the recorded trajectory, encode MP4.

    No physics stepping, no policy, no PID. The drone is moved kinematically
    each substep via set_pos/set_quat. Because there is only one camera, the
    BatchRenderer uniform-resolution constraint does not apply -- renderer
    choice is free. We use the Rasterizer (set_pos updates visuals directly).
    """
    positions = record["positions"]
    quaternions = record["quaternions"]
    obstacle_positions = record["obstacle_positions"]
    target_pos = record["target_pos"]
    ox, oy, oz = record["obstacle_size"]
    dt = record["dt"]

    fov = 90
    cam_pos, lookat = _compute_camera_framing(
        positions, target_pos, obstacle_positions, fov=fov,
    )

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

    # Replay loop.
    frames = []
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

        # Incremental trail update.
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
        frames.append(rgb)

    # Final trail segment.
    n = len(trail_positions)
    if n > last_drawn_idx + 1:
        new_pts = np.array(trail_positions[last_drawn_idx:n])
        mesh = _build_path_mesh(
            new_pts, radius=0.015, color=(0.0, 1.0, 0.0, 0.8),
        )
        if mesh is not None:
            ctx.draw_debug_mesh(mesh)

    ctx.clear_debug_objects()

    if not frames:
        return 0

    # Clamp raised to 120 so render_every=1 gives real-time 100 fps playback
    # (dt=0.01 -> physics_fps=100). The old 50-cap made episodes play 2x slow.
    physics_fps = 1.0 / dt
    fps = min(120.0, max(10.0, physics_fps / render_every))

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    return len(frames)


# ---------------------------------------------------------------------------
# Orchestrator -- preserves the old record_landing() signature
# ---------------------------------------------------------------------------

def record_landing(EnvClass, env_cfg, obs_cfg, reward_cfg, train_cfg,
                   log_dir, ckpt, seed=42, render_every=1, res=(1920, 1080),
                   no_obstacles=False):
    """Record one landing episode and save as MP4.

    Pass 1 runs the policy in the real env to record the trajectory.
    Pass 2 replays that trajectory in a minimal scene with only the video cam.

    Returns (out_path, outcome, final_dist, num_frames).
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

    out_path = os.path.join(log_dir, f"landing_ckpt_{ckpt}.mp4")
    num_frames = _pass2_render_video(record, out_path, res=res,
                                      render_every=render_every)
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
                        help="Video resolution WxH (default: 1920x1080)")
    parser.add_argument("--no-obstacles", action="store_true",
                        help="Hide all obstacles (curriculum Phase 1). Default is strategic placement.")
    parser.add_argument("--placement", choices=["strategic", "vineyard"], default=None,
                        help="Override placement strategy regardless of how the model was trained. "
                             "Default: use whatever cfgs.pkl specifies.")
    args = parser.parse_args()

    try:
        w_str, h_str = args.res.lower().split("x")
        res = (int(w_str), int(h_str))
    except ValueError:
        print(f"Invalid --res '{args.res}'; expected format WxH (e.g. 1920x1080)")
        sys.exit(1)

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

    print(f"Recording landing video ...")
    print(f"  Checkpoint : {resume_path}  (iteration {ckpt_iter})")
    print(f"  Env        : {'v2' if is_v2 else 'v1'}")
    print(f"  Seed       : {args.seed}")
    print(f"  Render     : every {args.render_every} substep(s)")
    print(f"  Resolution : {res[0]}x{res[1]}")

    # Eval overrides
    reward_cfg["reward_scales"] = {}
    apply_placement_override(env_cfg, args.placement)

    result = record_landing(
        EnvClass, env_cfg, obs_cfg, reward_cfg, train_cfg,
        log_dir, ckpt_iter, seed=args.seed, render_every=args.render_every,
        res=res, no_obstacles=args.no_obstacles,
    )

    if result[0] is None:
        print("Recording failed.")
        sys.exit(1)

    out_path, outcome, final_dist, num_frames = result
    print(f"\n{'='*48}")
    print(f"  Outcome      : {outcome}")
    print(f"  Final dist   : {final_dist:.3f} m")
    print(f"  Frames       : {num_frames}")
    print(f"  Video saved  : {out_path}")
    print(f"{'='*48}")


if __name__ == "__main__":
    main()
