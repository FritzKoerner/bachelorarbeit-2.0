#!/usr/bin/env python3
"""Record a single landing episode as MP4 video for visual confirmation on HPC.

Runs headless (no viewer, no W&B). Output: a single MP4 file you can scp back.
Obstacles and target sphere are visible in the video.

Captures frames at physics-step resolution (not decision-step) for smooth video,
using a substep_callback inside env.step(). The recording camera is registered
pre-build so it works with both BatchRenderer and Rasterizer.

Usage:
    python record_landing.py --hpc my_run              # latest checkpoint
    python record_landing.py --hpc my_run --ckpt 400   # specific checkpoint
    python record_landing.py --ckpt 300                # local logs/
    python record_landing.py --hpc list                # list available HPC runs
"""

import argparse
import copy
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
from eval_rl_wb import find_latest_checkpoint, resolve_hpc_log_dir
from train_rl_wb import DictConfig  # needed to unpickle cfgs.pkl
from visualize_paths import _build_path_mesh


# ---------------------------------------------------------------------------
# Video recording (two-pass: dry run for framing, replay for capture)
# ---------------------------------------------------------------------------

def record_landing(EnvClass, env_cfg, obs_cfg, reward_cfg, train_cfg,
                   log_dir, ckpt, seed=42, render_every=1, res=(960, 540)):
    """Record one landing episode and save as MP4.

    Pass 1 runs the full episode to determine the trajectory and compute
    optimal camera framing.  Pass 2 replays with the same seed, capturing
    a frame every ``render_every`` physics substeps via substep_callback.

    Returns (out_path, outcome, final_dist, num_frames).
    """
    cfg = copy.deepcopy(env_cfg)
    cfg["visualize_target"] = True
    cfg["curriculum_steps"] = 0

    # Recording camera at dummy position — repositioned after dry run.
    # Added via pre_build_cameras so BatchRenderer allocates buffers for it.
    rec_cam_cfg = dict(
        res=res, pos=(0, 0, 30), lookat=(0, 0, 0), fov=90, GUI=False,
    )

    render_env = EnvClass(
        num_envs=1, env_cfg=cfg, obs_cfg=obs_cfg,
        reward_cfg=copy.deepcopy(reward_cfg), show_viewer=False,
    )
    render_env.build(pre_build_cameras=[rec_cam_cfg])
    rec_cam = render_env.extra_cameras[0]

    runner = OnPolicyRunner(render_env, copy.deepcopy(train_cfg), log_dir,
                            device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    max_steps = render_env.max_episode_length

    # --- Pass 1: dry run to get full path for camera framing ---
    torch.manual_seed(seed)
    obs = render_env.reset()
    start_pos = render_env.base_pos[0].cpu().numpy().copy()
    target = render_env.target_pos[0].cpu().numpy().copy()

    positions = [start_pos.copy()]
    outcome = "timeout"
    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = render_env.step(actions)
            if dones[0]:
                if render_env.success_condition[0].item():
                    outcome = "success"
                elif render_env.obstacle_collision[0].item():
                    outcome = "obstacle_collision"
                elif render_env.crash_condition[0].item():
                    outcome = "crash"
                break
            positions.append(render_env.base_pos[0].cpu().numpy().copy())

    positions = np.array(positions)
    final_dist = float(np.linalg.norm(positions[-1] - target))

    # Compute camera framing from full path + obstacle positions
    obs_positions = render_env.obstacle_positions[0].cpu().numpy()
    all_points = np.vstack([positions, target.reshape(1, 3), obs_positions])
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = np.linalg.norm(bbox_max - bbox_min)

    fov = 90
    padding = 1.4
    dist = (extent * padding) / (2.0 * math.tan(math.radians(fov / 2.0)))
    dist = max(dist, 2.0)

    cam_pos = (
        center[0] + dist * 0.15,
        center[1] - dist * 0.75,
        center[2] + dist * 0.55,
    )
    lookat = tuple(center)

    # Reposition recording camera to computed framing
    rec_cam.set_pose(pos=cam_pos, lookat=lookat)

    # --- Pass 2: replay with per-substep frame capture ---
    torch.manual_seed(seed)
    obs = render_env.reset()
    ctx = render_env.scene.visualizer.context

    frames = []
    substep_count = [0]
    trail_positions = [render_env.drone.get_pos()[0].cpu().numpy().copy()]
    last_drawn_idx = [0]
    trail_interval = max(1, 50 // render_every)  # draw trail every ~50 substeps

    def capture_substep():
        substep_count[0] += 1
        if substep_count[0] % render_every != 0:
            return

        # Update trail periodically
        pos = render_env.drone.get_pos()[0].cpu().numpy().copy()
        trail_positions.append(pos)
        n = len(trail_positions)
        if n > last_drawn_idx[0] + trail_interval:
            new_pts = np.array(trail_positions[last_drawn_idx[0]:n])
            mesh = _build_path_mesh(
                new_pts, radius=0.015, color=(0.0, 1.0, 0.0, 0.8),
            )
            if mesh is not None:
                ctx.draw_debug_mesh(mesh)
            last_drawn_idx[0] = n - 1

        rgb, _, _, _ = rec_cam.render()
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        frames.append(rgb)

    with torch.no_grad():
        for step_i in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = render_env.step(
                actions, substep_callback=capture_substep,
            )
            if dones[0]:
                break

    # Draw final trail segment
    n = len(trail_positions)
    if n > last_drawn_idx[0] + 1:
        new_pts = np.array(trail_positions[last_drawn_idx[0]:n])
        mesh = _build_path_mesh(
            new_pts, radius=0.015, color=(0.0, 1.0, 0.0, 0.8),
        )
        if mesh is not None:
            ctx.draw_debug_mesh(mesh)

    ctx.clear_debug_objects()

    # Save as MP4
    if not frames:
        print("No frames captured!")
        return None, outcome, 0.0, 0

    out_path = os.path.join(log_dir, f"landing_ckpt_{ckpt}.mp4")
    # Video FPS: real-time physics rate / render_every, capped at 50
    physics_fps = 1.0 / render_env.dt
    fps = min(50.0, max(10.0, physics_fps / render_every))

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    return out_path, outcome, final_dist, len(frames)


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
    args = parser.parse_args()

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

    # Eval overrides
    reward_cfg["reward_scales"] = {}

    result = record_landing(
        EnvClass, env_cfg, obs_cfg, reward_cfg, train_cfg,
        log_dir, ckpt_iter, seed=args.seed, render_every=args.render_every,
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
