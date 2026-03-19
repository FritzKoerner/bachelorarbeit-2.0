#!/usr/bin/env python3
"""Visualize drone flight paths with obstacles across training checkpoints.

Extends the prototyp_global_coordinate visualizer with:
  - Obstacle boxes drawn in matplotlib plots
  - obstacle_collision as a 4th outcome (purple)
  - Obstacles visible in Genesis screenshots/GIFs automatically

Usage:
    python visualize_paths.py --ckpt 100 300
    python visualize_paths.py --ckpt 100 300 --no_render
    python visualize_paths.py --ckpt 100 300 --video
"""

import argparse
import copy
import math
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import torch
from PIL import Image

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from envs.obstacle_avoidance_env import ObstacleAvoidanceEnv


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_paths(env, policy, num_episodes, seed=42):
    """Run episodes and collect 3D position trajectories + obstacle positions."""
    torch.manual_seed(seed)
    num_envs = env.num_envs

    ep_positions = [[] for _ in range(num_envs)]
    results = []

    obs = env.reset()

    with torch.no_grad():
        while len(results) < num_episodes:
            pos_np = env.base_pos.cpu().numpy()
            for i in range(num_envs):
                ep_positions[i].append(pos_np[i].copy())

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            for i in dones.nonzero(as_tuple=False).flatten().tolist():
                if len(results) >= num_episodes:
                    break
                success    = bool(env.success_condition[i].item())
                crash      = bool(env.crash_condition[i].item())
                obs_collid = bool(env.obstacle_collision[i].item())
                results.append({
                    "positions":          np.array(ep_positions[i]),
                    "success":            success,
                    "crash":              crash and not obs_collid,
                    "obstacle_collision": obs_collid,
                    "timeout":            not success and not crash,
                    "obstacle_positions": env.obstacle_positions[i].cpu().numpy().copy(),
                })
                ep_positions[i] = []

    return results[:num_episodes]


# ---------------------------------------------------------------------------
# Matplotlib plotting
# ---------------------------------------------------------------------------

OUTCOME_COLORS = {
    "success":            "#60aa64",
    "crash":              "#c44a40",
    "timeout":            "#e8a340",
    "obstacle_collision": "#8e44ad",
}


def _outcome_color(r):
    if r["success"]:
        return OUTCOME_COLORS["success"]
    if r.get("obstacle_collision"):
        return OUTCOME_COLORS["obstacle_collision"]
    if r["crash"]:
        return OUTCOME_COLORS["crash"]
    return OUTCOME_COLORS["timeout"]


def _draw_obstacles_xy(ax, obstacle_positions, obstacle_size):
    """Draw obstacle footprints as semi-transparent rectangles in top-down view."""
    ox, oy, _ = obstacle_size
    for pos in obstacle_positions:
        if pos[2] < -10:  # skip hidden obstacles
            continue
        rect = Rectangle(
            (pos[0] - ox / 2, pos[1] - oy / 2), ox, oy,
            linewidth=0.8, edgecolor="#8e44ad", facecolor="#8e44ad", alpha=0.25,
        )
        ax.add_patch(rect)


def _draw_obstacles_xz(ax, obstacle_positions, obstacle_size):
    """Draw obstacle side profiles as rectangles in XZ view."""
    ox, _, oz = obstacle_size
    for pos in obstacle_positions:
        if pos[2] < -10:
            continue
        rect = Rectangle(
            (pos[0] - ox / 2, 0), ox, oz,
            linewidth=0.8, edgecolor="#8e44ad", facecolor="#8e44ad", alpha=0.25,
        )
        ax.add_patch(rect)


def plot_comparison(all_results, ckpt_labels, target_pos, obstacle_size, out_path):
    """3-row comparison figure: 3D / top-down XY / side XZ, one column per ckpt."""
    n_ckpts = len(all_results)
    fig = plt.figure(figsize=(6 * n_ckpts, 16))

    for ci, (results, label) in enumerate(zip(all_results, ckpt_labels)):
        n_ep = len(results)
        succ = sum(r["success"] for r in results)
        crash = sum(r["crash"] for r in results)
        obs_coll = sum(r.get("obstacle_collision", False) for r in results)
        timeout = n_ep - succ - crash - obs_coll

        # Use obstacle positions from first episode for plot backdrop
        obs_pos = results[0]["obstacle_positions"] if results else np.zeros((0, 3))

        # 3D view
        ax = fig.add_subplot(3, n_ckpts, ci + 1, projection="3d")
        for r in results:
            pos = r["positions"]
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                    c=_outcome_color(r), alpha=0.35, linewidth=0.6)
        ax.scatter(*target_pos, c="red", s=30, marker="o", zorder=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{label}\nS:{succ} C:{crash} O:{obs_coll} T:{timeout} (n={n_ep})",
                      fontsize=11, fontweight="bold")

        # Top-down XY
        ax = fig.add_subplot(3, n_ckpts, n_ckpts + ci + 1)
        _draw_obstacles_xy(ax, obs_pos, obstacle_size)
        for r in results:
            pos = r["positions"]
            ax.plot(pos[:, 0], pos[:, 1],
                    c=_outcome_color(r), alpha=0.35, linewidth=0.6)
        ax.scatter(target_pos[0], target_pos[1],
                   c="red", s=80, marker="o", zorder=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Top-down (XY)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Side XZ
        ax = fig.add_subplot(3, n_ckpts, 2 * n_ckpts + ci + 1)
        _draw_obstacles_xz(ax, obs_pos, obstacle_size)
        for r in results:
            pos = r["positions"]
            ax.plot(pos[:, 0], pos[:, 2],
                    c=_outcome_color(r), alpha=0.35, linewidth=0.6)
        ax.scatter(target_pos[0], target_pos[2],
                   c="red", s=80, marker="o", zorder=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title("Side view (XZ)")
        ax.grid(True, alpha=0.3)

    legend_patches = [
        mpatches.Patch(color=OUTCOME_COLORS["success"],            label="Success"),
        mpatches.Patch(color=OUTCOME_COLORS["crash"],              label="Crash"),
        mpatches.Patch(color=OUTCOME_COLORS["obstacle_collision"], label="Obstacle"),
        mpatches.Patch(color=OUTCOME_COLORS["timeout"],            label="Timeout"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=12, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Flight Path Comparison (Obstacle Avoidance)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Genesis screenshot
# ---------------------------------------------------------------------------

def _add_camera(scene, pos, lookat, res=(1920, 1080), fov=60, far=100.0):
    """Add a camera to an already-built scene."""
    denoise = sys.platform != "darwin"
    cam = scene._visualizer.add_camera(
        res, pos, lookat,
        (0.0, 0.0, 1.0), "pinhole", fov, 2.0, None,
        False, 256, denoise, 0.1, far, None, False,
    )
    cam.build()
    return cam


def _subsample(positions, max_segments=200):
    n = len(positions)
    if n <= max_segments:
        return positions
    step = max(1, n // max_segments)
    sampled = positions[::step]
    if not np.array_equal(sampled[-1], positions[-1]):
        sampled = np.vstack([sampled, positions[-1:]])
    return sampled


def _build_path_mesh(positions, radius=0.008, color=(1.0, 1.0, 1.0, 1.0)):
    import trimesh
    from genesis.utils.mesh import create_line as gs_create_line

    segments = []
    for i in range(len(positions) - 1):
        seg = gs_create_line(
            positions[i].astype(np.float32),
            positions[i + 1].astype(np.float32),
            radius, color,
        )
        segments.append(seg)
    if not segments:
        return None
    return trimesh.util.concatenate(segments)


def run_screenshot(env_cfg, obs_cfg, reward_cfg, train_cfg,
                   log_dir, ckpt, seed=42):
    """Build 1-env scene, fly one episode, capture screenshot with obstacles visible."""
    cfg = copy.deepcopy(env_cfg)
    cfg["visualize_target"] = True
    cfg["curriculum_steps"] = 0

    render_env = ObstacleAvoidanceEnv(
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
    start_pos = render_env.base_pos[0].cpu().numpy().copy()
    start_quat = render_env.base_quat[0].clone()
    target = render_env.target_pos[0].cpu().numpy().copy()
    max_steps = render_env.max_episode_length

    positions = [start_pos.copy()]
    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = render_env.step(actions)
            if dones[0]:
                break
            positions.append(render_env.base_pos[0].cpu().numpy().copy())

    positions = np.array(positions)

    ctx = render_env.scene.visualizer.context
    sampled = _subsample(positions)
    mesh = _build_path_mesh(sampled, radius=0.015, color=(0.0, 1.0, 0.0, 0.8))
    if mesh is not None:
        ctx.draw_debug_mesh(mesh)

    all_points = np.vstack([positions, target.reshape(1, 3)])
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = np.linalg.norm(bbox_max - bbox_min)

    fov = 90
    padding = 1.4
    dist = (extent * padding) / (2.0 * math.tan(math.radians(fov / 2.0)))
    dist = max(dist, 2.0)

    cam_pos = (center[0] + dist * 0.15, center[1] - dist * 0.75, center[2] + dist * 0.55)
    lookat = tuple(center)

    cam = _add_camera(render_env.scene, pos=cam_pos, lookat=lookat,
                      fov=fov, far=max(dist * 3, 100.0))

    render_env.drone.set_pos(
        torch.tensor([start_pos], device=gs.device, dtype=torch.float32),
        zero_velocity=True,
    )
    render_env.drone.set_quat(start_quat.unsqueeze(0), zero_velocity=True)

    render_env.scene.step()
    rgb, _, _, _ = cam.render()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()

    out_path = os.path.join(log_dir, f"screenshot_ckpt_{ckpt}.png")
    Image.fromarray(rgb).save(out_path)
    print(f"  Screenshot saved -> {out_path}")
    ctx.clear_debug_objects()


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------

def record_video(env_cfg, obs_cfg, reward_cfg, train_cfg,
                 log_dir, ckpt, seed=42, every_n=3, fps=30):
    """Build 1-env scene, fly with progressive trail, save as GIF."""
    cfg = copy.deepcopy(env_cfg)
    cfg["visualize_target"] = True
    cfg["curriculum_steps"] = 0

    render_env = ObstacleAvoidanceEnv(
        num_envs=1, env_cfg=cfg, obs_cfg=obs_cfg,
        reward_cfg=copy.deepcopy(reward_cfg), show_viewer=False,
    )
    render_env.build()

    runner = OnPolicyRunner(render_env, copy.deepcopy(train_cfg), log_dir,
                            device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    max_steps = render_env.max_episode_length

    # Pass 1: dry run for camera framing
    torch.manual_seed(seed)
    obs = render_env.reset()
    start_pos = render_env.base_pos[0].cpu().numpy().copy()
    target = render_env.target_pos[0].cpu().numpy().copy()

    positions = [start_pos.copy()]
    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = render_env.step(actions)
            if dones[0]:
                break
            positions.append(render_env.base_pos[0].cpu().numpy().copy())

    positions = np.array(positions)

    all_points = np.vstack([positions, target.reshape(1, 3)])
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = np.linalg.norm(bbox_max - bbox_min)

    fov = 90
    padding = 1.4
    dist = (extent * padding) / (2.0 * math.tan(math.radians(fov / 2.0)))
    dist = max(dist, 2.0)

    cam_pos = (center[0] + dist * 0.15, center[1] - dist * 0.75, center[2] + dist * 0.55)
    lookat = tuple(center)

    cam = _add_camera(render_env.scene, pos=cam_pos, lookat=lookat,
                      res=(960, 540), fov=fov, far=max(dist * 3, 100.0))

    # Pass 2: replay with trail
    torch.manual_seed(seed)
    obs = render_env.reset()
    ctx = render_env.scene.visualizer.context

    frames = []
    trail_positions = [render_env.base_pos[0].cpu().numpy().copy()]
    last_drawn_idx = 0

    with torch.no_grad():
        for step_i in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = render_env.step(actions)

            if not dones[0]:
                trail_positions.append(render_env.base_pos[0].cpu().numpy().copy())

            if step_i % every_n == 0 or dones[0]:
                n = len(trail_positions)
                if n > last_drawn_idx + 1:
                    new_pts = np.array(trail_positions[last_drawn_idx:n])
                    mesh = _build_path_mesh(new_pts, radius=0.015, color=(0.0, 1.0, 0.0, 0.8))
                    if mesh is not None:
                        ctx.draw_debug_mesh(mesh)
                    last_drawn_idx = n - 1

                rgb, _, _, _ = cam.render()
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                frames.append(Image.fromarray(rgb))

            if dones[0]:
                break

    ctx.clear_debug_objects()

    if frames:
        out_path = os.path.join(log_dir, f"landing_ckpt_{ckpt}.gif")
        frame_duration_ms = max(20, int(every_n * render_env.dt * 1000))
        frames[0].save(
            out_path, save_all=True, append_images=frames[1:],
            duration=frame_duration_ms, loop=0,
        )
        print(f"  Video saved -> {out_path}  "
              f"({len(frames)} frames, {frame_duration_ms}ms/frame)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize obstacle avoidance flight paths")
    parser.add_argument("-e", "--exp_name", type=str, default="obstacle-avoidance")
    parser.add_argument("--ckpt", type=int, nargs="+", required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--fixed_spawn", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    reward_cfg["reward_scales"] = {}
    env_cfg["curriculum_steps"] = 0
    env_cfg["visualize_target"] = False

    if args.fixed_spawn:
        env_cfg["spawn_offset"] = 0.0

    obstacle_size = env_cfg.get("obstacle_size", [1.0, 1.0, 2.0])

    target_pos = np.array([
        np.mean(env_cfg["target_x_range"]),
        np.mean(env_cfg["target_y_range"]),
        np.mean(env_cfg["target_z_range"]),
    ])

    for ckpt in args.ckpt:
        path = os.path.join(log_dir, f"model_{ckpt}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Force Rasterizer: visualization adds debug cameras post-build which
    # conflicts with BatchRenderer's "all cameras same resolution" constraint
    env_cfg["use_batch_renderer"] = False

    # Phase 1: collect episodes
    env = ObstacleAvoidanceEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, show_viewer=False,
    )
    env.build()

    all_results = []
    ckpt_labels = []

    for ckpt in args.ckpt:
        resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
        runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir,
                                device=gs.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=gs.device)

        print(f"Collecting {args.num_episodes} episodes for checkpoint {ckpt} ...")
        results = collect_paths(env, policy, args.num_episodes, seed=args.seed)

        succ = sum(r["success"] for r in results)
        crash = sum(r["crash"] for r in results)
        obs_coll = sum(r.get("obstacle_collision", False) for r in results)
        timeout = len(results) - succ - crash - obs_coll
        print(f"  -> S:{succ}  C:{crash}  O:{obs_coll}  T:{timeout}")

        all_results.append(results)
        ckpt_labels.append(f"Checkpoint {ckpt}")

    out_path = os.path.join(log_dir, "paths_comparison.png")
    plot_comparison(all_results, ckpt_labels, target_pos, obstacle_size, out_path)

    # Phase 2: screenshots
    if not args.no_render:
        for ckpt in args.ckpt:
            print(f"Rendering screenshot for checkpoint {ckpt} ...")
            run_screenshot(env_cfg, obs_cfg, reward_cfg, train_cfg,
                           log_dir, ckpt, seed=args.seed)

    # Phase 3: videos
    if args.video:
        for ckpt in args.ckpt:
            print(f"Recording landing video for checkpoint {ckpt} ...")
            record_video(env_cfg, obs_cfg, reward_cfg, train_cfg,
                         log_dir, ckpt, seed=args.seed)


if __name__ == "__main__":
    main()
