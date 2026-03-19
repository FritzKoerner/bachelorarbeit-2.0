#!/usr/bin/env python3
"""Visualize drone flight paths across training checkpoints.

Runs N episodes per checkpoint for statistics, and a separate 1-env
simulation per checkpoint for a Genesis-rendered screenshot.

Outputs per run:
  1. Matplotlib comparison plot   (paths_comparison.png)
  2. Genesis-rendered screenshot   (screenshot_ckpt_<N>.png)  per checkpoint
     — shows drone at start, target sphere, and the flight path trail
  3. Landing GIF (--video)         (landing_ckpt_<N>.gif)     per checkpoint
     — animated recording with progressive trail drawn during flight

Usage:
    python visualize_paths.py --ckpt 100 300
    python visualize_paths.py --ckpt 0 100 200 300 400
    python visualize_paths.py --ckpt 100 300 --num_episodes 50 --num_envs 20
    python visualize_paths.py --ckpt 100 300 --fixed_spawn
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
import numpy as np
import torch
from PIL import Image

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from envs.coordinate_landing_env import CoordinateLandingEnv


# ---------------------------------------------------------------------------
# Data collection (N parallel envs, fast)
# ---------------------------------------------------------------------------

def collect_paths(env, policy, num_episodes, seed=42):
    """Run episodes and collect 3D position trajectories.

    Positions are recorded BEFORE each step() call so that auto-reset
    never contaminates a trajectory with the next episode's spawn position.
    """
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
                success = bool(env.success_condition[i].item())
                crash = bool(env.crash_condition[i].item())
                results.append({
                    "positions": np.array(ep_positions[i]),
                    "success": success,
                    "crash": crash,
                    "timeout": not success and not crash,
                })
                ep_positions[i] = []

    return results[:num_episodes]


# ---------------------------------------------------------------------------
# Matplotlib plotting
# ---------------------------------------------------------------------------

OUTCOME_COLORS = {
    "success": "#60aa64",
    "crash":   "#c44a40",
    "timeout": "#e8a340",
}

OUTCOME_COLORS_RGBA = {
    "success": (0.376, 0.667, 0.392, 0.45),
    "crash":   (0.769, 0.290, 0.251, 0.40),
    "timeout": (0.910, 0.639, 0.251, 0.40),
}


def _outcome_color(r):
    if r["success"]:
        return OUTCOME_COLORS["success"]
    if r["crash"]:
        return OUTCOME_COLORS["crash"]
    return OUTCOME_COLORS["timeout"]


def _outcome_rgba(r):
    if r["success"]:
        return OUTCOME_COLORS_RGBA["success"]
    if r["crash"]:
        return OUTCOME_COLORS_RGBA["crash"]
    return OUTCOME_COLORS_RGBA["timeout"]


def plot_comparison(all_results, ckpt_labels, target_pos, out_path):
    """3-row comparison figure: 3D / top-down XY / side XZ, one column per ckpt."""
    n_ckpts = len(all_results)
    fig = plt.figure(figsize=(6 * n_ckpts, 16))

    for ci, (results, label) in enumerate(zip(all_results, ckpt_labels)):
        n_ep = len(results)
        succ = sum(r["success"] for r in results)
        crash = sum(r["crash"] for r in results)
        timeout = n_ep - succ - crash

        ax = fig.add_subplot(3, n_ckpts, ci + 1, projection="3d")
        for r in results:
            pos = r["positions"]
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                    c=_outcome_color(r), alpha=0.35, linewidth=0.6)
        ax.scatter(*target_pos, c="red", s=30, marker="o", zorder=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{label}\nS:{succ}  C:{crash}  T:{timeout}  (n={n_ep})",
                      fontsize=11, fontweight="bold")

        ax = fig.add_subplot(3, n_ckpts, n_ckpts + ci + 1)
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

        ax = fig.add_subplot(3, n_ckpts, 2 * n_ckpts + ci + 1)
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
        mpatches.Patch(color=OUTCOME_COLORS["success"], label="Success"),
        mpatches.Patch(color=OUTCOME_COLORS["crash"],   label="Crash"),
        mpatches.Patch(color=OUTCOME_COLORS["timeout"], label="Timeout"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=12, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Flight Path Comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Genesis screenshot (dedicated 1-env simulation)
# ---------------------------------------------------------------------------

def _add_camera(scene, pos, lookat, res=(1920, 1080), fov=60, far=100.0):
    """Add a camera to an already-built scene.

    scene.add_camera() is blocked post-build by @assert_not_built, but the
    underlying visualizer.add_camera() + camera.build() work fine because
    they only add a pyrender camera node — a purely dynamic operation.
    """
    denoise = sys.platform != "darwin"
    cam = scene._visualizer.add_camera(
        res, pos, lookat,
        (0.0, 0.0, 1.0),   # up
        "pinhole",          # model
        fov,
        2.0,                # aperture (ignored for pinhole)
        None,               # focus_dist (auto-computed)
        False,              # GUI
        256,                # spp (rasterizer ignores this)
        denoise,
        0.1,                # near
        far,
        None,               # env_idx (defaults to rendered_envs_idx[0])
        False,              # debug
    )
    cam.build()
    return cam


def _subsample(positions, max_segments=200):
    """Reduce a path to at most max_segments points for debug drawing."""
    n = len(positions)
    if n <= max_segments:
        return positions
    step = max(1, n // max_segments)
    sampled = positions[::step]
    if not np.array_equal(sampled[-1], positions[-1]):
        sampled = np.vstack([sampled, positions[-1:]])
    return sampled


def _build_path_mesh(positions, radius=0.008, color=(1.0, 1.0, 1.0, 1.0)):
    """Build a single trimesh for an entire flight path (one node, no UID issues)."""
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
    """Build a dedicated 1-env scene, fly one episode, and capture a screenshot.

    Steps:
      1. Build 1-env CoordinateLandingEnv with target sphere visible
      2. Add a camera post-build
      3. Run one episode with the policy, collecting positions
      4. Draw the flight path as a debug mesh
      5. Reset drone to its starting position
      6. Render + save screenshot
    """
    # Build a fresh 1-env scene with the target sphere visible
    cfg = copy.deepcopy(env_cfg)
    cfg["visualize_target"] = True
    cfg["curriculum_steps"] = 0

    render_env = CoordinateLandingEnv(
        num_envs=1,
        env_cfg=cfg,
        obs_cfg=obs_cfg,
        reward_cfg=copy.deepcopy(reward_cfg),
        show_viewer=False,
    )
    render_env.build()

    # Load policy
    runner = OnPolicyRunner(render_env, copy.deepcopy(train_cfg), log_dir,
                            device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # Run one episode
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

            pos = render_env.base_pos[0].cpu().numpy()
            positions.append(pos.copy())

    positions = np.array(positions)

    # Draw the flight path
    ctx = render_env.scene.visualizer.context
    sampled = _subsample(positions)
    mesh = _build_path_mesh(sampled, radius=0.015, color=(0.0, 1.0, 0.0, 0.8))
    if mesh is not None:
        ctx.draw_debug_mesh(mesh)

    # Compute auto-framing camera from path bounding box
    all_points = np.vstack([positions, target.reshape(1, 3)])
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = np.linalg.norm(bbox_max - bbox_min)

    fov = 90
    padding = 1.4  # extra margin so path isn't clipped at edges
    dist = (extent * padding) / (2.0 * math.tan(math.radians(fov / 2.0)))
    dist = max(dist, 2.0)  # minimum distance to avoid being inside the scene

    # Place camera at a 3/4 view: offset in -Y and +Z from center
    cam_pos = (
        center[0] + dist * 0.15,
        center[1] - dist * 0.75,
        center[2] + dist * 0.55,
    )
    lookat = tuple(center)

    cam = _add_camera(
        render_env.scene,
        pos=cam_pos,
        lookat=lookat,
        fov=fov,
        far=max(dist * 3, 100.0),
    )

    # Reset drone to starting position so it's visible at the origin
    render_env.drone.set_pos(
        torch.tensor([start_pos], device=gs.device, dtype=torch.float32),
        zero_velocity=True,
    )
    render_env.drone.set_quat(
        start_quat.unsqueeze(0),
        zero_velocity=True,
    )

    # Step once so the visual state updates, then render
    render_env.scene.step()
    rgb, _, _, _ = cam.render()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()

    out_path = os.path.join(log_dir, f"screenshot_ckpt_{ckpt}.png")
    Image.fromarray(rgb).save(out_path)
    print(f"  Screenshot saved -> {out_path}")

    ctx.clear_debug_objects()


# ---------------------------------------------------------------------------
# Video recording (dedicated 1-env simulation, two-pass)
# ---------------------------------------------------------------------------

def record_video(env_cfg, obs_cfg, reward_cfg, train_cfg,
                 log_dir, ckpt, seed=42, every_n=3, fps=30):
    """Build a dedicated 1-env scene, fly one episode with progressive trail, save as GIF.

    Two-pass approach:
      Pass 1 — dry run to collect positions for camera auto-framing
      Pass 2 — replay with same seed, progressively draw trail, capture frames
    """
    cfg = copy.deepcopy(env_cfg)
    cfg["visualize_target"] = True
    cfg["curriculum_steps"] = 0

    render_env = CoordinateLandingEnv(
        num_envs=1,
        env_cfg=cfg,
        obs_cfg=obs_cfg,
        reward_cfg=copy.deepcopy(reward_cfg),
        show_viewer=False,
    )
    render_env.build()

    # Load policy
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
    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = render_env.step(actions)
            if dones[0]:
                break
            positions.append(render_env.base_pos[0].cpu().numpy().copy())

    positions = np.array(positions)

    # Compute camera framing from full path
    all_points = np.vstack([positions, target.reshape(1, 3)])
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

    cam = _add_camera(
        render_env.scene,
        pos=cam_pos,
        lookat=lookat,
        res=(960, 540),
        fov=fov,
        far=max(dist * 3, 100.0),
    )

    # --- Pass 2: replay with progressive trail + frame capture ---
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
                trail_positions.append(
                    render_env.base_pos[0].cpu().numpy().copy()
                )

            # Capture frame every N steps or on episode end
            if step_i % every_n == 0 or dones[0]:
                n = len(trail_positions)
                if n > last_drawn_idx + 1:
                    new_pts = np.array(trail_positions[last_drawn_idx:n])
                    mesh = _build_path_mesh(
                        new_pts, radius=0.015, color=(0.0, 1.0, 0.0, 0.8),
                    )
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

    # Save as GIF (PIL fallback — moviepy/ffmpeg not available)
    if frames:
        out_path = os.path.join(log_dir, f"landing_ckpt_{ckpt}.gif")
        # real-time: each frame spans every_n * dt seconds of sim time
        frame_duration_ms = max(20, int(every_n * render_env.dt * 1000))
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        print(f"  Video saved -> {out_path}  "
              f"({len(frames)} frames, {frame_duration_ms}ms/frame)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize drone flight paths across training checkpoints")
    parser.add_argument("-e", "--exp_name", type=str, default="drone-landing")
    parser.add_argument("--ckpt", type=int, nargs="+", required=True,
                        help="Checkpoint numbers to compare, e.g. --ckpt 100 300")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=50,
                        help="Parallel envs for faster collection")
    parser.add_argument("--fixed_spawn", action="store_true",
                        help="Disable spawn randomisation (all episodes from same start)")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip Genesis rendering (matplotlib only)")
    parser.add_argument("--video", action="store_true",
                        help="Record landing GIF with progressive trail per checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Eval overrides
    reward_cfg["reward_scales"] = {}
    env_cfg["curriculum_steps"] = 0
    env_cfg["visualize_target"] = False

    if args.fixed_spawn:
        env_cfg["spawn_offset"] = 0.0

    target_pos = np.array([
        np.mean(env_cfg["target_x_range"]),
        np.mean(env_cfg["target_y_range"]),
        np.mean(env_cfg["target_z_range"]),
    ])

    # Verify all requested checkpoints exist
    for ckpt in args.ckpt:
        path = os.path.join(log_dir, f"model_{ckpt}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    # --- Phase 1: collect episodes with N parallel envs ---
    env = CoordinateLandingEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
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
        timeout = len(results) - succ - crash
        print(f"  -> S:{succ}  C:{crash}  T:{timeout}")

        all_results.append(results)
        ckpt_labels.append(f"Checkpoint {ckpt}")

    # Matplotlib comparison plot
    out_path = os.path.join(log_dir, "paths_comparison.png")
    plot_comparison(all_results, ckpt_labels, target_pos, out_path)

    # --- Phase 2: Genesis screenshots (separate 1-env sim per checkpoint) ---
    if not args.no_render:
        for ckpt in args.ckpt:
            print(f"Rendering screenshot for checkpoint {ckpt} ...")
            run_screenshot(env_cfg, obs_cfg, reward_cfg, train_cfg,
                           log_dir, ckpt, seed=args.seed)

    # --- Phase 3: Landing video (separate 1-env sim per checkpoint) ---
    if args.video:
        for ckpt in args.ckpt:
            print(f"Recording landing video for checkpoint {ckpt} ...")
            record_video(env_cfg, obs_cfg, reward_cfg, train_cfg,
                         log_dir, ckpt, seed=args.seed)


if __name__ == "__main__":
    main()
