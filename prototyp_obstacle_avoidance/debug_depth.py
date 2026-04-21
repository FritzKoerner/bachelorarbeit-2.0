"""Verify depth camera and CNN pipeline.

Runs 2 envs for a few steps, then:
  1. Saves per-env depth heatmaps as PNGs
  2. Prints depth tensor statistics
  3. Feeds the TensorDict observation through the CNN actor and prints shapes/stats
"""

import copy
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from envs.obstacle_avoidance_env import ObstacleAvoidanceEnv
from train_rl_wb import get_cfgs, get_train_cfg


def save_depth_heatmaps(depth_buf, obstacle_positions, base_pos, target_pos, step, out_dir):
    """Save per-env depth images as annotated heatmaps."""
    n_envs = depth_buf.shape[0]
    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        d = depth_buf[i, -1].cpu().numpy()  # (H, W), most recent frame, normalised [0,1]
        im = ax.imshow(d, cmap="viridis", vmin=0, vmax=1, origin="upper")
        ax.set_title(
            f"Env {i}  |  min={d.min():.3f}  max={d.max():.3f}  mean={d.mean():.3f}\n"
            f"drone z={base_pos[i, 2].item():.2f}m  "
            f"target=({target_pos[i, 0].item():.1f},{target_pos[i, 1].item():.1f},{target_pos[i, 2].item():.1f})"
        )
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="depth / max_depth")

    fig.suptitle(f"Depth observations — step {step}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, f"depth_step_{step:04d}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Debug depth camera pipeline")
    parser.add_argument("--renderer", choices=["auto", "batch", "rasterizer"],
                        default="auto", help="Renderer backend (default: auto)")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    out_dir = "debug_depth_output"
    os.makedirs(out_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    # Force obstacles on immediately (skip curriculum)
    env_cfg["curriculum_steps"] = 0
    env_cfg["visualize_target"] = False
    # Renderer selection from CLI
    renderer_map = {"auto": "auto", "batch": True, "rasterizer": False}
    env_cfg["use_batch_renderer"] = renderer_map[args.renderer]

    n_envs = 2
    env = ObstacleAvoidanceEnv(n_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False)
    env.build()

    # ── 1. Depth images after reset ──────────────────────────────────────
    print("\n═══ Step 0 (after reset) ═══")
    obs = env.get_observations()
    depth_buf = obs["depth"]  # (n, 1, 64, 64)
    print(f"  depth_buf shape : {depth_buf.shape}")
    print(f"  depth_buf dtype : {depth_buf.dtype}")
    print(f"  depth_buf device: {depth_buf.device}")
    print(f"  min={depth_buf.min().item():.4f}  max={depth_buf.max().item():.4f}  "
          f"mean={depth_buf.mean().item():.4f}  std={depth_buf.std().item():.4f}")
    print(f"  fraction > 0   : {(depth_buf > 0).float().mean().item():.2%}")
    print(f"  fraction < 1   : {(depth_buf < 1).float().mean().item():.2%}  (pixels closer than max_depth)")

    save_depth_heatmaps(depth_buf, env.obstacle_positions, env.base_pos, env.target_pos, 0, out_dir)

    # Print obstacle layout for context
    print(f"\n  Obstacle positions (env 0):")
    for j in range(env.num_obstacles):
        p = env.obstacle_positions[0, j]
        print(f"    obs[{j}]: ({p[0].item():.2f}, {p[1].item():.2f}, {p[2].item():.2f})")

    # ── 2. Run a few steps and capture again ─────────────────────────────
    print("\n═══ Running 10 steps with random actions ═══")
    for step in range(1, 11):
        actions = torch.randn(n_envs, env.num_actions, device=gs.device) * 0.3
        obs, rew, done, extras = env.step(actions)

    depth_buf = obs["depth"]
    print(f"  Step 10 depth — min={depth_buf.min().item():.4f}  max={depth_buf.max().item():.4f}  "
          f"mean={depth_buf.mean().item():.4f}")
    save_depth_heatmaps(depth_buf, env.obstacle_positions, env.base_pos, env.target_pos, 10, out_dir)

    # ── 3. Verify depth variation across envs ────────────────────────────
    print("\n═══ Per-env depth comparison ═══")
    for i in range(n_envs):
        d = depth_buf[i, -1]  # most recent frame
        unique_vals = d.unique().numel()
        print(f"  Env {i}: unique pixel values = {unique_vals:>5}, "
              f"min={d.min().item():.4f}, max={d.max().item():.4f}")

    envs_differ = not torch.allclose(depth_buf[0], depth_buf[1], atol=1e-3)
    print(f"  Env 0 ≠ Env 1  : {envs_differ}  (should be True — different obstacle layouts)")

    # ── 4. CNN forward pass ──────────────────────────────────────────────
    print("\n═══ CNN forward pass verification ═══")
    train_cfg = get_train_cfg("debug", 5)
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), out_dir, device=gs.device)
    actor = runner.get_inference_policy()

    print(f"  Actor model type: {type(actor).__name__}")
    print(f"  Obs keys        : {list(obs.keys())}")
    print(f"  state shape     : {obs['state'].shape}")
    print(f"  depth shape     : {obs['depth'].shape}")

    with torch.no_grad():
        actor_out = actor(obs)

    print(f"  Actor output    : {actor_out.shape}")
    print(f"  Action min={actor_out.min().item():.4f}  max={actor_out.max().item():.4f}  "
          f"mean={actor_out.mean().item():.4f}")

    actions_vary = actor_out.std(dim=0).mean().item() > 1e-6
    print(f"  Actions vary across envs: {actions_vary}  (should be True)")

    # ── 5. Gradient flow check ───────────────────────────────────────────
    print("\n═══ Gradient flow through CNN ═══")
    obs_grad = obs.clone()
    obs_grad["depth"] = obs_grad["depth"].requires_grad_(True)
    out = actor(obs_grad)
    loss = out.sum()
    loss.backward()
    grad = obs_grad["depth"].grad
    print(f"  d(loss)/d(depth) shape: {grad.shape}")
    print(f"  grad nonzero fraction : {(grad.abs() > 1e-8).float().mean().item():.2%}")
    print(f"  grad norm             : {grad.norm().item():.6f}")

    has_grad = grad.abs().max().item() > 0
    print(f"  Gradients flow to depth input: {has_grad}  (should be True)")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Depth heatmaps saved to: {os.path.abspath(out_dir)}/")
    checks = [
        ("Depth has variation (not all same value)", depth_buf.std().item() > 1e-4),
        ("Envs produce different depth images", envs_differ),
        ("CNN produces varying actions", actions_vary),
        ("Gradients flow through depth → CNN → actions", has_grad),
    ]
    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")
    print(f"{'═'*60}")
    if all_pass:
        print("  All checks passed!")
    else:
        print("  Some checks FAILED — inspect the saved heatmaps for details.")


if __name__ == "__main__":
    main()
