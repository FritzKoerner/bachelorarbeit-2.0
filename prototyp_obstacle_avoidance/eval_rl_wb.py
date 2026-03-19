import argparse
import copy
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from envs.obstacle_avoidance_env import ObstacleAvoidanceEnv
from train_rl_wb import DictConfig  # needed to unpickle cfgs.pkl


# ---------------------------------------------------------------------------
# Find latest checkpoint
# ---------------------------------------------------------------------------

def find_latest_checkpoint(log_dir: str) -> tuple[str, int]:
    """Return (path, iteration) for the highest-numbered model_*.pt file."""
    pattern = os.path.join(log_dir, "model_*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    def iter_num(path):
        m = re.search(r"model_(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1
    best = max(files, key=iter_num)
    return best, iter_num(best)


# ---------------------------------------------------------------------------
# Episode collection (with obstacle collision tracking)
# ---------------------------------------------------------------------------

def collect_episodes(env, policy, num_episodes: int) -> list[dict]:
    """Run policy until num_episodes complete. Returns one dict per episode."""
    num_envs = env.num_envs

    ep_step     = torch.zeros(num_envs, dtype=torch.int)
    ep_min_dist = [float("inf")] * num_envs
    ep_reward   = torch.zeros(num_envs, dtype=torch.float, device=gs.device)
    ep_dists    = [[] for _ in range(num_envs)]

    results = []
    obs = env.reset()

    with torch.no_grad():
        while len(results) < num_episodes:
            dist = torch.norm(env.rel_pos, dim=1).cpu().tolist()
            ep_step += 1
            for i in range(num_envs):
                if dist[i] < ep_min_dist[i]:
                    ep_min_dist[i] = dist[i]
                ep_dists[i].append(dist[i])

            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)
            ep_reward += rewards

            for i in dones.nonzero(as_tuple=False).flatten().tolist():
                success    = bool(env.success_condition[i].item())
                crash      = bool(env.crash_condition[i].item())
                obs_collid = bool(env.obstacle_collision[i].item())
                results.append({
                    "success":            success,
                    "crash":              crash and not obs_collid,
                    "obstacle_collision": obs_collid,
                    "timeout":            not success and not crash,
                    "length":             int(ep_step[i].item()),
                    "min_dist":           ep_min_dist[i],
                    "reward":             float(ep_reward[i].item()),
                    "dists":              ep_dists[i].copy(),
                })
                ep_step[i]     = 0
                ep_min_dist[i] = float("inf")
                ep_reward[i]   = 0.0
                ep_dists[i]    = []

                if len(results) >= num_episodes:
                    break

    return results[:num_episodes]


# ---------------------------------------------------------------------------
# Console stats
# ---------------------------------------------------------------------------

def print_stats(results: list[dict]) -> dict:
    """Print and return aggregate stats."""
    n          = len(results)
    successes  = sum(r["success"] for r in results)
    crashes    = sum(r["crash"] for r in results)
    obs_colls  = sum(r["obstacle_collision"] for r in results)
    timeouts   = sum(r["timeout"] for r in results)
    min_dists  = [r["min_dist"] for r in results]
    lengths    = [r["length"] for r in results]
    rewards    = [r["reward"] for r in results]

    stats = {
        "num_episodes":             n,
        "success_rate":             successes / n,
        "crash_rate":               crashes / n,
        "obstacle_collision_rate":  obs_colls / n,
        "timeout_rate":             timeouts / n,
        "min_dist_mean":            float(np.mean(min_dists)),
        "min_dist_std":             float(np.std(min_dists)),
        "min_dist_median":          float(np.median(min_dists)),
        "min_dist_best":            float(min(min_dists)),
        "episode_length_mean":      float(np.mean(lengths)),
        "episode_length_std":       float(np.std(lengths)),
        "reward_mean":              float(np.mean(rewards)),
        "reward_std":               float(np.std(rewards)),
    }

    print(f"\n{'='*52}")
    print(f"  Evaluation -- {n} episodes")
    print(f"{'='*52}")
    print(f"  Success            : {successes:3d} / {n}  ({100*stats['success_rate']:5.1f} %)")
    print(f"  Crash              : {crashes:3d} / {n}  ({100*stats['crash_rate']:5.1f} %)")
    print(f"  Obstacle collision : {obs_colls:3d} / {n}  ({100*stats['obstacle_collision_rate']:5.1f} %)")
    print(f"  Timeout            : {timeouts:3d} / {n}  ({100*stats['timeout_rate']:5.1f} %)")
    print(f"{'─'*52}")
    print(f"  Closest approach  mean +/- std : {stats['min_dist_mean']:.3f} +/- {stats['min_dist_std']:.3f} m")
    print(f"  Closest approach  median       : {stats['min_dist_median']:.3f} m")
    print(f"  Best single approach           : {stats['min_dist_best']:.3f} m")
    print(f"  Episode length  mean +/- std   : {stats['episode_length_mean']:.0f} +/- {stats['episode_length_std']:.0f} steps")
    print(f"  Reward  mean +/- std           : {stats['reward_mean']:.1f} +/- {stats['reward_std']:.1f}")
    print(f"{'='*52}\n")

    return stats


# ---------------------------------------------------------------------------
# Plots (saved locally + uploaded to W&B)
# ---------------------------------------------------------------------------

def make_plots(results: list[dict]) -> plt.Figure:
    """Create evaluation figure with 4-outcome bar chart."""
    n          = len(results)
    successes  = sum(r["success"] for r in results)
    crashes    = sum(r["crash"] for r in results)
    obs_colls  = sum(r["obstacle_collision"] for r in results)
    timeouts   = sum(r["timeout"] for r in results)
    min_dists  = [r["min_dist"] for r in results]
    lengths    = [r["length"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Evaluation -- {n} episodes", fontsize=14, fontweight="bold")

    # 1. Outcome bar chart (4 categories)
    ax = axes[0, 0]
    bars = ax.bar(
        ["Success", "Crash", "Obstacle", "Timeout"],
        [successes, crashes, obs_colls, timeouts],
        color=["#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"],
        edgecolor="white", width=0.5,
    )
    ax.set_ylabel("Count")
    ax.set_title("Episode Outcomes")
    ax.set_ylim(0, n * 1.15)
    for bar, v in zip(bars, [successes, crashes, obs_colls, timeouts]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{100*v/n:.1f}%", ha="center", va="bottom", fontsize=11)

    # 2. Closest approach histogram
    ax = axes[0, 1]
    ax.hist(min_dists, bins=25, color="#3498db", edgecolor="white")
    ax.axvline(np.mean(min_dists), color="red", linestyle="--",
               label=f"mean = {np.mean(min_dists):.2f} m")
    ax.axvline(np.median(min_dists), color="orange", linestyle=":",
               label=f"median = {np.median(min_dists):.2f} m")
    ax.set_xlabel("Min distance to target (m)")
    ax.set_ylabel("Episodes")
    ax.set_title("Closest Approach per Episode")
    ax.legend()

    # 3. Episode length histogram
    ax = axes[1, 0]
    ax.hist(lengths, bins=25, color="#9b59b6", edgecolor="white")
    ax.axvline(np.mean(lengths), color="red", linestyle="--",
               label=f"mean = {np.mean(lengths):.0f} steps")
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Episodes")
    ax.set_title("Episode Length Distribution")
    ax.legend()

    # 4. Distance over time (mean +/- std)
    ax = axes[1, 1]
    all_dists = [r["dists"] for r in results]
    max_len   = max(len(d) for d in all_dists)
    arr = np.full((n, max_len), np.nan)
    for i, d in enumerate(all_dists):
        arr[i, :len(d)] = d
    mean_d = np.nanmean(arr, axis=0)
    std_d  = np.nanstd(arr, axis=0)
    steps  = np.arange(max_len)
    ax.plot(steps, mean_d, color="#e67e22", linewidth=1.5, label="mean")
    ax.fill_between(steps, mean_d - std_d, mean_d + std_d,
                    alpha=0.2, color="#e67e22", label="+/-1 std")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to target (m)")
    ax.set_title("Distance over Episode (all episodes)")
    ax.legend()

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def log_to_wandb(results: list[dict], stats: dict, fig: plt.Figure, ckpt_iter: int) -> None:
    """Log per-episode data, summary stats, and plots to W&B."""
    # Per-episode table
    table = wandb.Table(columns=[
        "episode", "outcome", "length", "min_dist", "reward",
    ])
    for i, r in enumerate(results):
        if r["success"]:
            outcome = "success"
        elif r["obstacle_collision"]:
            outcome = "obstacle_collision"
        elif r["crash"]:
            outcome = "crash"
        else:
            outcome = "timeout"
        table.add_data(i, outcome, r["length"], r["min_dist"], r["reward"])
    wandb.log({"eval/episodes": table})

    # Summary metrics
    for key, val in stats.items():
        wandb.summary[f"eval/{key}"] = val
    wandb.summary["eval/checkpoint"] = ckpt_iter

    # Upload figure
    wandb.log({"eval/stats_plot": wandb.Image(fig)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",   type=str, default="obstacle-avoidance")
    parser.add_argument("--ckpt",             type=int, default=None,
                        help="Checkpoint iteration (default: latest)")
    parser.add_argument("--num_envs",         type=int, default=50)
    parser.add_argument("--num_episodes",     type=int, default=100)
    parser.add_argument("--wandb_project",    type=str, default="obstacle-avoidance")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Find checkpoint
    if args.ckpt is not None:
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        ckpt_iter = args.ckpt
    else:
        resume_path, ckpt_iter = find_latest_checkpoint(log_dir)
    print(f"Loading checkpoint: {resume_path}  (iteration {ckpt_iter})")

    # Eval-time overrides
    reward_cfg["reward_scales"] = {}
    env_cfg["curriculum_steps"] = 0
    env_cfg["visualize_target"] = False

    env = ObstacleAvoidanceEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
    )
    env.build()

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    # Collect episodes
    print(f"Collecting {args.num_episodes} episodes across {args.num_envs} parallel envs ...")
    results = collect_episodes(env, policy, args.num_episodes)

    # Stats + plots
    stats = print_stats(results)
    fig = make_plots(results)

    # Save plot locally
    plot_path = os.path.join(log_dir, "eval_stats.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {plot_path}")

    # Log to W&B
    run_name = f"{args.exp_name}-eval-iter{ckpt_iter}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        job_type="eval",
        config={
            "exp_name": args.exp_name,
            "checkpoint": ckpt_iter,
            "num_episodes": args.num_episodes,
            "env_cfg": dict(env_cfg),
            "obs_cfg": obs_cfg,
        },
    )
    log_to_wandb(results, stats, fig, ckpt_iter)
    wandb.finish()
    print(f"W&B eval run logged: {run_name}")

    plt.close(fig)


if __name__ == "__main__":
    main()

"""
# Evaluate latest checkpoint, log to W&B
python eval_rl_wb.py

# Evaluate specific checkpoint
python eval_rl_wb.py --ckpt 300

# Custom episode count
python eval_rl_wb.py --num_episodes 200 --num_envs 100
"""
