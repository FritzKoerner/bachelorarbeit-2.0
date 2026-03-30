import argparse
import copy
import glob
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from envs.coordinate_landing_env import CoordinateLandingEnv
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
    # Parse iteration number from filename
    def iter_num(path):
        m = re.search(r"model_(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1
    best = max(files, key=iter_num)
    return best, iter_num(best)


def resolve_hpc_log_dir(run_name):
    """Resolve --hpc run name to log_dir path, or list available runs and exit."""
    proto_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    hpc_base = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "hpc_results", proto_dir
    ))
    if run_name == "list":
        if not os.path.isdir(hpc_base):
            print(f"No HPC results directory: {hpc_base}")
            sys.exit(1)
        runs = sorted(d for d in os.listdir(hpc_base)
                      if os.path.isdir(os.path.join(hpc_base, d)))
        if not runs:
            print(f"No runs in {hpc_base}")
            sys.exit(1)
        print(f"\nAvailable HPC runs ({hpc_base}):")
        for r in runs:
            ckpts = glob.glob(os.path.join(hpc_base, r, "model_*.pt"))
            iters = sorted(int(re.search(r"model_(\d+)", c).group(1))
                           for c in ckpts if re.search(r"model_(\d+)", c))
            print(f"  {r:30s}  checkpoints: {iters}")
        sys.exit(0)
    run_dir = os.path.join(hpc_base, run_name)
    if not os.path.isdir(run_dir):
        print(f"HPC run not found: {run_dir}")
        if os.path.isdir(hpc_base):
            runs = [d for d in os.listdir(hpc_base)
                    if os.path.isdir(os.path.join(hpc_base, d))]
            if runs:
                print(f"Available: {', '.join(sorted(runs))}")
        sys.exit(1)
    return run_dir


# ---------------------------------------------------------------------------
# Episode collection  (identical to eval_rl.py)
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
                success = bool(env.success_condition[i].item())
                crash   = bool(env.crash_condition[i].item())
                results.append({
                    "success":  success,
                    "crash":    crash,
                    "timeout":  not success and not crash,
                    "length":   int(ep_step[i].item()),
                    "min_dist": ep_min_dist[i],
                    "reward":   float(ep_reward[i].item()),
                    "dists":    ep_dists[i].copy(),
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
    n         = len(results)
    successes = sum(r["success"] for r in results)
    crashes   = sum(r["crash"]   for r in results)
    timeouts  = sum(r["timeout"] for r in results)
    min_dists = [r["min_dist"] for r in results]
    lengths   = [r["length"]   for r in results]
    rewards   = [r["reward"]   for r in results]

    stats = {
        "num_episodes":        n,
        "success_rate":        successes / n,
        "crash_rate":          crashes / n,
        "timeout_rate":        timeouts / n,
        "min_dist_mean":       float(np.mean(min_dists)),
        "min_dist_std":        float(np.std(min_dists)),
        "min_dist_median":     float(np.median(min_dists)),
        "min_dist_best":       float(min(min_dists)),
        "episode_length_mean": float(np.mean(lengths)),
        "episode_length_std":  float(np.std(lengths)),
        "reward_mean":         float(np.mean(rewards)),
        "reward_std":          float(np.std(rewards)),
    }

    print(f"\n{'═'*48}")
    print(f"  Evaluation — {n} episodes")
    print(f"{'═'*48}")
    print(f"  Success : {successes:3d} / {n}  ({100*stats['success_rate']:5.1f} %)")
    print(f"  Crash   : {crashes:3d} / {n}  ({100*stats['crash_rate']:5.1f} %)")
    print(f"  Timeout : {timeouts:3d} / {n}  ({100*stats['timeout_rate']:5.1f} %)")
    print(f"{'─'*48}")
    print(f"  Closest approach  mean ± std : {stats['min_dist_mean']:.3f} ± {stats['min_dist_std']:.3f} m")
    print(f"  Closest approach  median     : {stats['min_dist_median']:.3f} m")
    print(f"  Best single approach         : {stats['min_dist_best']:.3f} m")
    print(f"  Episode length  mean ± std   : {stats['episode_length_mean']:.0f} ± {stats['episode_length_std']:.0f} steps")
    print(f"  Reward  mean ± std           : {stats['reward_mean']:.1f} ± {stats['reward_std']:.1f}")
    print(f"{'═'*48}\n")

    return stats


# ---------------------------------------------------------------------------
# Plots (saved locally + uploaded to W&B)
# ---------------------------------------------------------------------------

def make_plots(results: list[dict]) -> plt.Figure:
    """Create evaluation figure and return it (for W&B upload)."""
    n         = len(results)
    successes = sum(r["success"] for r in results)
    crashes   = sum(r["crash"]   for r in results)
    timeouts  = sum(r["timeout"] for r in results)
    min_dists = [r["min_dist"] for r in results]
    lengths   = [r["length"]   for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Evaluation — {n} episodes", fontsize=14, fontweight="bold")

    # 1. Outcome bar chart
    ax = axes[0, 0]
    bars = ax.bar(
        ["Success", "Crash", "Timeout"],
        [successes, crashes, timeouts],
        color=["#2ecc71", "#e74c3c", "#f39c12"],
        edgecolor="white", width=0.5,
    )
    ax.set_ylabel("Count")
    ax.set_title("Episode Outcomes")
    ax.set_ylim(0, n * 1.15)
    for bar, v in zip(bars, [successes, crashes, timeouts]):
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

    # 4. Distance over time (mean ± std)
    ax = axes[1, 1]
    all_dists = [r["dists"] for r in results]
    max_len   = max(len(d) for d in all_dists)
    arr = np.full((n, max_len), np.nan)
    for i, d in enumerate(all_dists):
        arr[i, : len(d)] = d
    mean_d = np.nanmean(arr, axis=0)
    std_d  = np.nanstd(arr,  axis=0)
    steps  = np.arange(max_len)
    ax.plot(steps, mean_d, color="#e67e22", linewidth=1.5, label="mean")
    ax.fill_between(steps, mean_d - std_d, mean_d + std_d,
                    alpha=0.2, color="#e67e22", label="±1 std")
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
    table = wandb.Table(columns=["episode", "outcome", "length", "min_dist", "reward"])
    for i, r in enumerate(results):
        outcome = "success" if r["success"] else ("crash" if r["crash"] else "timeout")
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
    parser.add_argument("-e", "--exp_name",   type=str, default="drone-landing")
    parser.add_argument("--hpc",              nargs="?", const="list", default=None,
                        help="HPC run name (e.g. 'new_version'). No value = list runs.")
    parser.add_argument("--log_dir",          type=str, default=None,
                        help="Direct path to log dir (overrides --exp_name)")
    parser.add_argument("--ckpt",             type=int, default=None,
                        help="Checkpoint iteration (default: latest)")
    parser.add_argument("--num_envs",         type=int, default=50)
    parser.add_argument("--num_episodes",     type=int, default=100)
    parser.add_argument("--wandb_project",    type=str, default="drone-landing")
    parser.add_argument("--vis", action="store_true",
                        help="Viewer mode: 1 env, 10 episodes, no W&B logging")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

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
        ckpt_iter = args.ckpt
    else:
        resume_path, ckpt_iter = find_latest_checkpoint(log_dir)
    print(f"Loading checkpoint: {resume_path}  (iteration {ckpt_iter})")

    # Eval-time overrides
    reward_cfg["reward_scales"] = {}
    env_cfg["curriculum_steps"] = 0

    if args.vis:
        env_cfg["visualize_target"] = True
        num_envs    = 1
        show_viewer = True
    else:
        env_cfg["visualize_target"] = False
        num_envs    = args.num_envs
        show_viewer = False

    env = CoordinateLandingEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=show_viewer,
    )
    env.build()

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    if args.vis:
        # Interactive viewer: run 10 episodes sequentially
        num_vis_episodes = 10
        max_steps = int(env_cfg["episode_length_s"] * 100)
        print(f"Viewer mode: running {num_vis_episodes} episodes (checkpoint {ckpt_iter})")
        obs = env.reset()
        episode = 0
        step = 0
        with torch.no_grad():
            while episode < num_vis_episodes:
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                step += 1
                if dones.any() or step >= max_steps:
                    episode += 1
                    outcome = ("success" if env.success_condition[0].item()
                               else "crash" if env.crash_condition[0].item()
                               else "timeout")
                    print(f"  Episode {episode}/{num_vis_episodes}: {outcome} ({step} steps)")
                    obs = env.reset()
                    step = 0
        print("Done.")
    else:
        # Collect episodes
        print(f"Collecting {args.num_episodes} episodes across {num_envs} parallel envs ...")
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

# List available HPC runs
python eval_rl_wb.py --hpc

# Evaluate HPC run (latest checkpoint)
python eval_rl_wb.py --hpc new_version

# Evaluate HPC run (specific checkpoint)
python eval_rl_wb.py --hpc new_version --ckpt 400

# Evaluate specific checkpoint
python eval_rl_wb.py --ckpt 300

# Custom episode count
python eval_rl_wb.py --num_episodes 200 --num_envs 100

# Visual viewer (1 env, 10 episodes, no W&B)
python eval_rl_wb.py --vis
python eval_rl_wb.py --ckpt 300 --vis
"""
