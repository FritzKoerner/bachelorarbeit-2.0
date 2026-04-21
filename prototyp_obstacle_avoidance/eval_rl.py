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

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from envs.obstacle_avoidance_env import ObstacleAvoidanceEnv
from train_rl_wb import DictConfig  # needed to unpickle cfgs.pkl from W&B training
from eval_rl_wb import apply_placement_override


# ---------------------------------------------------------------------------
# HPC run resolution
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
# Episode collection
# ---------------------------------------------------------------------------

def collect_episodes(env, policy, num_episodes: int) -> list[dict]:
    """Run policy until num_episodes complete. Returns one dict per episode."""
    num_envs = env.num_envs

    ep_step     = torch.zeros(num_envs, dtype=torch.int)
    ep_min_dist = [float("inf")] * num_envs
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
            for i in range(num_envs):
                ep_dists[i].append(dist[i])

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

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
                    "dists":              ep_dists[i].copy(),
                })
                ep_step[i]     = 0
                ep_min_dist[i] = float("inf")
                ep_dists[i]    = []

                if len(results) >= num_episodes:
                    break

    return results[:num_episodes]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(results: list[dict]) -> None:
    n          = len(results)
    successes  = sum(r["success"] for r in results)
    crashes    = sum(r["crash"] for r in results)
    obs_colls  = sum(r["obstacle_collision"] for r in results)
    timeouts   = sum(r["timeout"] for r in results)
    min_dists  = [r["min_dist"] for r in results]
    lengths    = [r["length"] for r in results]

    print(f"\n{'═'*52}")
    print(f"  Evaluation — {n} episodes")
    print(f"{'═'*52}")
    print(f"  Success            : {successes:3d} / {n}  ({100*successes/n:5.1f} %)")
    print(f"  Crash              : {crashes:3d} / {n}  ({100*crashes/n:5.1f} %)")
    print(f"  Obstacle collision : {obs_colls:3d} / {n}  ({100*obs_colls/n:5.1f} %)")
    print(f"  Timeout            : {timeouts:3d} / {n}  ({100*timeouts/n:5.1f} %)")
    print(f"{'─'*52}")
    print(f"  Closest approach  mean ± std : {np.mean(min_dists):.3f} ± {np.std(min_dists):.3f} m")
    print(f"  Closest approach  median     : {np.median(min_dists):.3f} m")
    print(f"  Episode length  mean ± std   : {np.mean(lengths):.0f} ± {np.std(lengths):.0f} steps")
    print(f"{'═'*52}\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(results: list[dict], out_dir: str) -> None:
    n          = len(results)
    successes  = sum(r["success"] for r in results)
    crashes    = sum(r["crash"] for r in results)
    obs_colls  = sum(r["obstacle_collision"] for r in results)
    timeouts   = sum(r["timeout"] for r in results)
    min_dists  = [r["min_dist"] for r in results]
    lengths    = [r["length"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Evaluation — {n} episodes", fontsize=14, fontweight="bold")

    # 1. Outcome bar chart
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

    # 4. Distance over time
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
                    alpha=0.2, color="#e67e22", label="±1 std")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to target (m)")
    ax.set_title("Distance over Episode (all episodes)")
    ax.legend()

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "eval_stats.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",   type=str, default="obstacle-avoidance")
    parser.add_argument("--log_dir",          type=str, default=None,
                        help="Direct path to log dir (overrides --exp_name)")
    parser.add_argument("--hpc",              nargs="?", const="list", default=None,
                        help="HPC run name (e.g. 'my_run'). No value = list runs.")
    parser.add_argument("--ckpt",             type=int, default=None,
                        help="Checkpoint iteration (default: latest)")
    parser.add_argument("--num_envs",         type=int, default=50)
    parser.add_argument("--num_episodes",     type=int, default=100)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--no-obstacles", action="store_true",
                        help="Hide all obstacles (curriculum Phase 1). Default is strategic placement.")
    parser.add_argument("--placement", choices=["strategic", "vineyard"], default=None,
                        help="Override placement strategy regardless of how the model was trained. "
                             "Default: use whatever cfgs.pkl specifies.")
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
        ckpt_iter = args.ckpt
    else:
        _, ckpt_iter = find_latest_checkpoint(log_dir)
    resume_path = os.path.join(log_dir, f"model_{ckpt_iter}.pt")
    print(f"Loading checkpoint: {resume_path}  (iteration {ckpt_iter})")

    reward_cfg["reward_scales"] = {}
    # Default: strategic obstacles from step 0. --no-obstacles forces Phase 1.
    env_cfg["curriculum_steps"] = float("inf") if args.no_obstacles else 0
    apply_placement_override(env_cfg, args.placement)

    if args.vis:
        env_cfg["visualize_target"] = True
        num_envs    = 1
        show_viewer = True
    else:
        env_cfg["visualize_target"] = False
        num_envs    = args.num_envs
        show_viewer = False

    env = ObstacleAvoidanceEnv(
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
        obs = env.reset()
        max_steps = env.max_episode_length
        with torch.no_grad():
            for _ in range(max_steps):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
    else:
        print(f"Collecting {args.num_episodes} episodes "
              f"across {num_envs} parallel envs ...")
        results = collect_episodes(env, policy, args.num_episodes)
        print_stats(results)
        save_plots(results, log_dir)


if __name__ == "__main__":
    main()

"""
# Latest checkpoint
python eval_rl.py

# List available HPC runs
python eval_rl.py --hpc

# Evaluate HPC run (latest checkpoint)
python eval_rl.py --hpc my_run

# Evaluate HPC run (specific checkpoint)
python eval_rl.py --hpc my_run --ckpt 400

# Custom episode / env count
python eval_rl.py --ckpt 300 --num_episodes 200 --num_envs 100

# Visual viewer
python eval_rl.py --ckpt 300 --vis
"""
