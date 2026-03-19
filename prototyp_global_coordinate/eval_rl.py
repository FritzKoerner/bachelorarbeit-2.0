import argparse
import copy
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from envs.coordinate_landing_env import CoordinateLandingEnv


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
            # Snapshot distance BEFORE step so done-envs use pre-reset values
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
                success = bool(env.success_condition[i].item())
                crash   = bool(env.crash_condition[i].item())
                results.append({
                    "success":  success,
                    "crash":    crash,
                    "timeout":  not success and not crash,
                    "length":   int(ep_step[i].item()),
                    "min_dist": ep_min_dist[i],
                    "dists":    ep_dists[i].copy(),
                })
                ep_step[i]     = 0
                ep_min_dist[i] = float("inf")  # plain Python float, no device issue
                ep_dists[i]    = []

                if len(results) >= num_episodes:
                    break

    return results[:num_episodes]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(results: list[dict]) -> None:
    n         = len(results)
    successes = sum(r["success"] for r in results)
    crashes   = sum(r["crash"]   for r in results)
    timeouts  = sum(r["timeout"] for r in results)
    min_dists = [r["min_dist"] for r in results]
    lengths   = [r["length"]   for r in results]

    print(f"\n{'═'*48}")
    print(f"  Evaluation — {n} episodes")
    print(f"{'═'*48}")
    print(f"  Success : {successes:3d} / {n}  ({100*successes/n:5.1f} %)")
    print(f"  Crash   : {crashes:3d}   / {n}  ({100*crashes/n:5.1f} %)")
    print(f"  Timeout : {timeouts:3d}  / {n}  ({100*timeouts/n:5.1f} %)")
    print(f"{'─'*48}")
    print(f"  Closest approach  mean ± std : {np.mean(min_dists):.3f} ± {np.std(min_dists):.3f} m")
    print(f"  Closest approach  median     : {np.median(min_dists):.3f} m")
    print(f"  Best single approach         : {min(min_dists):.3f} m")
    print(f"  Episode length  mean ± std   : {np.mean(lengths):.0f} ± {np.std(lengths):.0f} steps")
    print(f"{'═'*48}\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(results: list[dict], out_dir: str) -> None:
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

    # 4. Distance over time (mean ± std across all episodes)
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
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "eval_stats.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",   type=str, default="drone-landing")
    parser.add_argument("--ckpt",             type=int, default=300)
    parser.add_argument("--num_envs",         type=int, default=50,
                        help="Parallel envs for stats collection")
    parser.add_argument("--num_episodes",     type=int, default=100)
    parser.add_argument("--vis", action="store_true",
                        help="Viewer mode: single env, no stats")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    reward_cfg["reward_scales"] = {}   # no reward needed during eval
    env_cfg["curriculum_steps"] = 0   # always use full spawn range in eval

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

    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    if args.vis:
        obs = env.reset()
        max_steps = int(env_cfg["episode_length_s"] * 100)
        with torch.no_grad():
            for _ in range(max_steps):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
    else:
        print(f"Collecting {args.num_episodes} episodes "
              f"across {num_envs} parallel envs …")
        results = collect_episodes(env, policy, args.num_episodes)
        print_stats(results)
        save_plots(results, log_dir)


if __name__ == "__main__":
    main()

"""
# Stats over 100 episodes (default)
python eval_rl.py --ckpt 300

# Custom episode / env count
python eval_rl.py --ckpt 300 --num_episodes 200 --num_envs 100

# Visual viewer (single env, no stats)
python eval_rl.py --ckpt 300 --vis
"""
