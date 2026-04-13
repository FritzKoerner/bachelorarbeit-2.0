"""Evaluate an SB3 PPO discrete-action model.

Usage:
    python eval_rl_sb3_discrete.py                         # latest checkpoint
    python eval_rl_sb3_discrete.py --ckpt 2048000          # specific checkpoint (by steps)
    python eval_rl_sb3_discrete.py --vis                   # viewer mode
    python eval_rl_sb3_discrete.py --hpc                   # list HPC runs
    python eval_rl_sb3_discrete.py --hpc sb3_discrete_01   # eval HPC run
"""

import argparse
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

from stable_baselines3 import PPO
import genesis as gs

from envs.coordinate_landing_simple_discrete_env import CoordinateLandingSimpleDiscreteEnv
from envs.coordinate_landing_simple_discrete_env_v2 import CoordinateLandingSimpleDiscreteEnvV2


# ---------------------------------------------------------------------------
# Find checkpoint
# ---------------------------------------------------------------------------

def find_latest_checkpoint(log_dir: str) -> tuple[str, int]:
    """Return (path, steps) for the highest-numbered ppo_*_steps.zip file."""
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    # Check both checkpoints/ subdir and log_dir root for final_model
    patterns = [
        os.path.join(ckpt_dir, "ppo_*_steps.zip"),
        os.path.join(log_dir, "final_model.zip"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")

    def step_num(path):
        m = re.search(r"ppo_(\d+)_steps\.zip$", path)
        if m:
            return int(m.group(1))
        if "final_model" in path:
            return float("inf")  # final is always latest
        return -1

    best = max(files, key=step_num)
    steps = step_num(best)
    return best, int(steps) if steps != float("inf") else -1


def find_checkpoint_by_steps(log_dir: str, steps: int) -> str:
    """Find checkpoint matching the given step count."""
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    path = os.path.join(ckpt_dir, f"ppo_{steps}_steps.zip")
    if os.path.exists(path):
        return path
    # Try final_model
    final = os.path.join(log_dir, "final_model.zip")
    if steps == -1 and os.path.exists(final):
        return final
    raise FileNotFoundError(
        f"Checkpoint not found: {path}\n"
        f"Available: {glob.glob(os.path.join(ckpt_dir, 'ppo_*_steps.zip'))}"
    )


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
            ckpt_dir = os.path.join(hpc_base, r, "checkpoints")
            ckpts = glob.glob(os.path.join(ckpt_dir, "ppo_*_steps.zip"))
            steps = sorted(int(re.search(r"ppo_(\d+)_steps", c).group(1))
                           for c in ckpts if re.search(r"ppo_(\d+)_steps", c))
            print(f"  {r:30s}  checkpoints (steps): {steps}")
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

def collect_episodes(env, model, num_episodes: int) -> list[dict]:
    """Run SB3 model on raw Genesis env. Handles numpy↔torch conversion."""
    num_envs = env.num_envs

    ep_step     = torch.zeros(num_envs, dtype=torch.int)
    ep_min_dist = [float("inf")] * num_envs
    ep_reward   = torch.zeros(num_envs, dtype=torch.float, device=gs.device)
    ep_dists    = [[] for _ in range(num_envs)]

    results = []
    obs_td = env.reset()
    obs_np = obs_td["policy"].cpu().numpy()

    while len(results) < num_episodes:
        dist = torch.norm(env.rel_pos, dim=1).cpu().tolist()
        ep_step += 1
        for i in range(num_envs):
            if dist[i] < ep_min_dist[i]:
                ep_min_dist[i] = dist[i]
            ep_dists[i].append(dist[i])

        # SB3 predict: numpy in, numpy out
        actions_np, _ = model.predict(obs_np, deterministic=True)
        # Convert to torch for Genesis env: (n_envs,) int → (n_envs, 1) float
        actions_t = torch.as_tensor(actions_np, device=gs.device, dtype=torch.float32)
        if actions_t.dim() == 1:
            actions_t = actions_t.unsqueeze(-1)

        obs_td, rewards, dones, extras = env.step(actions_t)
        obs_np = obs_td["policy"].cpu().numpy()
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
# Plots
# ---------------------------------------------------------------------------

def make_plots(results: list[dict]) -> plt.Figure:
    n         = len(results)
    successes = sum(r["success"] for r in results)
    crashes   = sum(r["crash"]   for r in results)
    timeouts  = sum(r["timeout"] for r in results)
    min_dists = [r["min_dist"] for r in results]
    lengths   = [r["length"]   for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Evaluation — {n} episodes", fontsize=14, fontweight="bold")

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

    ax = axes[1, 0]
    ax.hist(lengths, bins=25, color="#9b59b6", edgecolor="white")
    ax.axvline(np.mean(lengths), color="red", linestyle="--",
               label=f"mean = {np.mean(lengths):.0f} steps")
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Episodes")
    ax.set_title("Episode Length Distribution")
    ax.legend()

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

def log_to_wandb(results, stats, fig, ckpt_steps):
    table = wandb.Table(columns=["episode", "outcome", "length", "min_dist", "reward"])
    for i, r in enumerate(results):
        outcome = "success" if r["success"] else ("crash" if r["crash"] else "timeout")
        table.add_data(i, outcome, r["length"], r["min_dist"], r["reward"])
    wandb.log({"eval/episodes": table})

    for key, val in stats.items():
        wandb.summary[f"eval/{key}"] = val
    wandb.summary["eval/checkpoint_steps"] = ckpt_steps

    wandb.log({"eval/stats_plot": wandb.Image(fig)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",   type=str, default="drone-landing-sb3-discrete")
    parser.add_argument("--hpc",              nargs="?", const="list", default=None,
                        help="HPC run name. No value = list runs.")
    parser.add_argument("--log_dir",          type=str, default=None,
                        help="Direct path to log dir (overrides --exp_name)")
    parser.add_argument("--ckpt",             type=int, default=None,
                        help="Checkpoint step count (default: latest)")
    parser.add_argument("--num_envs",         type=int, default=50)
    parser.add_argument("--num_episodes",     type=int, default=100)
    parser.add_argument("--wandb_project",    type=str, default=None,
                        help="W&B project (default: auto from env version)")
    parser.add_argument("--vis", action="store_true",
                        help="Viewer mode: 1 env, 10 episodes, no W&B logging")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # --- Resolve log directory ---
    if args.hpc is not None:
        log_dir = resolve_hpc_log_dir(args.hpc)
    elif args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = f"logs/{args.exp_name}"

    # --- Load config ---
    cfgs_path = os.path.join(log_dir, "cfgs.pkl")
    if os.path.exists(cfgs_path):
        cfgs = pickle.load(open(cfgs_path, "rb"))
        env_cfg     = cfgs["env_cfg"]
        obs_cfg     = cfgs["obs_cfg"]
        reward_cfg  = cfgs["reward_cfg"]
        env_v2      = cfgs.get("env_v2", False)
    else:
        # Fallback: import defaults from training script
        print(f"Warning: {cfgs_path} not found, using default configs")
        from train_rl_sb3_discrete import get_env_cfgs
        env_cfg, obs_cfg, reward_cfg = get_env_cfgs()
        env_v2 = False

    # --- Find checkpoint ---
    if args.ckpt is not None:
        resume_path = find_checkpoint_by_steps(log_dir, args.ckpt)
        ckpt_steps = args.ckpt
    else:
        resume_path, ckpt_steps = find_latest_checkpoint(log_dir)
    print(f"Loading checkpoint: {resume_path}  (steps: {ckpt_steps})")

    # --- Eval-time overrides ---
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

    # --- Build env ---
    EnvClass = CoordinateLandingSimpleDiscreteEnvV2 if env_v2 else CoordinateLandingSimpleDiscreteEnv
    env = EnvClass(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=show_viewer,
    )
    env.build()

    # --- Load SB3 model ---
    model = PPO.load(resume_path, device="cuda")

    # --- Run ---
    if args.vis:
        num_vis_episodes = 10
        print(f"Viewer mode: running {num_vis_episodes} episodes")
        obs_td = env.reset()
        obs_np = obs_td["policy"].cpu().numpy()
        episode = 0
        with torch.no_grad():
            while episode < num_vis_episodes:
                action, _ = model.predict(obs_np, deterministic=True)
                action_t = torch.as_tensor(action, device=gs.device, dtype=torch.float32)
                if action_t.dim() == 1:
                    action_t = action_t.unsqueeze(-1)
                obs_td, _, dones, _ = env.step(action_t)
                obs_np = obs_td["policy"].cpu().numpy()
                if dones.any():
                    episode += 1
                    outcome = ("success" if env.success_condition[0].item()
                               else "crash" if env.crash_condition[0].item()
                               else "timeout")
                    print(f"  Episode {episode}/{num_vis_episodes}: {outcome}")
        print("Done.")
    else:
        print(f"Collecting {args.num_episodes} episodes across {num_envs} parallel envs ...")
        results = collect_episodes(env, model, args.num_episodes)

        stats = print_stats(results)
        fig = make_plots(results)

        plot_path = os.path.join(log_dir, "eval_stats.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved -> {plot_path}")

        run_name = f"{args.exp_name}-eval-{ckpt_steps}steps"
        wb_project = args.wandb_project or ("drone-sb3-discrete-v2" if env_v2 else "drone-sb3-discrete")
        wandb.init(
            project=wb_project,
            name=run_name,
            job_type="eval",
            config={
                "exp_name": args.exp_name,
                "checkpoint_steps": ckpt_steps,
                "num_episodes": args.num_episodes,
                "env_cfg": dict(env_cfg),
                "obs_cfg": obs_cfg,
                "env_v2": env_v2,
            },
        )
        log_to_wandb(results, stats, fig, ckpt_steps)
        wandb.finish()
        print(f"W&B eval run logged: {run_name}")

        plt.close(fig)


if __name__ == "__main__":
    main()
