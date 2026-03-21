"""Per-step RL diagnostics: observations, actions, rewards, and value estimates.

Loads a checkpoint (or uses random/zero policy), runs episodes in a single env,
captures every step's data, and generates a multi-panel diagnostic plot.

Usage:
    python debug_rl.py --ckpt 300                       # 5 episodes, diagnostic plot
    python debug_rl.py --ckpt 300 --episodes 1 --csv    # 1 episode + CSV export
    python debug_rl.py --ckpt 300 --episodes 1 --stochastic  # stochastic actions
    python debug_rl.py --random                          # random policy (env sanity check)
    python debug_rl.py --zero                            # zero actions (hover check)
    python debug_rl.py --log_dir ../hpc_results/... --ckpt 400
"""

import argparse
import copy
import csv
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
from train_rl_wb import DictConfig  # needed to unpickle cfgs.pkl from W&B training


# ---------------------------------------------------------------------------
# Per-step data collection
# ---------------------------------------------------------------------------

def collect_episode_debug(env, actor, critic, *, stochastic=False):
    """Run one episode, recording per-step diagnostics."""
    data = {
        "obs": [], "actions": [], "action_mean": [], "action_std": [],
        "reward_total": [], "reward_components": [], "value": [],
        "pos": [], "distance": [],
    }

    obs = env.reset()

    while True:
        state_vec = obs["policy"][0].cpu().numpy().copy()
        dist = torch.norm(env.rel_pos[0]).item()
        pos = env.base_pos[0].cpu().numpy().copy()

        with torch.no_grad():
            actions = actor(obs, stochastic_output=True)
            action_mean = actor.output_mean[0].cpu().numpy().copy()
            action_std = actor.output_std[0].cpu().numpy().copy()
            value = critic(obs)[0].item()

        if not stochastic:
            actions = torch.as_tensor(
                action_mean, dtype=actions.dtype, device=actions.device
            ).unsqueeze(0)

        obs, rew, dones, _ = env.step(actions)

        components = {
            name: vals[0].item() for name, vals in env.reward_components.items()
        }

        data["obs"].append(state_vec)
        data["actions"].append(actions[0].cpu().numpy().copy())
        data["action_mean"].append(action_mean)
        data["action_std"].append(action_std)
        data["reward_total"].append(rew[0].item())
        data["reward_components"].append(components)
        data["value"].append(value)
        data["pos"].append(pos)
        data["distance"].append(dist)

        if dones[0].item():
            data["outcome"] = (
                "success" if env.success_condition[0].item()
                else "crash" if env.crash_condition[0].item()
                else "timeout"
            )
            break

    return data


def collect_episode_policy(env, policy_fn):
    """Simpler collector for random/zero policies."""
    data = {
        "obs": [], "actions": [], "reward_total": [], "reward_components": [],
        "distance": [], "pos": [],
    }
    obs = env.reset()

    while True:
        state_vec = obs["policy"][0].cpu().numpy().copy()
        dist = torch.norm(env.rel_pos[0]).item()
        pos = env.base_pos[0].cpu().numpy().copy()

        with torch.no_grad():
            actions = policy_fn(obs)

        obs, rew, dones, _ = env.step(actions)

        components = {
            name: vals[0].item() for name, vals in env.reward_components.items()
        }

        data["obs"].append(state_vec)
        data["actions"].append(actions[0].cpu().numpy().copy())
        data["reward_total"].append(rew[0].item())
        data["reward_components"].append(components)
        data["distance"].append(dist)
        data["pos"].append(pos)

        if dones[0].item():
            data["outcome"] = (
                "success" if env.success_condition[0].item()
                else "crash" if env.crash_condition[0].item()
                else "timeout"
            )
            break

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

OBS_LABELS = [
    "rel_x", "rel_y", "rel_z",         # 0-2
    "qw", "qx", "qy", "qz",           # 3-6
    "vx", "vy", "vz",                  # 7-9
    "wx", "wy", "wz",                  # 10-12
    "a_x", "a_y", "a_z", "a_yaw",     # 13-16
]

ACTION_LABELS = ["ax (x-offset)", "ay (y-offset)", "az (z-offset)", "ayaw (yaw)"]


def plot_episode(data, ep_idx, out_dir, ckpt_label):
    """Generate a multi-panel diagnostic figure for one episode."""
    n_steps = len(data["obs"])
    steps = np.arange(n_steps)
    obs_arr = np.array(data["obs"])
    act_arr = np.array(data["actions"])
    has_dist = "action_mean" in data and len(data["action_mean"]) > 0

    # Panels: obs, actions, rewards, [value if model], distance
    n_rows = 5 if has_dist else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.2 * n_rows), sharex=True)
    outcome = data.get("outcome", "?")
    fig.suptitle(
        f"Episode {ep_idx + 1}  |  {n_steps} steps  |  outcome: {outcome}  |  ckpt: {ckpt_label}",
        fontsize=13, fontweight="bold",
    )
    ax_idx = 0

    # --- Panel 1: Observations ---
    ax = axes[ax_idx]; ax_idx += 1
    for d, lbl in [(0, "rel_x"), (1, "rel_y"), (2, "rel_z"),
                    (7, "vx"), (8, "vy"), (9, "vz"),
                    (10, "wx"), (11, "wy"), (12, "wz")]:
        ax.plot(steps, obs_arr[:, d], label=lbl, linewidth=0.9)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.7)
    ax.axhline(-1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_ylabel("Scaled value")
    ax.set_title("Observations (clipped dims: rel_pos, lin_vel, ang_vel)")
    ax.legend(ncol=3, fontsize=7, loc="upper right")

    # --- Panel 2: Actions ---
    ax = axes[ax_idx]; ax_idx += 1
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]
    for d in range(4):
        ax.plot(steps, act_arr[:, d], label=ACTION_LABELS[d], color=colors[d], linewidth=0.9)
        if has_dist:
            mean_arr = np.array(data["action_mean"])
            std_arr = np.array(data["action_std"])
            ax.fill_between(
                steps,
                mean_arr[:, d] - std_arr[:, d],
                mean_arr[:, d] + std_arr[:, d],
                alpha=0.15, color=colors[d],
            )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.7)
    ax.axhline(-1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_ylabel("Action value")
    ax.set_title("Actions" + (" (line = action, band = policy mean +/- std)" if has_dist else ""))
    ax.legend(ncol=4, fontsize=7, loc="upper right")

    # --- Panel 3: Reward breakdown ---
    ax = axes[ax_idx]; ax_idx += 1
    comp_names = list(data["reward_components"][0].keys())
    reward_colors = {
        "distance": "#3498db", "time": "#95a5a6",
        "crash": "#e74c3c", "success": "#2ecc71",
    }
    for name in comp_names:
        vals = [c[name] for c in data["reward_components"]]
        ax.plot(steps, vals, label=name, linewidth=0.9,
                color=reward_colors.get(name))
    ax.plot(steps, data["reward_total"], label="total", color="black",
            linewidth=1.2, linestyle="--")
    ax.set_ylabel("Reward")
    ax.set_title("Per-step reward breakdown")
    ax.legend(ncol=3, fontsize=7, loc="upper right")

    # --- Panel 4: Value vs return (only with model) ---
    if has_dist:
        ax = axes[ax_idx]; ax_idx += 1
        values = np.array(data["value"])
        gamma = 0.99
        rewards = np.array(data["reward_total"])
        returns = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            returns[t] = running
        ax.plot(steps, values, label="V(s) critic", color="#e74c3c", linewidth=1.0)
        ax.plot(steps, returns, label="G(t) actual return", color="#2ecc71", linewidth=1.0)
        ax.set_ylabel("Value")
        ax.set_title("Value estimate vs actual discounted return")
        ax.legend(fontsize=8)

    # --- Last panel: Distance to target ---
    ax = axes[-1]
    ax.plot(steps, data["distance"], label="dist to target", color="#e67e22", linewidth=1.2)
    ax.axhline(0.3, color="#2ecc71", linestyle=":", alpha=0.6, label="success radius (0.3m)")
    ax.set_ylabel("Distance (m)")
    ax.set_xlabel("Step")
    ax.set_title("Distance to target")
    ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = f"debug_rl_{ckpt_label}_ep{ep_idx + 1}.png"
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {path}")


def print_summary(data, ep_idx):
    """Print per-episode summary statistics to console."""
    obs_arr = np.array(data["obs"])
    act_arr = np.array(data["actions"])
    n = len(obs_arr)

    print(f"\n{'─'*60}")
    print(f"  Episode {ep_idx + 1}  |  {n} steps  |  outcome: {data.get('outcome', '?')}")
    print(f"{'─'*60}")

    print("  Observation ranges (clipped dims):")
    for name, cols in [("rel_pos", slice(0, 3)), ("lin_vel", slice(7, 10)), ("ang_vel", slice(10, 13))]:
        mins = obs_arr[:, cols].min(axis=0)
        maxs = obs_arr[:, cols].max(axis=0)
        clipped = (np.abs(obs_arr[:, cols]) > 0.99).any(axis=1).mean()
        print(f"    {name:8s}  min={np.array2string(mins, precision=3, separator=', ')}"
              f"  max={np.array2string(maxs, precision=3, separator=', ')}"
              f"  clip_rate={clipped:.1%}")

    print("  Action stats:")
    for d, name in enumerate(ACTION_LABELS):
        col = act_arr[:, d]
        sat_rate = (np.abs(col) > 0.99).mean()
        print(f"    {name:16s}  mean={col.mean():+.3f}  std={col.std():.3f}"
              f"  sat_rate={sat_rate:.1%}")

    comp_names = list(data["reward_components"][0].keys())
    total = sum(data["reward_total"])
    print(f"  Cumulative reward: {total:.2f}")
    for name in comp_names:
        vals = [c[name] for c in data["reward_components"]]
        print(f"    {name:24s}  sum={sum(vals):+.3f}  mean={np.mean(vals):+.5f}")

    print(f"  Final distance: {data['distance'][-1]:.3f} m")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(data, path):
    """Export per-step data to CSV."""
    n_obs = len(data["obs"][0])
    comp_names = list(data["reward_components"][0].keys())
    has_model = "action_mean" in data and len(data["action_mean"]) > 0

    fieldnames = ["step"]
    fieldnames += [f"obs_{i}_{OBS_LABELS[i]}" for i in range(n_obs)]
    fieldnames += [f"action_{i}" for i in range(4)]
    if has_model:
        fieldnames += [f"action_mean_{i}" for i in range(4)]
        fieldnames += [f"action_std_{i}" for i in range(4)]
        fieldnames += ["value"]
    fieldnames += ["reward_total"] + [f"rew_{n}" for n in comp_names]
    fieldnames += ["distance", "pos_x", "pos_y", "pos_z"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in range(len(data["obs"])):
            row = {"step": t}
            for i in range(n_obs):
                row[f"obs_{i}_{OBS_LABELS[i]}"] = f"{data['obs'][t][i]:.6f}"
            for i in range(4):
                row[f"action_{i}"] = f"{data['actions'][t][i]:.6f}"
            if has_model:
                for i in range(4):
                    row[f"action_mean_{i}"] = f"{data['action_mean'][t][i]:.6f}"
                    row[f"action_std_{i}"] = f"{data['action_std'][t][i]:.6f}"
                row["value"] = f"{data['value'][t]:.6f}"
            row["reward_total"] = f"{data['reward_total'][t]:.6f}"
            for n in comp_names:
                row[f"rew_{n}"] = f"{data['reward_components'][t][n]:.6f}"
            row["distance"] = f"{data['distance'][t]:.4f}"
            row["pos_x"] = f"{data['pos'][t][0]:.4f}"
            row["pos_y"] = f"{data['pos'][t][1]:.4f}"
            row["pos_z"] = f"{data['pos'][t][2]:.4f}"
            writer.writerow(row)
    print(f"  CSV saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-step RL diagnostics")
    parser.add_argument("-e", "--exp_name", type=str, default="drone-landing")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Direct path to log dir (overrides --exp_name)")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint iteration to load")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--csv", action="store_true",
                        help="Export per-step data to CSV")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic (sampled) actions instead of deterministic")
    parser.add_argument("--random", action="store_true",
                        help="Random policy (ignores --ckpt)")
    parser.add_argument("--zero", action="store_true",
                        help="Zero actions / hover (ignores --ckpt)")
    args = parser.parse_args()

    if not args.random and not args.zero and args.ckpt is None:
        parser.error("Specify --ckpt N, --random, or --zero")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    log_dir = args.log_dir if args.log_dir else f"logs/{args.exp_name}"

    if args.random or args.zero:
        cfg_path = os.path.join(log_dir, "cfgs.pkl")
        if os.path.exists(cfg_path):
            env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
        else:
            from train_rl_wb import env_cfg, obs_cfg, reward_cfg, train_cfg
        ckpt_label = "random" if args.random else "zero"
    else:
        env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(
            open(os.path.join(log_dir, "cfgs.pkl"), "rb")
        )
        ckpt_label = f"ckpt{args.ckpt}"

    env_cfg["curriculum_steps"] = 0

    env = CoordinateLandingEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
    )
    env.build()

    use_model = not args.random and not args.zero
    actor = critic = None

    if use_model:
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir, device=gs.device)
        runner.load(resume_path)
        runner.alg.eval_mode()
        actor = runner.alg.actor
        critic = runner.alg.critic
        print(f"Loaded checkpoint: {resume_path}")
    else:
        if args.random:
            print("Using random policy")
        else:
            print("Using zero actions (hover)")

    out_dir = os.path.join(log_dir, "debug")

    print(f"Running {args.episodes} episode(s) ...")
    for ep in range(args.episodes):
        if use_model:
            data = collect_episode_debug(
                env, actor, critic, stochastic=args.stochastic
            )
        else:
            if args.random:
                def random_policy(obs):
                    return torch.rand(1, 4, device=gs.device) * 2 - 1
                data = collect_episode_policy(env, random_policy)
            else:
                def zero_policy(obs):
                    return torch.zeros(1, 4, device=gs.device)
                data = collect_episode_policy(env, zero_policy)

        print_summary(data, ep)
        plot_episode(data, ep, out_dir, ckpt_label)

        if args.csv:
            csv_path = os.path.join(out_dir, f"debug_rl_{ckpt_label}_ep{ep + 1}.csv")
            export_csv(data, csv_path)

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
