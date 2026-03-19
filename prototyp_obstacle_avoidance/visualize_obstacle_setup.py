"""Visualize obstacle avoidance environment layout: curriculum vs post-curriculum.

Shows:
- Curriculum: sparse random obstacles in wide area
- Post-curriculum: strategic placement with guaranteed blocker, corridor, and ring
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(42)

# --- Config (from train_rl.py) ---
TARGET = np.array([3.0, 3.0, 1.0])
SPAWN_OFFSET = 5.0
SPAWN_HEIGHT = 10.0

# Curriculum
OBS_X_RANGE = (-8.0, 12.0)
OBS_Y_RANGE = (-8.0, 12.0)
CURRICULUM_N_OBS = 5

# Post-curriculum
NUM_OBSTACLES = 8
N_CORRIDOR = 3
N_RING = 4
RING_R_MIN, RING_R_MAX = 1.5, 3.5
CORRIDOR_LATERAL = 2.0
POST_RANGE = 5.0

OBS_SIZE = (1.0, 1.0, 2.0)
COLLISION_RADIUS = 0.8
SAFETY_RADIUS = 3.0


def place_strategic(spawn_xy, target_xy):
    """Replicate _place_obstacles_strategic in numpy."""
    direction = target_xy - spawn_xy
    length = np.linalg.norm(direction)
    dir_norm = direction / (length + 1e-6)
    perp = np.array([-dir_norm[1], dir_norm[0]])

    positions = []

    # Guaranteed blocker
    t = np.random.uniform(0.4, 0.6)
    pos = spawn_xy + t * direction + np.random.uniform(-0.3, 0.3) * perp
    positions.append(pos)

    # Additional corridor
    for i in range(1, N_CORRIDOR):
        t_lo = 0.15 + (i - 1) * 0.25
        t = np.random.uniform(t_lo, t_lo + 0.25)
        offset = np.random.uniform(-CORRIDOR_LATERAL, CORRIDOR_LATERAL)
        pos = spawn_xy + t * direction + offset * perp
        positions.append(pos)

    # Ring around target
    for _ in range(N_RING):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(RING_R_MIN, RING_R_MAX)
        pos = target_xy + np.array([np.cos(angle), np.sin(angle)]) * radius
        positions.append(pos)

    # Remaining random near target
    remaining = NUM_OBSTACLES - len(positions)
    for _ in range(remaining):
        pos = target_xy + np.random.uniform(-POST_RANGE, POST_RANGE, 2)
        positions.append(pos)

    return np.array(positions)


fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Obstacle Setup: Curriculum vs Post-Curriculum", fontsize=14, fontweight="bold")

# ========== CURRICULUM (LEFT) ==========
ax = axes[0]
ax.set_title("Curriculum Phase (first 20k steps)\n5 sparse random obstacles", fontsize=11)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Obstacle region
ax.add_patch(patches.Rectangle(
    (OBS_X_RANGE[0], OBS_Y_RANGE[0]),
    OBS_X_RANGE[1] - OBS_X_RANGE[0], OBS_Y_RANGE[1] - OBS_Y_RANGE[0],
    linewidth=2, edgecolor="orange", facecolor="orange", alpha=0.06,
    linestyle="--", label="Obstacle spawn region",
))

# Drone spawn region
ax.add_patch(patches.Rectangle(
    (-SPAWN_OFFSET, -SPAWN_OFFSET), 2 * SPAWN_OFFSET, 2 * SPAWN_OFFSET,
    linewidth=2, edgecolor="blue", facecolor="blue", alpha=0.06,
    linestyle=":", label="Drone spawn region",
))

# 3 sample configs
colors_curr = ["#e74c3c", "#e67e22", "#9b59b6"]
for s in range(3):
    ox = np.random.uniform(*OBS_X_RANGE, CURRICULUM_N_OBS)
    oy = np.random.uniform(*OBS_Y_RANGE, CURRICULUM_N_OBS)
    for i in range(CURRICULUM_N_OBS):
        ax.add_patch(patches.Rectangle(
            (ox[i] - 0.5, oy[i] - 0.5), 1.0, 1.0,
            linewidth=1.5, edgecolor=colors_curr[s], facecolor=colors_curr[s], alpha=0.4,
        ))
        ax.add_patch(plt.Circle((ox[i], oy[i]), SAFETY_RADIUS,
            fill=False, edgecolor=colors_curr[s], alpha=0.15, linestyle="--", linewidth=0.6))
    if s == 0:
        ax.plot([], [], "s", color=colors_curr[s], markersize=8, label="Obstacles (3 samples)")

# Target (curriculum: near drone, shown as general area)
ax.plot(TARGET[0], TARGET[1], "*", color="green", markersize=18, zorder=10,
        label=f"Target ({TARGET[0]}, {TARGET[1]})")

# Sample drone spawns
for _ in range(6):
    dx, dy = np.random.uniform(-SPAWN_OFFSET, SPAWN_OFFSET, 2)
    ax.plot(dx, dy, "^", color="blue", markersize=8, alpha=0.5, zorder=9)
ax.plot([], [], "^", color="blue", markersize=8, alpha=0.5, label="Drone spawns")

ax.set_xlim(-12, 16)
ax.set_ylim(-12, 16)
ax.legend(loc="upper left", fontsize=8)

# ========== POST-CURRICULUM (RIGHT) ==========
ax2 = axes[1]
ax2.set_title("Post-Curriculum Phase\n8 strategic obstacles (blocker + corridor + ring)", fontsize=11)
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.3)

# Drone spawn region
ax2.add_patch(patches.Rectangle(
    (-SPAWN_OFFSET, -SPAWN_OFFSET), 2 * SPAWN_OFFSET, 2 * SPAWN_OFFSET,
    linewidth=2, edgecolor="blue", facecolor="blue", alpha=0.06,
    linestyle=":", label="Drone spawn region",
))

# Target
ax2.plot(TARGET[0], TARGET[1], "*", color="green", markersize=18, zorder=10,
         label=f"Target ({TARGET[0]}, {TARGET[1]})")

# Ring zone visualization
ring_theta = np.linspace(0, 2 * np.pi, 100)
for r, ls in [(RING_R_MIN, ":"), (RING_R_MAX, ":")]:
    ax2.plot(TARGET[0] + r * np.cos(ring_theta), TARGET[1] + r * np.sin(ring_theta),
             color="purple", linestyle=ls, alpha=0.3, linewidth=1.0)
ax2.fill_between(
    TARGET[0] + RING_R_MAX * np.cos(ring_theta),
    TARGET[1] + RING_R_MAX * np.sin(ring_theta),
    TARGET[1] + RING_R_MIN * np.sin(ring_theta),
    alpha=0.04, color="purple",
)
ax2.plot([], [], "-", color="purple", alpha=0.3, label=f"Ring zone ({RING_R_MIN}–{RING_R_MAX}m)")

# 3 sample configurations with different drone spawns
sample_colors = ["#e74c3c", "#2ecc71", "#3498db"]
sample_spawns = [
    np.array([-3.0, -2.0]),
    np.array([4.0, -4.0]),
    np.array([-1.0, 4.0]),
]

for s, (spawn_xy, color) in enumerate(zip(sample_spawns, sample_colors)):
    obstacles = place_strategic(spawn_xy, TARGET[:2])

    # Draw flight path
    ax2.annotate("", xy=TARGET[:2], xytext=spawn_xy,
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.3, ls="--"))

    # Drone spawn
    ax2.plot(spawn_xy[0], spawn_xy[1], "^", color=color, markersize=10, zorder=9)

    for i, pos in enumerate(obstacles):
        alpha = 0.7 if i == 0 else 0.4  # blocker is more opaque
        lw = 2.5 if i == 0 else 1.2

        # Obstacle box
        ax2.add_patch(patches.Rectangle(
            (pos[0] - 0.5, pos[1] - 0.5), 1.0, 1.0,
            linewidth=lw, edgecolor=color, facecolor=color, alpha=alpha,
        ))

        # Collision radius
        ax2.add_patch(plt.Circle((pos[0], pos[1]), COLLISION_RADIUS,
            fill=False, edgecolor=color, alpha=0.3, linestyle="-", linewidth=0.8))

        # Safety radius
        ax2.add_patch(plt.Circle((pos[0], pos[1]), SAFETY_RADIUS,
            fill=False, edgecolor=color, alpha=0.1, linestyle="--", linewidth=0.5))

        # Label blocker
        if i == 0 and s == 0:
            ax2.annotate("blocker", (pos[0], pos[1] + 0.8),
                         fontsize=7, ha="center", color=color, fontweight="bold")

    if s == 0:
        ax2.plot([], [], "s", color=color, markersize=8, alpha=0.7, label="Blocker (guaranteed)")
        ax2.plot([], [], "s", color=color, markersize=6, alpha=0.4, label="Corridor + ring + random")

ax2.plot([], [], "^", color="gray", markersize=8, label="Drone spawns (3 samples)")

ax2.set_xlim(-10, 12)
ax2.set_ylim(-10, 12)
ax2.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("prototyp_obstacle_avoidance/obstacle_setup_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: prototyp_obstacle_avoidance/obstacle_setup_comparison.png")
plt.close()


# ========== SIDE VIEW ==========
fig2, ax3 = plt.subplots(1, 1, figsize=(12, 5))
ax3.set_title("Side View (XZ) — Post-Curriculum Approach", fontsize=12)
ax3.set_xlabel("X (m)")
ax3.set_ylabel("Z (m)")
ax3.set_aspect("equal")
ax3.grid(True, alpha=0.3)

# Ground
ax3.axhline(y=0, color="brown", linewidth=2, alpha=0.5)
ax3.fill_between([-10, 12], 0, -0.5, color="brown", alpha=0.1)

# One sample config
spawn_xy = np.array([-3.0, -2.0])
obstacles = place_strategic(spawn_xy, TARGET[:2])

# Project obstacles onto XZ plane (show X position, height is OBS_SIZE[2])
for i, pos in enumerate(obstacles):
    color = "#e74c3c" if i == 0 else "#e67e22"
    alpha = 0.6 if i == 0 else 0.35
    ax3.add_patch(patches.Rectangle(
        (pos[0] - 0.5, 0), 1.0, OBS_SIZE[2],
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=alpha,
    ))
    if i == 0:
        ax3.text(pos[0], OBS_SIZE[2] + 0.3, "blocker", ha="center", fontsize=7, color=color)

# Drone spawn
ax3.plot(spawn_xy[0], SPAWN_HEIGHT, "^", color="blue", markersize=12, zorder=10, label="Drone spawn")

# Target
ax3.plot(TARGET[0], TARGET[2], "*", color="green", markersize=18, zorder=10,
         label=f"Target (z={TARGET[2]}m)")

# Descent path
ax3.annotate("", xy=(TARGET[0], TARGET[2] + 0.3), xytext=(spawn_xy[0], SPAWN_HEIGHT - 0.3),
             arrowprops=dict(arrowstyle="->", color="blue", lw=1.5, alpha=0.4, ls="--"))

ax3.set_xlim(-10, 12)
ax3.set_ylim(-1, 13)
ax3.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("prototyp_obstacle_avoidance/obstacle_setup_sideview.png", dpi=150, bbox_inches="tight")
print("Saved: prototyp_obstacle_avoidance/obstacle_setup_sideview.png")
plt.close()


# ========== STATS ==========
print("\n=== Configuration Comparison ===")
print(f"{'':30s} {'Curriculum':>15s} {'Post-curriculum':>15s}")
print(f"{'Active obstacles':30s} {CURRICULUM_N_OBS:>15d} {NUM_OBSTACLES:>15d}")
print(f"{'Placement':30s} {'random sparse':>15s} {'strategic':>15s}")
print(f"{'Guaranteed blocker':30s} {'no':>15s} {'yes':>15s}")
print(f"{'Obstacle area':30s} {'400 m²':>15s} {'~50 m²':>15s}")
print(f"{'Target':30s} {'near drone':>15s} {'(3,3,1) fixed':>15s}")

# Blocker effectiveness check
print("\n=== Blocker Effectiveness ===")
n_tests = 10000
blocked = 0
for _ in range(n_tests):
    sx = np.random.uniform(-SPAWN_OFFSET, SPAWN_OFFSET)
    sy = np.random.uniform(-SPAWN_OFFSET, SPAWN_OFFSET)
    spawn = np.array([sx, sy])
    obs = place_strategic(spawn, TARGET[:2])
    blocker = obs[0]
    # Check: does the direct path from spawn to target pass within collision_radius?
    # Point-to-line distance
    d = TARGET[:2] - spawn
    t_proj = np.dot(blocker - spawn, d) / (np.dot(d, d) + 1e-9)
    t_proj = np.clip(t_proj, 0, 1)
    closest = spawn + t_proj * d
    dist = np.linalg.norm(blocker - closest)
    if dist < COLLISION_RADIUS + OBS_SIZE[0] / 2:
        blocked += 1
print(f"Direct path blocked by blocker: {blocked/n_tests:.1%} of {n_tests} random spawns")

# Safety radius encounter rate
encounters = 0
for _ in range(n_tests):
    sx = np.random.uniform(-SPAWN_OFFSET, SPAWN_OFFSET)
    sy = np.random.uniform(-SPAWN_OFFSET, SPAWN_OFFSET)
    spawn = np.array([sx, sy])
    obs = place_strategic(spawn, TARGET[:2])
    # Check if any obstacle is within safety_radius of the direct path
    d = TARGET[:2] - spawn
    for pos in obs:
        t_proj = np.dot(pos - spawn, d) / (np.dot(d, d) + 1e-9)
        t_proj = np.clip(t_proj, 0, 1)
        closest = spawn + t_proj * d
        if np.linalg.norm(pos - closest) < SAFETY_RADIUS:
            encounters += 1
            break
print(f"Path enters safety radius:     {encounters/n_tests:.1%} of {n_tests} random spawns")
