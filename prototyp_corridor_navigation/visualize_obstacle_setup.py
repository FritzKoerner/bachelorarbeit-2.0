"""Visualize corridor navigation obstacle placement.

Pure numpy + matplotlib replica of ``_place_obstacles_corridor`` in
``envs/corridor_navigation_env.py``.  Keeps iteration fast (< 1 s) by
skipping Genesis scene build.  If the env's placement formula changes,
mirror the change here.

Renders three sample configs in top-down (XY) and side (XZ) views, plus
a small stats block verifying the pair-gap and first-distance ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(42)


# ------------------------------------------------------------------
# Config mirrors prototyp_corridor_navigation/train_rl_wb.py env_cfg.
# ------------------------------------------------------------------

# Corridor geometry (metres)
CORRIDOR_X_RANGE = (0.0, 16.0)
CORRIDOR_Y_RANGE = (-4.0, 4.0)
CORRIDOR_Z_RANGE = (0.0, 6.0)
SPAWN_X_RANGE = (0.5, 2.5)
SPAWN_Y_RANGE = (-2.5, 2.5)
SPAWN_Z = 4.0
TARGET = np.array([15.0, 0.0, 1.0])
SLICE_CENTRES = [5.0, 10.0]

# Proximity / collision (mirror env_cfg)
SAFETY_RADIUS = 1.5
COLLISION_RADIUS = 0.3

# Physical corridor walls (L, R, ceiling) — thin boxes hugging the corridor
# bounds. Ground plane at z=0 is handled by gs.morphs.Plane() in the real env.
WALL_THICKNESS = 0.2

# Pair-straddling knobs (must match env_cfg)
FIRST_OFFSET_RANGE = (0.0, 3.0)
PAIR_GAP_RANGE = (1.0, 2.0)
X_JITTER = 0.2

# Obstacle size specs (must match env_cfg lists). Shape-interleaved:
# [Box0, Sphere0, Cyl0, Pillar0] — one of each shape, paired across 2 slices.
BOX_SIZES = [(2.5, 2.5, 3.0)]
SPHERE_RADII = [1.2]
CYLINDER_SPECS = [(1.0, 3.0)]     # (radius, height)
PILLAR_SPECS = [(0.5, 4.5)]       # (radius, height)


def build_specs():
    """Return a list of 4*N shape dicts in the same interleaved order as the env."""
    specs = []
    for i in range(len(BOX_SIZES)):
        bx = BOX_SIZES[i]
        specs.append(dict(
            morph="box", size=bx,
            half_y=bx[1] / 2, half_z=bx[2] / 2, z_rest=bx[2] / 2,
            is_pillar=False,
        ))
        sr = SPHERE_RADII[i]
        specs.append(dict(
            morph="sphere", radius=sr,
            half_y=sr, half_z=sr, z_rest=sr + 0.3,
            is_pillar=False,
        ))
        cr, ch = CYLINDER_SPECS[i]
        specs.append(dict(
            morph="cylinder", radius=cr, height=ch,
            half_y=cr, half_z=ch / 2, z_rest=ch / 2,
            is_pillar=False,
        ))
        pr, ph = PILLAR_SPECS[i]
        specs.append(dict(
            morph="cylinder", radius=pr, height=ph,
            half_y=pr, half_z=ph / 2, z_rest=ph / 2,
            is_pillar=True,
        ))
    return specs


SPECS = build_specs()
N_SLICES = len(SLICE_CENTRES)
assert len(SPECS) == 2 * N_SLICES, (
    f"Shape specs produce {len(SPECS)} obstacles but slice_centres "
    f"wants {2 * N_SLICES} (2 per slice)"
)

# Per-slice pillar flag: if either obstacle in the pair is a pillar,
# axis is locked to Y (vertical detour is geometrically unreasonable
# for 4 m-tall pillars in a 5.7 m corridor).
PAIR_HAS_PILLAR = [
    SPECS[2 * s]["is_pillar"] or SPECS[2 * s + 1]["is_pillar"]
    for s in range(N_SLICES)
]

# Spawn centre for the line anchor.
SPAWN_CX = 0.5 * (SPAWN_X_RANGE[0] + SPAWN_X_RANGE[1])
SPAWN_CY = 0.5 * (SPAWN_Y_RANGE[0] + SPAWN_Y_RANGE[1])
LINE_START = np.array([SPAWN_CX, SPAWN_CY, SPAWN_Z])
LINE_END = TARGET


def line_yz_at_x(x):
    """Return (y, z) on the spawn->target line at the given x."""
    t = (x - LINE_START[0]) / (LINE_END[0] - LINE_START[0])
    y = LINE_START[1] + t * (LINE_END[1] - LINE_START[1])
    z = LINE_START[2] + t * (LINE_END[2] - LINE_START[2])
    return y, z


def place_corridor():
    """Numpy twin of ``_place_obstacles_corridor``.  Returns (8, 3) positions."""
    positions = np.zeros((8, 3))
    for slice_i, x_centre in enumerate(SLICE_CENTRES):
        idx_a = 2 * slice_i
        idx_b = 2 * slice_i + 1
        spec_a = SPECS[idx_a]
        spec_b = SPECS[idx_b]

        x_a = x_centre + np.random.uniform(-X_JITTER, X_JITTER)
        x_b = x_centre + np.random.uniform(-X_JITTER, X_JITTER)

        y_line_a, z_line_a = line_yz_at_x(x_a)
        y_line_b, z_line_b = line_yz_at_x(x_b)

        axis = 0 if PAIR_HAS_PILLAR[slice_i] else np.random.randint(2)  # 0=Y, 1=Z
        side = 1.0 if np.random.randint(2) == 0 else -1.0
        d = np.random.uniform(*FIRST_OFFSET_RANGE)
        gap = np.random.uniform(*PAIR_GAP_RANGE)

        if axis == 0:  # Y-axis pair
            y_a = y_line_a + side * d
            z_a = z_line_a
            y_b = y_a - side * (spec_a["half_y"] + gap + spec_b["half_y"])
            z_b = z_line_b
        else:           # Z-axis pair
            y_a = y_line_a
            z_a = z_line_a + side * d
            y_b = y_line_b
            z_b = z_a - side * (spec_a["half_z"] + gap + spec_b["half_z"])

        positions[idx_a] = [x_a, y_a, z_a]
        positions[idx_b] = [x_b, y_b, z_b]

    return positions


# ------------------------------------------------------------------
# Shape-aware drawing helpers
# ------------------------------------------------------------------

def draw_obstacle_xy(ax, spec, pos, color, alpha):
    x, y, _ = pos
    if spec["morph"] == "box":
        sx, sy, _ = spec["size"]
        ax.add_patch(patches.Rectangle(
            (x - sx / 2, y - sy / 2), sx, sy,
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=alpha,
        ))
    elif spec["morph"] == "sphere":
        ax.add_patch(plt.Circle(
            (x, y), spec["radius"],
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=alpha,
        ))
    else:  # cylinder / pillar: circle in XY
        ax.add_patch(plt.Circle(
            (x, y), spec["radius"],
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=alpha,
        ))


def draw_obstacle_xz(ax, spec, pos, color, alpha):
    x, _, z = pos
    if spec["morph"] == "box":
        sx, _, sz = spec["size"]
        ax.add_patch(patches.Rectangle(
            (x - sx / 2, z - sz / 2), sx, sz,
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=alpha,
        ))
    elif spec["morph"] == "sphere":
        ax.add_patch(plt.Circle(
            (x, z), spec["radius"],
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=alpha,
        ))
    else:  # vertical cylinder: rectangle in XZ
        ax.add_patch(patches.Rectangle(
            (x - spec["radius"], z - spec["height"] / 2),
            2 * spec["radius"], spec["height"],
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=alpha,
        ))


# ------------------------------------------------------------------
# Figures: one per sample, each with top-down (XY) + side (XZ) view.
# ------------------------------------------------------------------

N_SAMPLES = 2
sample_colors = ["#e74c3c", "#2ecc71"]
sample_positions = [place_corridor() for _ in range(N_SAMPLES)]


def draw_sample_figure(sample_idx, positions, color):
    fig, (ax_xy, ax_xz) = plt.subplots(2, 1, figsize=(16, 11))
    fig.suptitle(
        f"Corridor Obstacle Placement — Sample {sample_idx + 1} "
        "(pair-straddling strategy)",
        fontsize=14, fontweight="bold",
    )

    # ---------- Top-down (XY) ----------
    ax_xy.set_title(
        "Top-down (XY). Pair straddles a gap ∈ [1, 2] m on Y or Z; "
        "obstacle A distance from the line ∈ [0, 3] m.",
        fontsize=11,
    )
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)

    ax_xy.add_patch(patches.Rectangle(
        (CORRIDOR_X_RANGE[0], CORRIDOR_Y_RANGE[0]),
        CORRIDOR_X_RANGE[1] - CORRIDOR_X_RANGE[0],
        CORRIDOR_Y_RANGE[1] - CORRIDOR_Y_RANGE[0],
        linewidth=1, edgecolor="black", facecolor="none",
        linestyle=":", alpha=0.4,
    ))
    # Physical walls (L, R) — thin gray boxes just outside the bounds.
    for wall_y_centre in (CORRIDOR_Y_RANGE[0] - WALL_THICKNESS / 2.0,
                          CORRIDOR_Y_RANGE[1] + WALL_THICKNESS / 2.0):
        ax_xy.add_patch(patches.Rectangle(
            (CORRIDOR_X_RANGE[0], wall_y_centre - WALL_THICKNESS / 2.0),
            CORRIDOR_X_RANGE[1] - CORRIDOR_X_RANGE[0],
            WALL_THICKNESS,
            linewidth=1.2, edgecolor="#333", facecolor="#555", alpha=0.85,
        ))
    ax_xy.plot([], [], "s", color="#555", markersize=9, alpha=0.85, label="Wall (physical)")
    ax_xy.add_patch(patches.Rectangle(
        (SPAWN_X_RANGE[0], SPAWN_Y_RANGE[0]),
        SPAWN_X_RANGE[1] - SPAWN_X_RANGE[0],
        SPAWN_Y_RANGE[1] - SPAWN_Y_RANGE[0],
        linewidth=1.5, edgecolor="blue", facecolor="blue",
        alpha=0.1, linestyle=":", label="Spawn region",
    ))

    for xc in SLICE_CENTRES:
        ax_xy.axvline(x=xc, color="lightgray", linestyle=":", alpha=0.6, linewidth=0.8)

    ax_xy.plot([SPAWN_CX, TARGET[0]], [SPAWN_CY, TARGET[1]],
               "-", color="gray", alpha=0.5, linewidth=1.2,
               label="Spawn→target line")
    ax_xy.plot(TARGET[0], TARGET[1], "*", color="green", markersize=20,
               zorder=10, label=f"Target ({TARGET[0]:.0f}, {TARGET[1]:.0f})")

    for i, (pos, spec) in enumerate(zip(positions, SPECS)):
        draw_obstacle_xy(ax_xy, spec, pos, color, 0.55)
        # Pair link + mid-pair X label per slice
        if i % 2 == 1:
            pa = positions[i - 1]
            slice_i = i // 2
            ax_xy.plot([pa[0], pos[0]], [pa[1], pos[1]],
                       "-", color=color, alpha=0.7, linewidth=1.2)
            mid_x = 0.5 * (pa[0] + pos[0])
            mid_y = 0.5 * (pa[1] + pos[1])
            tag = "PILLAR" if PAIR_HAS_PILLAR[slice_i] else f"slice {slice_i}"
            ax_xy.annotate(tag, (mid_x, mid_y + 0.25), color=color,
                           fontsize=8, ha="center", fontweight="bold")

    ax_xy.plot([], [], "s", color="gray", markersize=9, alpha=0.55, label="Box footprint")
    ax_xy.plot([], [], "o", color="gray", markersize=9, alpha=0.55, label="Sphere / Cyl / Pillar")

    ax_xy.set_xlim(CORRIDOR_X_RANGE[0] - 1, CORRIDOR_X_RANGE[1] + 1)
    ax_xy.set_ylim(CORRIDOR_Y_RANGE[0] - 1.5, CORRIDOR_Y_RANGE[1] + 1.5)
    ax_xy.legend(loc="upper right", fontsize=8, ncol=2)

    # ---------- Side (XZ) ----------
    ax_xz.set_title(
        "Side (XZ). Pillar-containing slices are locked to Y-axis pairs; "
        "their two obstacles project onto a single column in this view.",
        fontsize=11,
    )
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_aspect("equal")
    ax_xz.grid(True, alpha=0.3)

    ax_xz.add_patch(patches.Rectangle(
        (CORRIDOR_X_RANGE[0], CORRIDOR_Z_RANGE[0]),
        CORRIDOR_X_RANGE[1] - CORRIDOR_X_RANGE[0],
        CORRIDOR_Z_RANGE[1] - CORRIDOR_Z_RANGE[0],
        linewidth=1, edgecolor="black", facecolor="none",
        linestyle=":", alpha=0.4,
    ))
    # Ceiling (physical top wall) — thin gray strip just above z_max.
    ax_xz.add_patch(patches.Rectangle(
        (CORRIDOR_X_RANGE[0], CORRIDOR_Z_RANGE[1]),
        CORRIDOR_X_RANGE[1] - CORRIDOR_X_RANGE[0],
        WALL_THICKNESS,
        linewidth=1.2, edgecolor="#333", facecolor="#555", alpha=0.85,
    ))
    ax_xz.plot([], [], "s", color="#555", markersize=9, alpha=0.85, label="Ceiling (physical)")
    ax_xz.axhline(y=0, color="brown", linewidth=2, alpha=0.5)
    ax_xz.fill_between([-1, 33], 0, -0.5, color="brown", alpha=0.1)

    for xc in SLICE_CENTRES:
        ax_xz.axvline(x=xc, color="lightgray", linestyle=":", alpha=0.6, linewidth=0.8)

    ax_xz.plot(SPAWN_CX, SPAWN_Z, "^", color="blue", markersize=12,
               zorder=10, label=f"Spawn centre (z={SPAWN_Z:.0f})")
    ax_xz.plot(TARGET[0], TARGET[2], "*", color="green", markersize=20,
               zorder=10, label=f"Target (z={TARGET[2]:.0f})")
    ax_xz.plot([SPAWN_CX, TARGET[0]], [SPAWN_Z, TARGET[2]],
               "-", color="gray", alpha=0.5, linewidth=1.2, label="Spawn→target line")

    for pos, spec in zip(positions, SPECS):
        draw_obstacle_xz(ax_xz, spec, pos, color, 0.55)

    ax_xz.set_xlim(CORRIDOR_X_RANGE[0] - 1, CORRIDOR_X_RANGE[1] + 1)
    ax_xz.set_ylim(-1, CORRIDOR_Z_RANGE[1] + 1.5)
    ax_xz.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = f"prototyp_corridor_navigation/obstacle_setup_corridor_sample{sample_idx + 1}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


for idx, (positions, color) in enumerate(zip(sample_positions, sample_colors)):
    draw_sample_figure(idx, positions, color)


# ------------------------------------------------------------------
# Numerical stats (10 k samples)
# ------------------------------------------------------------------

print("\n=== Placement-formula sanity (10 000 samples) ===")
n_tests = 10_000
gap_samples = []
d_samples = []
axis_counts = {0: 0, 1: 0}
axis_by_slice = {s: {0: 0, 1: 0} for s in range(N_SLICES)}  # N_SLICES-keyed

for _ in range(n_tests):
    positions = place_corridor()
    for slice_i in range(N_SLICES):
        idx_a, idx_b = 2 * slice_i, 2 * slice_i + 1
        spec_a = SPECS[idx_a]
        spec_b = SPECS[idx_b]
        pa = positions[idx_a]
        pb = positions[idx_b]
        y_line_a, z_line_a = line_yz_at_x(pa[0])

        dy_a = pa[1] - y_line_a
        dz_a = pa[2] - z_line_a
        if abs(dy_a) > abs(dz_a):
            axis = 0
            d = abs(dy_a)
            face_gap = abs(pa[1] - pb[1]) - (spec_a["half_y"] + spec_b["half_y"])
        else:
            axis = 1
            d = abs(dz_a)
            face_gap = abs(pa[2] - pb[2]) - (spec_a["half_z"] + spec_b["half_z"])
        gap_samples.append(face_gap)
        d_samples.append(d)
        axis_counts[axis] += 1
        axis_by_slice[slice_i][axis] += 1

gap_samples = np.array(gap_samples)
d_samples = np.array(d_samples)

print(f"face_gap min/mean/max:  {gap_samples.min():.3f} / {gap_samples.mean():.3f} / {gap_samples.max():.3f} m  (expected {PAIR_GAP_RANGE})")
print(f"distance min/mean/max:  {d_samples.min():.3f} / {d_samples.mean():.3f} / {d_samples.max():.3f} m  (expected {FIRST_OFFSET_RANGE})")
total = sum(axis_counts.values())
print(f"Axis split overall:     Y={axis_counts[0] / total:.1%}, Z={axis_counts[1] / total:.1%}")
for s in range(N_SLICES):
    tot = sum(axis_by_slice[s].values())
    y_pct = axis_by_slice[s][0] / tot
    z_pct = axis_by_slice[s][1] / tot
    pillar = "PILLAR" if PAIR_HAS_PILLAR[s] else "      "
    print(f"  slice {s} ({pillar}):   Y={y_pct:.1%}, Z={z_pct:.1%}")

# Assertions
assert PAIR_GAP_RANGE[0] - 1e-6 <= gap_samples.min(), "face_gap below min"
assert gap_samples.max() <= PAIR_GAP_RANGE[1] + 1e-6, "face_gap above max"
assert FIRST_OFFSET_RANGE[0] - 1e-6 <= d_samples.min(), "d below min"
assert d_samples.max() <= FIRST_OFFSET_RANGE[1] + 1e-6, "d above max"
for s in range(N_SLICES):
    if PAIR_HAS_PILLAR[s]:
        assert axis_by_slice[s][1] == 0, f"slice {s} has pillar but Z-axis draws occurred"
print("*** all formula invariants hold ***")
