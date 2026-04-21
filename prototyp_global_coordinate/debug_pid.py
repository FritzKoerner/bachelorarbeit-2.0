"""PID controller debug script — no RL, just fixed-target tracking.

Sends the drone to a fixed target position using the CascadingPIDController,
records state over time, and plots tracking performance.

Usage:
    cd prototyp_global_coordinate
    python debug_pid.py                              # default target (3, 3, 3), viewer on
    python debug_pid.py --target 5 5 2               # custom target
    python debug_pid.py --target_yaw 90              # track 90 deg yaw
    python debug_pid.py --no_viewer --steps 3000     # headless, more steps
    python debug_pid.py --yaw_test                   # yaw wrapping boundary test
"""

import argparse
import torch
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz

from controllers.pid_controller import CascadingPIDController


def run_yaw_test(args):
    """Yaw wrapping boundary test.

    Holds position at (0, 0, 3) and steps through yaw targets that exercise
    the +-180 deg discontinuity:  0 → 90 → 170 → -170 → 0.

    The 170 → -170 transition is the critical case: the physical shortest
    path is +20 deg, but without angular wrapping the PID sees -340 deg
    and commands a full rotation in the wrong direction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = 0.01
    n_envs = 1
    steps_per_target = 800  # 8 seconds per target

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(0, -8, 5),
            camera_lookat=(0, 0, 3),
            camera_fov=60,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=not args.no_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        gs.morphs.Drone(
            file="../assets/robots/draugas/draugas_genesis.urdf",
            pos=(0, 0, 3),
            euler=(0, 0, 0),
            propellers_link_name=["prop0_link", "prop1_link", "prop2_link", "prop3_link"],
            propellers_spin=[1, -1, 1, -1],
        )
    )
    scene.build(n_envs=n_envs, env_spacing=(40.0, 40.0))

    pid_params = {
        "base_rpm": 1789.2, "max_rpm": 2700.0,
        "max_tilt": 30.0, "max_vel_xy": 5.0, "max_vel_z": 3.0,
        "pid_params_pos_x": [1.0, 0.0, 0.7],
        "pid_params_pos_y": [1.0, 0.0, 0.7],
        "pid_params_pos_z": [1.5, 0.0, 1.0],
        "pid_params_vel_x": [16.0, 0.0, 8.0],
        "pid_params_vel_y": [16.0, 0.0, 8.0],
        "pid_params_vel_z": [100.0, 2.0, 10.0],
        "pid_params_roll":  [6.0, 0.0, 3.0],
        "pid_params_pitch": [6.0, 0.0, 3.0],
        "pid_params_yaw":   [0.5, 0.0, 0.8],
    }
    controller = CascadingPIDController(
        drone=drone, dt=dt,
        base_rpm=pid_params["base_rpm"], max_rpm=pid_params["max_rpm"],
        pid_params=pid_params, n_envs=n_envs, device=gs.device,
    )

    target_pos = torch.tensor([[0.0, 0.0, 3.0]], device=gs.device, dtype=torch.float32)
    yaw_targets = [0.0, 90.0, 170.0, -170.0, 0.0]

    history = {"yaw": [], "target_yaw": [], "roll": [], "pitch": [],
               "pos_z": [], "rpm_0": [], "rpm_1": [], "rpm_2": [], "rpm_3": []}

    print("Yaw wrapping boundary test")
    print(f"Yaw targets: {yaw_targets}  ({steps_per_target} steps each)")
    print(f"{'step':>5} | {'target':>7} | {'yaw':>7} | {'err':>7} | {'roll':>7} | {'pitch':>7} | {'z':>6}")
    print("-" * 65)

    for yi, yaw_deg in enumerate(yaw_targets):
        target_yaw = torch.tensor([yaw_deg], device=gs.device, dtype=torch.float32)
        for s in range(steps_per_target):
            global_step = yi * steps_per_target + s
            rpms = controller.update(target_pos, target_yaw)
            drone.set_propellels_rpm(rpms)
            scene.step()

            euler = quat_to_xyz(drone.get_quat(), rpy=True, degrees=True)[0].cpu()
            pos_z = drone.get_pos()[0, 2].item()
            r = rpms[0].cpu()

            history["yaw"].append(euler[2].item())
            history["target_yaw"].append(yaw_deg)
            history["roll"].append(euler[0].item())
            history["pitch"].append(euler[1].item())
            history["pos_z"].append(pos_z)
            history["rpm_0"].append(r[0].item())
            history["rpm_1"].append(r[1].item())
            history["rpm_2"].append(r[2].item())
            history["rpm_3"].append(r[3].item())

            if s % 200 == 0:
                yaw_err = yaw_deg - euler[2].item()
                # show the wrapped error for comparison
                wrapped_err = ((yaw_err + 180) % 360) - 180
                print(f"{global_step:5d} | {yaw_deg:+7.1f} | {euler[2].item():+7.1f} "
                      f"| {wrapped_err:+7.1f} | {euler[0].item():+7.2f} "
                      f"| {euler[1].item():+7.2f} | {pos_z:6.2f}")

    # --- Plot ---
    t = np.arange(len(history["yaw"])) * dt
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Yaw Wrapping Boundary Test", fontsize=14, fontweight="bold")

    # Panel 1: yaw tracking
    ax = axes[0]
    ax.plot(t, history["yaw"], label="actual yaw", color="#3498db", linewidth=1.2)
    ax.step(t, history["target_yaw"], label="target yaw", color="#e74c3c",
            linewidth=1.2, linestyle="--", where="post")
    ax.axhline(180, color="gray", linestyle=":", alpha=0.4)
    ax.axhline(-180, color="gray", linestyle=":", alpha=0.4)
    # Mark the critical transition
    cross_step = 2 * steps_per_target
    ax.axvline(cross_step * dt, color="#f39c12", alpha=0.5, linestyle="-.",
               label="170 → -170 transition")
    ax.set_ylabel("Yaw (deg)")
    ax.set_title("Yaw Tracking (critical: 170° → -170° should be +20° shortest path)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: roll/pitch stability
    ax = axes[1]
    ax.plot(t, history["roll"], label="roll", color="#e74c3c", linewidth=0.9)
    ax.plot(t, history["pitch"], label="pitch", color="#2ecc71", linewidth=0.9)
    ax.plot(t, history["pos_z"], label="altitude", color="#3498db", linewidth=0.9)
    ax.axhline(3.0, color="#3498db", linestyle="--", alpha=0.3, label="target z=3m")
    ax.axvline(cross_step * dt, color="#f39c12", alpha=0.5, linestyle="-.")
    ax.set_ylabel("deg / m")
    ax.set_title("Stability during yaw maneuvers (roll, pitch, altitude)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Motor RPMs
    ax = axes[2]
    ax.plot(t, history["rpm_0"], label="M0 (BR, CW)", alpha=0.8)
    ax.plot(t, history["rpm_1"], label="M1 (FR, CCW)", alpha=0.8)
    ax.plot(t, history["rpm_2"], label="M2 (FL, CW)", alpha=0.8)
    ax.plot(t, history["rpm_3"], label="M3 (BL, CCW)", alpha=0.8)
    ax.axhline(1789.2, color="gray", linestyle="--", alpha=0.4, label="base RPM")
    ax.axvline(cross_step * dt, color="#f39c12", alpha=0.5, linestyle="-.")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.set_title("Motor RPMs (CW/CCW pairs should split symmetrically for pure yaw)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "pid_yaw_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PID controller debug (no RL)")
    parser.add_argument("--target", type=float, nargs=3, default=[3.0, 3.0, 3.0],
                        help="Target position [x, y, z]")
    parser.add_argument("--target_yaw", type=float, default=0.0,
                        help="Target yaw (degrees)")
    parser.add_argument("--spawn", type=float, nargs=3, default=[0.0, 0.0, 5.0],
                        help="Drone spawn position [x, y, z]")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of sim steps (at 100 Hz)")
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--save", type=str, default="pid_debug.png",
                        help="Output plot filename")
    parser.add_argument("--yaw_test", action="store_true",
                        help="Yaw wrapping boundary test: hover at (0,0,3) and "
                             "step through yaw targets [0, 90, 170, -170, 0] "
                             "to exercise the +-180 deg discontinuity")
    args = parser.parse_args()

    if args.yaw_test:
        run_yaw_test(args)
        return

    dt = 0.01  # 100 Hz
    n_envs = 1

    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(7.0, 0.0, 5.0),
            camera_lookat=(3.0, 3.0, 3.0),
            camera_fov=60,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=not args.no_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    # Target marker
    target_vis = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.15,
            fixed=True,
            collision=False,
            batch_fixed_verts=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5))
        ),
    )

    drone = scene.add_entity(
        gs.morphs.Drone(
            file="../assets/robots/draugas/draugas_genesis.urdf",
            pos=tuple(args.spawn),
            euler=(0, 0, 0),
            propellers_link_name=["prop0_link", "prop1_link", "prop2_link", "prop3_link"],
            propellers_spin=[1, -1, 1, -1],
        )
    )

    scene.build(n_envs=n_envs, env_spacing=(40.0, 40.0))

    # PID controller — same params as train_rl_wb.py
    pid_params = {
        "base_rpm": 1789.2,
        "max_rpm": 2700.0,
        "max_tilt": 30.0,
        "max_vel_xy": 5.0,
        "max_vel_z": 3.0,
        "pid_params_pos_x": [1.0, 0.0, 0.7],
        "pid_params_pos_y": [1.0, 0.0, 0.7],
        "pid_params_pos_z": [1.5, 0.0, 1.0],
        "pid_params_vel_x": [16.0, 0.0, 8.0],
        "pid_params_vel_y": [16.0, 0.0, 8.0],
        "pid_params_vel_z": [100.0, 2.0, 10.0],
        "pid_params_roll":  [6.0, 0.0, 3.0],
        "pid_params_pitch": [6.0, 0.0, 3.0],
        "pid_params_yaw":   [0.5, 0.0, 0.8],
    }

    controller = CascadingPIDController(
        drone=drone, dt=dt,
        base_rpm=pid_params["base_rpm"],
        max_rpm=pid_params["max_rpm"],
        pid_params=pid_params,
        n_envs=n_envs, device=gs.device,
    )

    target_pos = torch.tensor([args.target], device=gs.device, dtype=torch.float32)
    target_yaw = torch.tensor([args.target_yaw], device=gs.device, dtype=torch.float32)

    target_vis.set_pos(target_pos, zero_velocity=True)

    # Recording buffers
    history = {
        "pos_x": [], "pos_y": [], "pos_z": [],
        "vel_x": [], "vel_y": [], "vel_z": [],
        "roll": [], "pitch": [], "yaw": [],
        "dist": [],
        "rpm_0": [], "rpm_1": [], "rpm_2": [], "rpm_3": [],
    }

    print(f"Target: {args.target}, Spawn: {args.spawn}, Steps: {args.steps}")
    print(f"{'step':>5} | {'dist':>7} | {'x':>7} {'y':>7} {'z':>7} | {'vx':>7} {'vy':>7} {'vz':>7}")
    print("-" * 75)

    for step in range(args.steps):
        rpms = controller.update(target_pos, target_yaw)
        drone.set_propellels_rpm(rpms)
        scene.step()

        pos = drone.get_pos()     # (1, 3)
        vel = drone.get_vel()     # (1, 3)
        quat = drone.get_quat()   # (1, 4)
        euler = quat_to_xyz(quat, rpy=True, degrees=True)  # (1, 3)

        p = pos[0].cpu()
        v = vel[0].cpu()
        e = euler[0].cpu()
        r = rpms[0].cpu()
        d = torch.norm(pos[0] - target_pos[0]).item()

        history["pos_x"].append(p[0].item())
        history["pos_y"].append(p[1].item())
        history["pos_z"].append(p[2].item())
        history["vel_x"].append(v[0].item())
        history["vel_y"].append(v[1].item())
        history["vel_z"].append(v[2].item())
        history["roll"].append(e[0].item())
        history["pitch"].append(e[1].item())
        history["yaw"].append(e[2].item())
        history["dist"].append(d)
        history["rpm_0"].append(r[0].item())
        history["rpm_1"].append(r[1].item())
        history["rpm_2"].append(r[2].item())
        history["rpm_3"].append(r[3].item())

        if step % 100 == 0:
            print(f"{step:5d} | {d:7.3f} | {p[0]:7.3f} {p[1]:7.3f} {p[2]:7.3f} "
                  f"| {v[0]:7.3f} {v[1]:7.3f} {v[2]:7.3f}")

    print(f"\nFinal distance to target: {history['dist'][-1]:.4f} m")

    # --- Plotting ---
    plot_pid_debug(history, args.target, dt, args.save, target_yaw=args.target_yaw)


def plot_pid_debug(history, target, dt, save_path, target_yaw=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(history["dist"])) * dt

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"PID Debug — Target ({target[0]}, {target[1]}, {target[2]}), yaw={target_yaw:.0f}°",
                 fontsize=14, fontweight="bold")

    # 1. Position tracking (XYZ)
    ax = axes[0, 0]
    ax.plot(t, history["pos_x"], label="x", color="#e74c3c")
    ax.plot(t, history["pos_y"], label="y", color="#2ecc71")
    ax.plot(t, history["pos_z"], label="z", color="#3498db")
    ax.axhline(target[0], color="#e74c3c", linestyle="--", alpha=0.4, label=f"target x={target[0]}")
    ax.axhline(target[1], color="#2ecc71", linestyle="--", alpha=0.4, label=f"target y={target[1]}")
    ax.axhline(target[2], color="#3498db", linestyle="--", alpha=0.4, label=f"target z={target[2]}")
    ax.set_ylabel("Position (m)")
    ax.set_title("Position Tracking")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Distance to target
    ax = axes[0, 1]
    ax.plot(t, history["dist"], color="#8e44ad", linewidth=1.5)
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance to Target")
    ax.grid(True, alpha=0.3)

    # 3. Velocity
    ax = axes[1, 0]
    ax.plot(t, history["vel_x"], label="vx", color="#e74c3c")
    ax.plot(t, history["vel_y"], label="vy", color="#2ecc71")
    ax.plot(t, history["vel_z"], label="vz", color="#3498db")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Attitude (roll, pitch, yaw)
    ax = axes[1, 1]
    ax.plot(t, history["roll"],  label="roll",  color="#e74c3c")
    ax.plot(t, history["pitch"], label="pitch", color="#2ecc71")
    ax.plot(t, history["yaw"],   label="yaw",   color="#3498db")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    if target_yaw != 0.0:
        ax.axhline(target_yaw, color="#3498db", linestyle="--", alpha=0.5,
                    label=f"target yaw={target_yaw:.0f}°")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Attitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Motor RPMs
    ax = axes[2, 0]
    ax.plot(t, history["rpm_0"], label="M0 (BR)", alpha=0.8)
    ax.plot(t, history["rpm_1"], label="M1 (FR)", alpha=0.8)
    ax.plot(t, history["rpm_2"], label="M2 (FL)", alpha=0.8)
    ax.plot(t, history["rpm_3"], label="M3 (BL)", alpha=0.8)
    ax.axhline(1789.2, color="gray", linestyle="--", alpha=0.4, label="base RPM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RPM")
    ax.set_title("Motor RPMs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Position error (per axis)
    ax = axes[2, 1]
    err_x = np.array(history["pos_x"]) - target[0]
    err_y = np.array(history["pos_y"]) - target[1]
    err_z = np.array(history["pos_z"]) - target[2]
    ax.plot(t, err_x, label="err_x", color="#e74c3c")
    ax.plot(t, err_y, label="err_y", color="#2ecc71")
    ax.plot(t, err_z, label="err_z", color="#3498db")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_title("Position Error per Axis")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved -> {save_path}")


if __name__ == "__main__":
    main()
