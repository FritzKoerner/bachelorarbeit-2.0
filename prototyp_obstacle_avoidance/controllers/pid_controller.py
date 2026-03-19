from typing import TYPE_CHECKING

import torch
import numpy as np

from genesis.utils.geom import quat_to_xyz

if TYPE_CHECKING:
    from genesis.engine.entities.drone_entity import DroneEntity


class VectorizedPIDController:
    """Per-env PID controller using batched torch tensors.

    All state (integral, prev_measurement) lives in (n_envs,) tensors so
    individual environments can be reset independently via reset_idx().

    Features:
    - Derivative-on-measurement (avoids derivative kick on setpoint changes)
    - EMA-filtered derivative (reduces noise amplification)
    - Integral anti-windup via conditional integration (clamping)
    """

    def __init__(self, kp: float, ki: float, kd: float, n_envs: int, device,
                 deriv_filter_alpha: float = 0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.deriv_filter_alpha = deriv_filter_alpha
        self.integral = torch.zeros(n_envs, device=device, dtype=torch.float32)
        self.prev_measurement = torch.zeros(n_envs, device=device, dtype=torch.float32)
        self.filtered_derivative = torch.zeros(n_envs, device=device, dtype=torch.float32)

    def update(self, setpoint, measurement, dt: float, output_limit=None):
        """
        Args:
            setpoint:     (n_envs,) tensor or scalar
            measurement:  (n_envs,) tensor
            dt:           float
            output_limit: optional float; clamps output and freezes integral
                          when saturated (anti-windup)
        Returns:
            output: (n_envs,) tensor
        """
        error = setpoint - measurement

        # EMA-filtered derivative-on-measurement
        raw_derivative = -(measurement - self.prev_measurement) / dt
        a = self.deriv_filter_alpha
        self.filtered_derivative = a * raw_derivative + (1 - a) * self.filtered_derivative
        self.prev_measurement = measurement.clone()

        # Tentatively accumulate integral
        new_integral = self.integral + error * dt

        output = self.kp * error + self.ki * new_integral + self.kd * self.filtered_derivative

        if output_limit is not None:
            clamped = torch.clamp(output, -output_limit, output_limit)
            # Anti-windup: only update integral where output is NOT saturated
            saturated = (output != clamped)
            self.integral = torch.where(saturated, self.integral, new_integral)
            output = clamped
        else:
            self.integral = new_integral

        return output

    def reset_idx(self, env_ids):
        self.integral[env_ids] = 0.0
        self.prev_measurement[env_ids] = 0.0
        self.filtered_derivative[env_ids] = 0.0

    def reset(self):
        self.integral.zero_()
        self.prev_measurement.zero_()
        self.filtered_derivative.zero_()


class PIDController:
    def __init__(self, kp, ki, kd, debug=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_measurement = 0.0
        self.debug = debug
        self._history = []

    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement

        self.integral += error * dt

        derivative = - (measurement - self.prev_measurement) / dt
        self.prev_measurement = measurement

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.debug:
            self._history.append({
                "error": float(error) if not isinstance(error, float) else error,
                "integral": float(self.integral) if not isinstance(self.integral, float) else self.integral,
                "derivative": float(derivative) if not isinstance(derivative, float) else derivative,
                "output": float(output) if not isinstance(output, float) else output,
                "setpoint": float(setpoint) if not isinstance(setpoint, float) else setpoint,
                "measurement": float(measurement) if not isinstance(measurement, float) else measurement,
            })

        return output

    def reset(self):
        self.integral = 0.0
        self.prev_measurement = 0.0
        self._history.clear()


class DronePIDController:
    def __init__(self, drone: "DroneEntity", dt, base_rpm, max_rpm, pid_params):
        self.__pid_vel_x = PIDController(*pid_params["pid_params_vel_x"])
        self.__pid_vel_y = PIDController(*pid_params["pid_params_vel_y"])
        self.__pid_vel_z = PIDController(*pid_params["pid_params_vel_z"])

        self.__pid_att_roll = PIDController(*pid_params["pid_params_roll"])
        self.__pid_att_pitch = PIDController(*pid_params["pid_params_pitch"])
        self.__pid_att_yaw = PIDController(*pid_params["pid_params_yaw"])

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm
        self.__max_rpm = max_rpm

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()

    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        return quat_to_xyz(quat, rpy=True, degrees=True)

    def __mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel) -> torch.Tensor:
        """
        Motor mixer for X-configuration quadcopter (draugas URDF).

        Motor positions derived from URDF prop joint origins:
            prop0: xyz=(-0.074, -0.074, ...)  → back-right   (−X, −Y)
            prop1: xyz=(+0.074, -0.074, ...)  → front-right  (+X, −Y)
            prop2: xyz=(+0.074, +0.074, ...)  → front-left   (+X, +Y)
            prop3: xyz=(-0.074, +0.074, ...)  → back-left    (−X, +Y)

        Spin directions (from env config propellers_spin=[1, -1, 1, -1]):
            prop0: CW (+1)    prop1: CCW (-1)
            prop2: CW (+1)    prop3: CCW (-1)

        Sign derivation per axis:
            roll  (torque around X-axis) → split along Y:
                +Y motors (left:  prop2, prop3) get +roll
                −Y motors (right: prop0, prop1) get −roll
            pitch (torque around Y-axis) → split along X:
                −X motors (back:  prop0, prop3) get +pitch
                +X motors (front: prop1, prop2) get −pitch
            yaw   (reaction torque) → opposite to spin direction:
                CW  motors (prop0, prop2) get −yaw
                CCW motors (prop1, prop3) get +yaw
            x_vel → same sign pattern as pitch (tilt to accelerate in X)
            y_vel → opposite sign pattern to roll (tilt to accelerate in Y)
        """
        M1 = self.__base_rpm + (thrust - roll + pitch + yaw + x_vel - y_vel)  # prop0: back-right,  CW
        M2 = self.__base_rpm + (thrust - roll - pitch - yaw - x_vel - y_vel)  # prop1: front-right, CCW
        M3 = self.__base_rpm + (thrust + roll - pitch + yaw - x_vel + y_vel)  # prop2: front-left,  CW
        M4 = self.__base_rpm + (thrust + roll + pitch - yaw + x_vel + y_vel)  # prop3: back-left,   CCW
        rpms = torch.Tensor([M1, M2, M3, M4])
        return torch.clamp(rpms, 0, self.__max_rpm)


    def update(self, target_velocity, target_body_rates) -> np.ndarray:
        curr_vel = self.__get_drone_vel()
        curr_att = self.__get_drone_att()

        error_vel_x = target_velocity[:, 0] - curr_vel[:, 0]
        error_vel_y = target_velocity[:, 1] - curr_vel[:, 1]
        error_vel_z = target_velocity[:, 2] - curr_vel[:, 2]

        x_vel_del = self.__pid_vel_x.update(error_vel_x, self.__dt)
        y_vel_del = self.__pid_vel_y.update(error_vel_y, self.__dt)
        thrust_des = self.__pid_vel_z.update(error_vel_z, self.__dt)

        err_roll = target_body_rates[:, 0] - curr_att[:, 0]
        err_pitch = target_body_rates[:, 1] - curr_att[:, 1]
        err_yaw = target_body_rates[:, 2] - curr_att[:, 2]

        roll_del = self.__pid_att_roll.update(err_roll, self.__dt)
        pitch_del = self.__pid_att_pitch.update(err_pitch, self.__dt)
        yaw_del = self.__pid_att_yaw.update(err_yaw, self.__dt)

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)

        return prop_rpms.cpu().numpy()


    def reset(self):
        self.__pid_vel_x.reset()
        self.__pid_vel_y.reset()
        self.__pid_vel_z.reset()
        self.__pid_att_roll.reset()
        self.__pid_att_pitch.reset()
        self.__pid_att_yaw.reset()

    def sanity_check(self):
        tests = [
            # (name, args, expected behavior)
            ("Thrust",  dict(thrust=100, roll=0, pitch=0, yaw=0, x_vel=0, y_vel=0),
             "All motors equal: +100"),
            ("Roll",    dict(thrust=0, roll=100, pitch=0, yaw=0, x_vel=0, y_vel=0),
             "FL,BL > 0 | FR,BR < 0  (left up, tilt right)"),
            ("Pitch",   dict(thrust=0, roll=0, pitch=100, yaw=0, x_vel=0, y_vel=0),
             "BL,BR > 0 | FL,FR < 0  (back up, nose down)"),
            ("Yaw",     dict(thrust=0, roll=0, pitch=0, yaw=100, x_vel=0, y_vel=0),
             "CW motors(FL,BR) < 0 | CCW motors(FR,BL) > 0  (yaw CW)"),
            ("X_vel",   dict(thrust=0, roll=0, pitch=0, yaw=0, x_vel=100, y_vel=0),
             "Same as pitch: BL,BR > 0 | FL,FR < 0  (accelerate +X)"),
            ("Y_vel",   dict(thrust=0, roll=0, pitch=0, yaw=0, x_vel=0, y_vel=100),
             "Opposite roll: FR,BR > 0 | FL,BL < 0  (accelerate +Y/left)"),
        ]
        base = self._DronePIDController__base_rpm
        for name, args, expected in tests:
            rpms = self._DronePIDController__mixer(**args)
            deltas = rpms - base  # show delta from base_rpm
            print(f"\n{name}:")
            print(f"  FL={deltas[0]:+.0f}  FR={deltas[1]:+.0f}  BR={deltas[2]:+.0f}  BL={deltas[3]:+.0f}")
            print(f"  Expected: {expected}")


if __name__ == "__main__":
    pid_params = {
        "pid_params_vel_x": [1.0, 0.0, 0.1],
        "pid_params_vel_y": [1.0, 0.0, 0.1],
        "pid_params_vel_z": [2.0, 0.0, 0.1],
        "pid_params_roll":  [0.5, 0.0, 0.1],
        "pid_params_pitch": [0.5, 0.0, 0.1],
        "pid_params_yaw":   [0.5, 0.0, 0.1],
    }
    ctrl = DronePIDController(
        drone=None,       # not needed for mixer test
        dt=0.01,
        base_rpm=1789.2,
        max_rpm=5000.0,
        pid_params=pid_params,
    )
    ctrl.sanity_check()


class PositionPIDController:
    def __init__(self, drone: "DroneEntity", dt, base_rpm, max_rpm, pid_params):
        self.__pid_pos_x = PIDController(*pid_params["pid_params_pos_x"])
        self.__pid_pos_y = PIDController(*pid_params["pid_params_pos_y"])
        self.__pid_pos_z = PIDController(*pid_params["pid_params_pos_z"])

        self.__pid_vel_x = PIDController(*pid_params["pid_params_vel_x"])
        self.__pid_vel_y = PIDController(*pid_params["pid_params_vel_y"])
        self.__pid_vel_z = PIDController(*pid_params["pid_params_vel_z"])

        self.__pid_att_roll = PIDController(*pid_params["pid_params_roll"])
        self.__pid_att_pitch = PIDController(*pid_params["pid_params_pitch"])
        self.__pid_att_yaw = PIDController(*pid_params["pid_params_yaw"])

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm
        self.__max_rpm = max_rpm

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()

    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        return quat_to_xyz(quat, rpy=True, degrees=True)

    def __mixer(self, thrust, roll, pitch, yaw, x_vel, y_vel) -> torch.Tensor:
        M1 = self.__base_rpm + (thrust - roll + pitch + yaw + x_vel - y_vel)  # prop0: back-right,  CW
        M2 = self.__base_rpm + (thrust - roll - pitch - yaw - x_vel - y_vel)  # prop1: front-right, CCW
        M3 = self.__base_rpm + (thrust + roll - pitch + yaw - x_vel + y_vel)  # prop2: front-left,  CW
        M4 = self.__base_rpm + (thrust + roll + pitch - yaw + x_vel + y_vel)  # prop3: back-left,   CCW
        rpms = torch.Tensor([M1, M2, M3, M4])
        return torch.clamp(rpms, 0, self.__max_rpm)

    def update(self, target) -> np.ndarray:
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_att = self.__get_drone_att()

        print("Current Position", curr_pos)
        print("Target Position", target)

        err_pos_x = target[:, 0] - curr_pos[:, 0]
        err_pos_y = target[:, 1] - curr_pos[:, 1]
        err_pos_z = target[:, 2] - curr_pos[:, 2]

        vel_des_x = self.__pid_pos_x.update(target[:, 0], curr_pos[:, 0], self.__dt)
        vel_des_y = self.__pid_pos_y.update(target[:, 1], curr_pos[:, 1], self.__dt)
        vel_des_z = self.__pid_pos_z.update(target[:, 2], curr_pos[:, 2], self.__dt)

        error_vel_x = vel_des_x - curr_vel[:, 0]
        error_vel_y = vel_des_y - curr_vel[:, 1]
        error_vel_z = vel_des_z - curr_vel[:, 2]

        x_vel_del = self.__pid_vel_x.update(vel_des_x, curr_vel[:, 0], self.__dt)
        y_vel_del = self.__pid_vel_y.update(vel_des_y, curr_vel[:, 1], self.__dt)
        thrust_des = self.__pid_vel_z.update(vel_des_z, curr_vel[:, 2], self.__dt)

        err_roll = 0 - curr_att[:, 0]
        err_pitch = 0 - curr_att[:, 1]
        err_yaw = 0 - curr_att[:, 2]

        roll_del = self.__pid_att_roll.update(0, curr_att[:, 0], self.__dt)
        pitch_del = self.__pid_att_pitch.update(0, curr_att[:, 1], self.__dt)
        yaw_del = self.__pid_att_yaw.update(0, curr_att[:, 2], self.__dt)

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)

        return prop_rpms.cpu().numpy()

    def reset(self):
        self.__pid_pos_x.reset()
        self.__pid_pos_y.reset()
        self.__pid_pos_z.reset()
        self.__pid_vel_x.reset()
        self.__pid_vel_y.reset()
        self.__pid_vel_z.reset()
        self.__pid_att_roll.reset()
        self.__pid_att_pitch.reset()
        self.__pid_att_yaw.reset()


class CascadingPIDController:
    """
    Proper cascading PID: Position PID → Velocity PID → Attitude PID → Mixer.

    Position PID outputs desired velocities.
    Velocity PID (X/Y) outputs desired roll/pitch angles; Velocity PID (Z) outputs thrust.
    Attitude PID tracks those desired angles and outputs corrections to the mixer.

    Supports vectorized (multi-env) operation when n_envs > 1.  All nine inner
    PIDs use VectorizedPIDController so per-env state can be reset independently
    via reset_idx().  update() returns a (n_envs, 4) GPU tensor of RPMs.
    """

    def __init__(self, drone: "DroneEntity", dt, base_rpm, max_rpm, pid_params,
                 n_envs: int = 1, device=None, debug: bool = False):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Outer loop: position → desired velocity
        self.__pid_pos_x = VectorizedPIDController(*pid_params["pid_params_pos_x"], n_envs, device)
        self.__pid_pos_y = VectorizedPIDController(*pid_params["pid_params_pos_y"], n_envs, device)
        self.__pid_pos_z = VectorizedPIDController(*pid_params["pid_params_pos_z"], n_envs, device)

        # Middle loop: velocity → desired attitude (roll/pitch) + thrust
        self.__pid_vel_x = VectorizedPIDController(*pid_params["pid_params_vel_x"], n_envs, device)
        self.__pid_vel_y = VectorizedPIDController(*pid_params["pid_params_vel_y"], n_envs, device)
        self.__pid_vel_z = VectorizedPIDController(*pid_params["pid_params_vel_z"], n_envs, device)

        # Inner loop: attitude → corrections for mixer
        self.__pid_att_roll  = VectorizedPIDController(*pid_params["pid_params_roll"],  n_envs, device)
        self.__pid_att_pitch = VectorizedPIDController(*pid_params["pid_params_pitch"], n_envs, device)
        self.__pid_att_yaw   = VectorizedPIDController(*pid_params["pid_params_yaw"],   n_envs, device)

        self.drone = drone
        self.__dt = dt
        self.__base_rpm = base_rpm
        self.__max_rpm = max_rpm

        # Max tilt angle (degrees) the velocity PID can command
        self.__max_tilt = pid_params.get("max_tilt", 30.0)
        # Max velocity (m/s) the position PID can command
        self.__max_vel_xy = pid_params.get("max_vel_xy", 5.0)
        self.__max_vel_z = pid_params.get("max_vel_z", 3.0)

    def __get_drone_pos(self) -> torch.Tensor:
        return self.drone.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.drone.get_vel()

    def __get_drone_att(self) -> torch.Tensor:
        quat = self.drone.get_quat()
        return quat_to_xyz(quat, rpy=True, degrees=True)

    def __mixer(self, thrust, roll, pitch, yaw) -> torch.Tensor:
        M1 = self.__base_rpm + (thrust - roll + pitch + yaw)  # prop0: back-right,  CW
        M2 = self.__base_rpm + (thrust - roll - pitch - yaw)  # prop1: front-right, CCW
        M3 = self.__base_rpm + (thrust + roll - pitch + yaw)  # prop2: front-left,  CW
        M4 = self.__base_rpm + (thrust + roll + pitch - yaw)  # prop3: back-left,   CCW
        # stack along last dim → (n_envs, 4)
        rpms = torch.stack([M1, M2, M3, M4], dim=-1)
        return torch.clamp(rpms, 0, self.__max_rpm)

    def update(self, target_pos, target_yaw=0.0) -> torch.Tensor:
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_att = self.__get_drone_att()

        # --- Outer loop: Position PID → desired velocity ---
        vel_des_x = self.__pid_pos_x.update(
            target_pos[:, 0], curr_pos[:, 0], self.__dt,
            output_limit=self.__max_vel_xy,
        )
        vel_des_y = self.__pid_pos_y.update(
            target_pos[:, 1], curr_pos[:, 1], self.__dt,
            output_limit=self.__max_vel_xy,
        )
        vel_des_z = self.__pid_pos_z.update(
            target_pos[:, 2], curr_pos[:, 2], self.__dt,
            output_limit=self.__max_vel_z,
        )

        # --- Middle loop: Velocity PID → desired attitude + thrust ---
        accel_x = self.__pid_vel_x.update(vel_des_x, curr_vel[:, 0], self.__dt)
        accel_y = self.__pid_vel_y.update(vel_des_y, curr_vel[:, 1], self.__dt)
        thrust = self.__pid_vel_z.update(vel_des_z, curr_vel[:, 2], self.__dt)

        # Velocity-to-attitude mapping (derived from URDF + mixer signs):
        #   +pitch → back motors up, nose down → +X accel  ⇒  des_pitch = +accel_x
        #   +roll  → left motors up, tilt right → -Y accel ⇒  des_roll  = -accel_y
        des_pitch = torch.clamp(accel_x, -self.__max_tilt, self.__max_tilt)
        des_roll = torch.clamp(-accel_y, -self.__max_tilt, self.__max_tilt)

        # --- Inner loop: Attitude PID → corrections for mixer ---
        roll_corr = self.__pid_att_roll.update(des_roll, curr_att[:, 0], self.__dt)
        pitch_corr = self.__pid_att_pitch.update(des_pitch, curr_att[:, 1], self.__dt)
        yaw_corr = self.__pid_att_yaw.update(target_yaw, curr_att[:, 2], self.__dt)

        # --- Mixer ---
        prop_rpms = self.__mixer(thrust, roll_corr, pitch_corr, yaw_corr)

        return prop_rpms  # (n_envs, 4) GPU tensor

    def reset_idx(self, env_ids):
        """Reset PID state for specific environments (e.g. after crash/timeout)."""
        for pid in (self.__pid_pos_x, self.__pid_pos_y, self.__pid_pos_z,
                    self.__pid_vel_x, self.__pid_vel_y, self.__pid_vel_z,
                    self.__pid_att_roll, self.__pid_att_pitch, self.__pid_att_yaw):
            pid.reset_idx(env_ids)

    def reset(self):
        self.__pid_pos_x.reset()
        self.__pid_pos_y.reset()
        self.__pid_pos_z.reset()
        self.__pid_vel_x.reset()
        self.__pid_vel_y.reset()
        self.__pid_vel_z.reset()
        self.__pid_att_roll.reset()
        self.__pid_att_pitch.reset()
        self.__pid_att_yaw.reset()

    def plot_debug(self, save_path=None):
        """Not available with VectorizedPIDController (no per-step history recorded)."""
        print("plot_debug() is not supported with VectorizedPIDController.")
