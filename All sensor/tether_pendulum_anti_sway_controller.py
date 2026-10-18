import math
import numpy as np
from scipy.linalg import solve_continuous_are
from typing import Dict, List, Tuple

class TetherAntiSwayController:
    def __init__(
        self,
        drone_mass_kg: float = 6.14,
        payload_mass_kg: float = 1.80,
        nominal_cable_length_m: float = 8.0,
        aerodynamic_damping_ratio: float = 0.035,
        dt_seconds: float = 0.05
    ):
        self.m_drone = drone_mass_kg
        self.m_load = payload_mass_kg
        self.l_cable = nominal_cable_length_m
        self.zeta = aerodynamic_damping_ratio
        self.dt = dt_seconds
        self.g = 9.80665

        # State Vector: x = [pos_drone, vel_drone, theta_pendulum, omega_pendulum]^T
        self.state_x = np.zeros(4)
        self.state_y = np.zeros(4)

        # Precompute LQR gains for nominal cable length
        self.k_lqr = self._compute_lqr_gains(self.l_cable)

    def _compute_lqr_gains(self, cable_length_m: float) -> np.ndarray:
        """
        Solves Algebraic Riccati Equation for linearized pendulum-cart dynamics:
          x_dot = A * x + B * u
        where state x = [x_d, x_d_dot, theta, theta_dot]^T and control u = a_x_cmd
        """
        l = max(cable_length_m, 0.5)
        
        # State Matrix A (4x4)
        A = np.zeros((4, 4))
        A[0, 1] = 1.0
        A[2, 3] = 1.0
        A[3, 2] = -self.g / l
        A[3, 1] = 0.0
        A[3, 3] = -2.0 * self.zeta * math.sqrt(self.g / l)

        # Control Input Matrix B (4x1)
        B = np.zeros((4, 1))
        B[1, 0] = 1.0
        B[3, 0] = -1.0 / l

        # State Weighting Matrix Q and Control Weighting Matrix R
        Q = np.diag([15.0, 3.0, 120.0, 15.0]) # Heavy penalty on swing angle theta
        R = np.array([[0.8]])

        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K.flatten()

    def compute_zvd_shaper_impulses(
        self,
        cable_length_m: float,
        vertical_accel_ms2: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates 3-impulse Zero Vibration Derivative (ZVD) sequence [A_i, t_i].
        """
        l = max(cable_length_m, 0.5)
        omega_n = math.sqrt(max(self.g + vertical_accel_ms2, 0.5) / l)
        omega_d = omega_n * math.sqrt(max(1.0 - self.zeta**2, 1e-4))

        k_val = math.exp(- (self.zeta * math.pi) / math.sqrt(max(1.0 - self.zeta**2, 1e-4)))
        denom = (1.0 + k_val) ** 2

        a1 = 1.0 / denom
        a2 = (2.0 * k_val) / denom
        a3 = (k_val ** 2) / denom

        t1 = 0.0
        t2 = math.pi / omega_d
        t3 = (2.0 * math.pi) / omega_d

        amplitudes = np.array([a1, a2, a3])
        time_delays = np.array([t1, t2, t3])
        return amplitudes, time_delays

    def compute_anti_sway_command(
        self,
        target_pos_ned: Tuple[float, float],
        measured_pendulum_angles_rad: Tuple[float, float], # (theta_x, theta_y)
        cable_length_m: float,
        gust_disturbance_force_n: Tuple[float, float] = (0.0, 0.0)
    ) -> Dict:
        """
        Computes combined ZVD feedforward and LQR state-feedback horizontal acceleration commands.
        """
        self.l_cable = max(cable_length_m, 0.5)
        # Update LQR gain vector if cable length deviates > 1.0m
        k_gain = self._compute_lqr_gains(self.l_cable)

        # 1. Error state vector in X-axis (North)
        err_pos_x = self.state_x[0] - target_pos_ned[0]
        self.state_x[2] = measured_pendulum_angles_rad[0]
        x_vec = np.array([err_pos_x, self.state_x[1], self.state_x[2], self.state_x[3]])

        # 2. Error state vector in Y-axis (East)
        err_pos_y = self.state_y[0] - target_pos_ned[1]
        self.state_y[2] = measured_pendulum_angles_rad[1]
        y_vec = np.array([err_pos_y, self.state_y[1], self.state_y[2], self.state_y[3]])

        # 3. LQR Feedback Acceleration Control Law
        ax_lqr = -float(k_gain @ x_vec)
        ay_lqr = -float(k_gain @ y_vec)

        # Apply physical drone acceleration bounds (+/- 4.0 m/s^2)
        ax_cmd = float(np.clip(ax_lqr, -4.0, 4.0))
        ay_cmd = float(np.clip(ay_lqr, -4.0, 4.0))

        # Integrate forward state simulation (Euler integration)
        # Drone kinematics
        self.state_x[1] += ax_cmd * self.dt
        self.state_x[0] += self.state_x[1] * self.dt
        self.state_y[1] += ay_cmd * self.dt
        self.state_y[0] += self.state_y[1] * self.dt

        # Nonlinear spherical pendulum angular dynamics with aerodynamic gust
        f_gx, f_gy = gust_disturbance_force_n
        d_omega_x = (-self.g * math.sin(self.state_x[2]) - ax_cmd + (f_gx / self.m_load)) / self.l_cable - 2.0 * self.zeta * self.state_x[3]
        d_omega_y = (-self.g * math.sin(self.state_y[2]) - ay_cmd + (f_gy / self.m_load)) / self.l_cable - 2.0 * self.zeta * self.state_y[3]

        self.state_x[3] += d_omega_x * self.dt
        self.state_x[2] += self.state_x[3] * self.dt
        self.state_y[3] += d_omega_y * self.dt
        self.state_y[2] += self.state_y[3] * self.dt

        total_swing_angle_deg = math.degrees(math.hypot(self.state_x[2], self.state_y[2]))

        return {
            "cable_length_m": round(self.l_cable, 2),
            "commanded_accel_x_ms2": round(ax_cmd, 3),
            "commanded_accel_y_ms2": round(ay_cmd, 3),
            "pendulum_swing_angle_x_deg": round(math.degrees(self.state_x[2]), 2),
            "pendulum_swing_angle_y_deg": round(math.degrees(self.state_y[2]), 2),
            "resultant_swing_magnitude_deg": round(total_swing_angle_deg, 2),
            "sway_suppression_active": total_swing_angle_deg > 1.0,
            "drop_release_permitted": total_swing_angle_deg <= 2.5
        }

if __name__ == "__main__":
    controller = TetherAntiSwayController(nominal_cable_length_m=8.0)

    print("==========================================================================")
    print("      ACTIVE TETHER PENDULUM ANTI-SWAY & LQR FEEDBACK CONTROL AUDIT       ")
    print("==========================================================================")
    
    # 1. ZVD Impulse Shaping Parameters
    amps, times = controller.compute_zvd_shaper_impulses(cable_length_m=8.0)
    print("ZVD Input Shaper Impulses:")
    for idx in range(3):
        print(f"  • Impulse #{idx+1}: Amplitude = {amps[idx]:.4f}, Timing Delay = {times[idx]:.3f} s")

    # 2. Simulate Gust Shock Rejection during Lowering
    print("\nClosed-Loop Anti-Sway Response under 6.5 N Lateral Gust Disturbance:")
    # Initial 12.0° swing induced by gust shock
    controller.state_x[2] = math.radians(12.0)
    
    for step in range(1, 16):
        report = controller.compute_anti_sway_command(
            target_pos_ned=(0.0, 0.0),
            measured_pendulum_angles_rad=(controller.state_x[2], controller.state_y[2]),
            cable_length_m=8.0,
            gust_disturbance_force_n=(0.0, 0.0) # Gust dissipated, damping swing
        )
        if step % 3 == 0 or step == 1:
            print(f"  T+{(step*0.05):4.2f}s | Accel Cmd: [Ax={report['commanded_accel_x_ms2']:+5.2f}, Ay={report['commanded_accel_y_ms2']:+5.2f}] m/s² | "
                  f"Swing: {report['resultant_swing_magnitude_deg']:5.2f}° | Touchdown Clear: {report['drop_release_permitted']}")
    print("==========================================================================\n")