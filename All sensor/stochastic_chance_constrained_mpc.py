import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, List, Tuple

class StochasticChanceConstrainedMPC:
    def __init__(
        self,
        horizon_steps: int = 10,
        dt_seconds: float = 0.1,
        total_mass_kg: float = 6.14,
        max_tilt_angle_rad: float = math.radians(35.0),
        max_thrust_n: float = 85.0,
        violation_probability_epsilon: float = 0.005 # 99.5% safety confidence
    ):
        self.np = horizon_steps
        self.dt = dt_seconds
        self.m = total_mass_kg
        self.g = 9.80665
        self.max_tilt = max_tilt_angle_rad
        self.max_t = max_thrust_n
        self.min_t = total_mass_kg * self.g * 0.3
        self.epsilon = violation_probability_epsilon

        # Quantile scaling factor: z_score = Phi^-1(1 - epsilon)
        self.z_score = float(norm.ppf(1.0 - self.epsilon))

        # Weighting matrices
        self.q_pos = np.diag([25.0, 25.0, 35.0])
        self.q_vel = np.diag([2.5, 2.5, 3.5])
        self.r_ctrl = np.diag([0.05, 0.05, 0.01])

        # Wind process noise covariance (turbulent gust variance)
        self.q_wind = np.diag([0.45**2, 0.45**2, 0.15**2]) # (m/s)^2

    def propagate_mean_and_covariance(
        self,
        state_mean: np.ndarray,
        state_cov: np.ndarray,
        control: np.ndarray,
        nominal_wind_ms: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates Gaussian state distribution N(x_hat, Sigma) through linearized translational dynamics:
          x_{k+1} = A * x_k + B * u_k + G * w_k
          Sigma_{k+1} = A * Sigma_k * A^T + G * Q_wind * G^T
        """
        pos = state_mean[0:3]
        vel = state_mean[3:6]
        pitch, roll, thrust = control[0], control[1], control[2]

        v_air = vel - nominal_wind_ms
        cd_parasite = 0.38
        a_front = 0.095
        rho = 1.225
        drag_force = 0.5 * rho * (v_air * np.linalg.norm(v_air)) * cd_parasite * a_front

        ax = (thrust / self.m) * math.sin(pitch) - (drag_force[0] / self.m)
        ay = -(thrust / self.m) * math.sin(roll) - (drag_force[1] / self.m)
        az = (thrust / self.m) * (math.cos(pitch) * math.cos(roll)) - self.g - (drag_force[2] / self.m)

        next_pos = pos + vel * self.dt + 0.5 * np.array([ax, ay, az]) * (self.dt ** 2)
        next_vel = vel + np.array([ax, ay, az]) * self.dt
        next_mean = np.concatenate([next_pos, next_vel])

        # State transition Jacobian A (6x6)
        A = np.eye(6)
        A[0:3, 3:6] = np.eye(3) * self.dt

        # Process noise mapping G (6x3)
        G = np.zeros((6, 3))
        G[0:3, 0:3] = 0.5 * (self.dt ** 2) * np.eye(3)
        G[3:6, 0:3] = self.dt * np.eye(3)

        next_cov = A @ state_cov @ A.T + G @ self.q_wind @ G.T
        return next_mean, next_cov

    def solve_chance_constrained_controls(
        self,
        current_state_mean: np.ndarray,
        current_state_cov: np.ndarray,
        reference_trajectory: np.ndarray,
        nominal_wind_ms: np.ndarray,
        geofence_half_width_m: float = 15.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solves CC-MPC: Penalizes deviation from trajectory while tightening corridor bounds
        to ensure state remains inside geofence with probability >= 1 - epsilon.
        """
        hover_thrust = self.m * self.g
        u_init = np.zeros(self.np * 3)
        u_init[2::3] = hover_thrust

        bounds = []
        for _ in range(self.np):
            bounds.append((-self.max_tilt, self.max_tilt))
            bounds.append((-self.max_tilt, self.max_tilt))
            bounds.append((self.min_t, self.max_t))

        def objective(u_flat):
            u_matrix = u_flat.reshape((self.np, 3))
            cost = 0.0
            x_curr = current_state_mean.copy()
            cov_curr = current_state_cov.copy()

            for k in range(self.np):
                u_k = u_matrix[k]
                ref_k = reference_trajectory[k]

                x_curr, cov_curr = self.propagate_mean_and_covariance(x_curr, cov_curr, u_k, nominal_wind_ms)

                # Tracking error penalty
                pos_err = x_curr[0:3] - ref_k[0:3]
                vel_err = x_curr[3:6] - ref_k[3:6]
                cost += pos_err.T @ self.q_pos @ pos_err
                cost += vel_err.T @ self.q_vel @ vel_err

                # Control effort penalty
                ctrl_err = np.array([u_k[0], u_k[1], u_k[2] - hover_thrust])
                cost += ctrl_err.T @ self.r_ctrl @ ctrl_err

                # Deterministic equivalent of chance constraint:
                # |y_mean| <= b_y - z_score * sigma_y
                sigma_y = math.sqrt(max(cov_curr[1, 1], 1e-6))
                tightened_bound = geofence_half_width_m - (self.z_score * sigma_y)
                
                # Soft penalty for tightened geofence infringement
                if abs(x_curr[1]) > tightened_bound:
                    cost += 5000.0 * (abs(x_curr[1]) - tightened_bound) ** 2

            return cost

        res = minimize(objective, u_init, method="SLSQP", bounds=bounds, options={"maxiter": 30, "ftol": 1e-3})
        opt_u_seq = res.x.reshape((self.np, 3))
        
        # Propagate forward with optimal inputs to calculate final tightened corridor
        x_prop = current_state_mean.copy()
        cov_prop = current_state_cov.copy()
        tightened_margins = []

        for k in range(self.np):
            x_prop, cov_prop = self.propagate_mean_and_covariance(x_prop, cov_prop, opt_u_seq[k], nominal_wind_ms)
            sigma_lat = math.sqrt(max(cov_prop[1, 1], 1e-6))
            tightened_margins.append(geofence_half_width_m - (self.z_score * sigma_lat))

        return opt_u_seq[0], {
            "optimization_success": res.success,
            "cost_value": round(float(res.fun), 3),
            "safety_confidence_pct": (1.0 - self.epsilon) * 100.0,
            "z_score_quantile": round(self.z_score, 3),
            "initial_corridor_width_m": geofence_half_width_m * 2.0,
            "terminal_tightened_corridor_width_m": round(tightened_margins[-1] * 2.0, 2),
            "uncertainty_safety_buffer_m": round(geofence_half_width_m - tightened_margins[-1], 2)
        }

if __name__ == "__main__":
    cc_mpc = StochasticChanceConstrainedMPC(horizon_steps=10, violation_probability_epsilon=0.005)

    x0_mean = np.array([0.0, 0.0, 25.0, 15.0, 0.0, 0.0])
    x0_cov = np.diag([0.2**2, 0.2**2, 0.1**2, 0.05**2, 0.05**2, 0.02**2])

    ref_traj = np.zeros((10, 6))
    for k in range(10):
        ref_traj[k, 0] = x0_mean[0] + 18.05 * (k + 1) * 0.1
        ref_traj[k, 1] = 0.0
        ref_traj[k, 2] = 25.0
        ref_traj[k, 3] = 18.05

    wind_nom = np.array([-3.5, 2.5, 0.0])

    u0, diagnostic = cc_mpc.solve_chance_constrained_controls(
        x0_mean, x0_cov, ref_traj, wind_nom, geofence_half_width_m=15.0
    )

    print("==========================================================================")
    print("      STOCHASTIC CHANCE-CONSTRAINED MPC (CC-MPC) TRAJECTORY AUDIT         ")
    print("==========================================================================")
    for k, v in diagnostic.items():
        print(f"  • {k:38s}: {v}")
    print(f"\nCommanded Control Action -> Pitch: {math.degrees(u0[0]):+.2f}°, Roll: {math.degrees(u0[1]):+.2f}°, Thrust: {u0[2]:.2f} N")
    print("==========================================================================\n")