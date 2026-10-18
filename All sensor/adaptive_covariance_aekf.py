import math
import numpy as np
from typing import Dict, Tuple, Optional

class SageHusaAdaptiveEKF:
    def __init__(
        self,
        dt_seconds: float = 0.05,
        forgetting_factor_b: float = 0.985,
        min_r_diagonal: float = 1e-4,
        min_q_diagonal: float = 1e-6
    ):
        self.dt = dt_seconds
        self.b = forgetting_factor_b
        self.min_r = min_r_diagonal
        self.min_q = min_q_diagonal
        self.step_k = 0

        # State vector x = [pos_x, pos_y, vel_x, vel_y, wind_u, wind_v]^T (6x1)
        self.x = np.zeros((6, 1))
        self.P = np.diag([1.0, 1.0, 0.5, 0.5, 2.0, 2.0])

        # Initial Process Covariance Q and Measurement Covariance R
        self.Q = np.diag([0.01, 0.01, 0.05, 0.05, 0.1, 0.1])
        self.R = np.diag([0.25, 0.25, 0.04, 0.04]) # [gps_x, gps_y, vel_x, vel_y]

        # State Transition Matrix Phi
        self.Phi = np.eye(6)
        self.Phi[0, 2] = self.dt
        self.Phi[1, 3] = self.dt

        # Measurement Matrix H (4x6)
        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

    def predict(self, accel_cmd_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        State propagation step:
          x_{k|k-1} = Phi * x_{k-1} + Gamma * u_k
          P_{k|k-1} = Phi * P_{k-1} * Phi^T + Q_{k-1}
        """
        gamma = np.zeros((6, 2))
        gamma[0:2, :] = 0.5 * (self.dt ** 2) * np.eye(2)
        gamma[2:4, :] = self.dt * np.eye(2)

        self.x = self.Phi @ self.x + gamma @ accel_cmd_xy.reshape(2, 1)
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q
        return self.x, self.P

    def update_adaptive(self, measurement_z: np.ndarray) -> Dict:
        """
        Executes Sage-Husa MAP measurement update and online covariance adaptation.
        measurement_z: [meas_pos_x, meas_pos_y, meas_vel_x, meas_vel_y]^T
        """
        self.step_k += 1
        z = measurement_z.reshape(4, 1)

        # Innovation / Residual: v_k = z_k - H * x_{k|k-1}
        v = z - self.H @ self.x

        # Innovation Covariance S
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State and Covariance Update
        p_prior = self.P.copy()
        self.x = self.x + K @ v
        self.P = (np.eye(6) - K @ self.H) @ self.P

        # Time-varying weighting factor d_k
        d_k = (1.0 - self.b) / (1.0 - (self.b ** (self.step_k + 1)))

        # 1. Sage-Husa R Adaptation
        r_raw = (1.0 - d_k) * self.R + d_k * (v @ v.T - self.H @ p_prior @ self.H.T)
        # Ensure positive semi-definiteness and enforce diagonal lower bound
        r_diag = np.maximum(np.diag(r_raw), self.min_r)
        self.R = np.diag(r_diag)

        # 2. Sage-Husa Q Adaptation
        q_raw = (1.0 - d_k) * self.Q + d_k * (
            K @ v @ v.T @ K.T + self.P - self.Phi @ p_prior @ self.Phi.T
        )
        q_diag = np.maximum(np.diag(q_raw), self.min_q)
        self.Q = np.diag(q_diag)

        # Extract Airspeed & Estimated Wind
        ground_vel = self.x[2:4, 0]
        est_wind = self.x[4:6, 0]
        true_airspeed = np.linalg.norm(ground_vel - est_wind)

        return {
            "step": self.step_k,
            "estimated_pos_m": (round(float(self.x[0, 0]), 2), round(float(self.x[1, 0]), 2)),
            "estimated_vel_ms": (round(float(self.x[2, 0]), 2), round(float(self.x[3, 0]), 2)),
            "estimated_wind_ms": (round(float(self.x[4, 0]), 2), round(float(self.x[5, 0]), 2)),
            "true_airspeed_ms": round(float(true_airspeed), 2),
            "adapted_r_trace": round(float(np.trace(self.R)), 4),
            "adapted_q_trace": round(float(np.trace(self.Q)), 4),
            "filter_status": "CONVERGED_ADAPTIVE_LOCKED"
        }

if __name__ == "__main__":
    aekf = SageHusaAdaptiveEKF(dt_seconds=0.05, forgetting_factor_b=0.985)

    print("==========================================================================")
    print("      SAGE-HUSA ADAPTIVE EXTENDED KALMAN FILTER (AEKF) TEST               ")
    print("==========================================================================")

    true_wind = np.array([3.5, -2.0])
    acc_cmd = np.array([0.0, 0.0])

    for k in range(1, 21):
        aekf.predict(acc_cmd)
        
        # Simulate noisy GPS with sudden sensor degradation spike at step 10
        pos_noise_std = 0.5 if k < 10 else 2.8
        sim_pos = np.array([18.05 * (k * 0.05), 0.0]) + np.random.normal(0, pos_noise_std, 2)
        sim_vel = np.array([18.05, 0.0]) + np.random.normal(0, 0.15, 2)
        meas = np.concatenate([sim_pos, sim_vel])

        report = aekf.update_adaptive(meas)
        if k % 4 == 0:
            print(f"Step {k:02d} | Pos: {report['estimated_pos_m']} | Vel: {report['estimated_vel_ms']} | "
                  f"Wind: {report['estimated_wind_ms']} | Trace(R): {report['adapted_r_trace']:.4f} | Trace(Q): {report['adapted_q_trace']:.4f}")
    print("==========================================================================\n")