import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class BatteryRULPrognosticsUKF:
    def __init__(
        self,
        nominal_capacity_ah: float = 22.0,
        eol_threshold_soh: float = 0.80,
        initial_cycle_count: int = 100
    ):
        self.c_nominal = nominal_capacity_ah
        self.eol_soh = eol_threshold_soh
        self.current_cycle = initial_cycle_count
        
        # State vector x = [SOH, Degradation_Rate_alpha]
        # Empirical degradation: SOH_{k} = SOH_{k-1} - alpha * exp(beta * C_rate)
        self.x = np.array([[0.96], [0.00018]])
        self.P = np.diag([0.001**2, (0.00002)**2])
        self.Q = np.diag([1e-7, 1e-10])  # Process noise
        self.R = np.array([[0.005**2]])   # Inspection measurement noise

        # UKF Scaling Parameters
        self.n = 2
        self.alpha_ukf = 1e-3
        self.kappa = 0.0
        self.beta_ukf = 2.0
        self.lambda_ = (self.alpha_ukf ** 2) * (self.n + self.kappa) - self.n
        self.gamma = math.sqrt(self.n + self.lambda_)

        # Weights
        self.Wm = np.full(2 * self.n + 1, 0.5 / (self.n + self.lambda_))
        self.Wc = np.full(2 * self.n + 1, 0.5 / (self.n + self.lambda_))
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alpha_ukf**2 + self.beta_ukf)

    def _generate_sigma_points(self) -> np.ndarray:
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.x.flatten()
        sqrt_P = np.linalg.cholesky(self.P)

        for i in range(self.n):
            sigma_points[i + 1] = self.x.flatten() + self.gamma * sqrt_P[:, i]
            sigma_points[self.n + i + 1] = self.x.flatten() - self.gamma * sqrt_P[:, i]

        return sigma_points

    def predict_cycle(self, mission_c_rate: float):
        """Propagates state forward by one operational mission discharge cycle."""
        sigma_pts = self._generate_sigma_points()
        propagated_pts = np.zeros_like(sigma_pts)

        for i in range(len(sigma_pts)):
            soh_i = sigma_pts[i, 0]
            alpha_i = sigma_pts[i, 1]
            c_stress = math.exp(0.45 * (mission_c_rate - 1.0))
            soh_next = soh_i - (alpha_i * c_stress)
            propagated_pts[i] = [soh_next, alpha_i]

        # Prior State Mean
        x_prior = np.sum(self.Wm[:, None] * propagated_pts, axis=0).reshape(self.n, 1)

        # Prior State Covariance
        P_prior = np.zeros((self.n, self.n))
        for i in range(len(sigma_pts)):
            diff = (propagated_pts[i] - x_prior.flatten()).reshape(self.n, 1)
            P_prior += self.Wc[i] * (diff @ diff.T)
        P_prior += self.Q

        self.x = x_prior
        self.P = P_prior
        self.current_cycle += 1

    def update_measurement(self, measured_soh: float):
        """Incorporates ground-truth capacity measurement from smart balancing charger."""
        sigma_pts = self._generate_sigma_points()
        z_pts = sigma_pts[:, 0]  # Measurement model h(x) = SOH
        
        z_mean = np.sum(self.Wm * z_pts)
        
        # Innovation Covariance
        P_zz = sum(self.Wc[i] * ((z_pts[i] - z_mean) ** 2) for i in range(len(z_pts))) + self.R[0, 0]
        
        # Cross Covariance
        P_xz = np.zeros((self.n, 1))
        for i in range(len(sigma_pts)):
            diff_x = (sigma_pts[i] - self.x.flatten()).reshape(self.n, 1)
            diff_z = z_pts[i] - z_mean
            P_xz += self.Wc[i] * (diff_x * diff_z)

        # Kalman Gain
        K = P_xz / P_zz
        residual = measured_soh - z_mean
        
        self.x = self.x + K * residual
        self.P = self.P - (K @ K.T) * P_zz

    def estimate_remaining_useful_life(self, projected_avg_c_rate: float = 1.25) -> Dict:
        """Projects degradation forward to determine cycles remaining until 80% SOH."""
        soh_cursor = float(self.x[0, 0])
        alpha_cursor = float(self.x[1, 0])
        c_stress = math.exp(0.45 * (projected_avg_c_rate - 1.0))
        
        remaining_cycles = 0
        while soh_cursor > self.eol_soh and remaining_cycles < 2000:
            soh_cursor -= (alpha_cursor * c_stress)
            remaining_cycles += 1

        total_projected_cycles = self.current_cycle + remaining_cycles
        return {
            "current_cycle": self.current_cycle,
            "estimated_soh_pct": round(float(self.x[0, 0]) * 100.0, 2),
            "estimated_alpha_decay": f"{float(self.x[1, 0]):.4e}",
            "remaining_useful_life_cycles": remaining_cycles,
            "projected_total_lifetime_cycles": total_projected_cycles,
            "pack_replacement_required": float(self.x[0, 0]) <= self.eol_soh
        }

if __name__ == "__main__":
    prognostics = BatteryRULPrognosticsUKF(initial_cycle_count=150)
    print("==========================================================================")
    print("       BATTERY STATE-OF-HEALTH (SOH) & RUL PROGNOSTICS (UKF)              ")
    print("==========================================================================")
    
    # Simulate 50 operational flights with alternating C-rates
    for cycle in range(1, 51):
        c_rate = 1.1 if cycle % 2 == 0 else 1.45
        prognostics.predict_cycle(mission_c_rate=c_rate)

        # Periodic charger measurement calibration every 10 cycles
        if cycle % 10 == 0:
            synthetic_measured_soh = 0.96 - (cycle * 0.00022) + np.random.normal(0, 0.002)
            prognostics.update_measurement(synthetic_measured_soh)

    rul_report = prognostics.estimate_remaining_useful_life(projected_avg_c_rate=1.25)
    for k, v in rul_report.items():
        print(f"  • {k:35s}: {v}")
    print("==========================================================================\n")