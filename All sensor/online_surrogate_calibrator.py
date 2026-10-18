import math
import numpy as np
from typing import Dict, Tuple

class OnlineSurrogateResidualCalibrator:
    def __init__(
        self,
        num_features: int = 4, # [bias_offset, speed_scale, payload_scale, wind_scale]
        forgetting_factor_lambda: float = 0.995,
        initial_covariance: float = 100.0
    ):
        self.dim = num_features
        self.lam = forgetting_factor_lambda
        
        # Parameter Vector theta: [Delta_P0, alpha_speed, alpha_payload, alpha_wind]
        self.theta = np.zeros(self.dim)
        # Error Covariance Matrix P
        self.P = np.eye(self.dim) * initial_covariance
        self.iteration_count = 0

    def update_calibration_step(
        self,
        surrogate_predicted_watts: float,
        actual_measured_watts: float,
        flight_features: Tuple[float, float, float] # (ground_speed_ms, payload_kg, wind_speed_ms)
    ) -> Dict:
        """
        Executes one recursive least squares update cycle on the power residual:
          Residual y = P_measured - P_surrogate = phi^T * theta + noise
        """
        self.iteration_count += 1
        speed, payload, wind = flight_features
        
        # Regressor vector phi
        phi = np.array([1.0, speed / 18.05, payload / 2.0, wind / 8.33])

        # Measured Residual
        y_residual = actual_measured_watts - surrogate_predicted_watts

        # 1. Compute Kalman Gain Vector K
        p_phi = self.P @ phi
        denom = self.lam + float(phi.T @ p_phi)
        k_gain = p_phi / max(denom, 1e-6)

        # 2. A Priori Estimation Error
        error_prior = y_residual - float(phi.T @ self.theta)

        # 3. Update Parameter Vector
        self.theta = self.theta + k_gain * error_prior

        # 4. Update Covariance Matrix: P = (1/lambda) * (P - K * phi^T * P)
        self.P = (1.0 / self.lam) * (self.P - np.outer(k_gain, phi.T @ self.P))

        # Corrected power estimate
        calibrated_power_watts = surrogate_predicted_watts + float(phi.T @ self.theta)

        return {
            "iteration": self.iteration_count,
            "raw_surrogate_watts": round(surrogate_predicted_watts, 2),
            "actual_measured_watts": round(actual_measured_watts, 2),
            "instantaneous_residual_watts": round(y_residual, 2),
            "online_corrected_power_watts": round(calibrated_power_watts, 2),
            "learned_bias_delta_p0_watts": round(float(self.theta[0]), 3),
            "learned_speed_scale": round(float(self.theta[1]), 3),
            "learned_payload_scale": round(float(self.theta[2]), 3),
            "learned_wind_scale": round(float(self.theta[3]), 3),
            "tracking_error_variance": round(float(np.trace(self.P)), 4)
        }

if __name__ == "__main__":
    calibrator = OnlineSurrogateResidualCalibrator()

    print("==========================================================================")
    print("      ONLINE RECURSIVE SURROGATE POWER RESIDUAL CALIBRATION AUDIT         ")
    print("==========================================================================")
    
    # Simulate a drone with +18.5W extra mechanical drag offset and payload aerodynamic scaling
    for step in range(1, 11):
        v_spd = 18.05 + np.random.normal(0, 0.5)
        m_pay = 1.5
        w_spd = 4.2
        
        # Nominal model predicts 290.0 W
        p_nominal = 290.0
        # True aircraft draws +22.0 W due to motor bearing friction and higher drag
        p_real = p_nominal + 22.0 + np.random.normal(0, 1.2)

        res = calibrator.update_calibration_step(
            surrogate_predicted_watts=p_nominal,
            actual_measured_watts=p_real,
            flight_features=(v_spd, m_pay, w_spd)
        )
        print(f"Step {step:02d}: Measured = {res['actual_measured_watts']:5.1f}W | "
              f"Surrogate = {res['raw_surrogate_watts']:5.1f}W | Calibrated = {res['online_corrected_power_watts']:5.1f}W | "
              f"Learned Bias = {res['learned_bias_delta_p0_watts']:+5.2f}W | Error Var = {res['tracking_error_variance']:.2f}")
    print("==========================================================================\n")