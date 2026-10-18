import math
import numpy as np
from scipy.stats import chi2
from typing import Dict, List, Tuple

class MultiSensorFaultDetectorFDI:
    def __init__(
        self,
        false_alarm_rate_alpha: float = 0.001, # 99.9% statistical confidence
        window_len: int = 5
    ):
        self.alpha = false_alarm_rate_alpha
        self.w_len = window_len

        # Degrees of freedom per sensor subsystem
        self.sensor_dof = {
            "GPS_POS": 2,      # x, y horizontal
            "GPS_VEL": 2,      # vx, vy horizontal
            "BAROMETER": 1,    # altitude z
            "AIRSPEED": 1,     # pitot airspeed
            "VIO_ODOMETRY": 2  # dead-reckoned x, y
        }

        # Precompute Chi-Square critical threshold limits
        self.chi2_thresholds = {
            s: float(chi2.ppf(1.0 - self.alpha, df=dof))
            for s, dof in self.sensor_dof.items()
        }

        # Historical residual buffers
        self.residual_history = {s: [] for s in self.sensor_dof}
        self.sensor_health_status = {s: "HEALTHY_OPERATIONAL" for s in self.sensor_dof}

    def evaluate_sensor_channel_innovation(
        self,
        sensor_name: str,
        innovation_residual: np.ndarray,
        innovation_covariance_s: np.ndarray
    ) -> Dict:
        """
        Calculates Mahalanobis distance metric gamma = y^T * S^-1 * y
        and evaluates against Chi-Square threshold.
        """
        if sensor_name not in self.sensor_dof:
            return {"error": "UNKNOWN_SENSOR"}

        y = np.atleast_1d(innovation_residual)
        s_mat = np.atleast_2d(innovation_covariance_s)

        # Mahalanobis distance / NIS (Normalized Innovation Squared)
        gamma = float(y.T @ np.linalg.inv(s_mat) @ y)
        thresh = self.chi2_thresholds[sensor_name]

        # Update sliding window
        self.residual_history[sensor_name].append(gamma)
        if len(self.residual_history[sensor_name]) > self.w_len:
            self.residual_history[sensor_name].pop(0)

        window_mean_nis = float(np.mean(self.residual_history[sensor_name]))

        # Fault Isolation Logic
        is_instantaneous_fault = gamma > thresh
        is_persistent_drift = window_mean_nis > (thresh * 0.75)

        action = "ACCEPT_MEASUREMENT"
        if is_instantaneous_fault:
            self.sensor_health_status[sensor_name] = "ISOLATED_STEP_DISCONTINUITY"
            action = "REJECT_OUTLIER_AND_ISOLATE"
        elif is_persistent_drift:
            self.sensor_health_status[sensor_name] = "DEGRADED_BIAS_DRIFT"
            action = "INFLATE_COVARIANCE_WEIGHT_DEGRADE"
        else:
            self.sensor_health_status[sensor_name] = "HEALTHY_OPERATIONAL"

        return {
            "sensor_subsystem": sensor_name,
            "instantaneous_nis_gamma": round(gamma, 3),
            "chi2_critical_threshold": round(thresh, 3),
            "window_mean_nis": round(window_mean_nis, 3),
            "fault_detected": is_instantaneous_fault or is_persistent_drift,
            "sensor_health": self.sensor_health_status[sensor_name],
            "fdi_mitigation_action": action
        }

if __name__ == "__main__":
    fdi = MultiSensorFaultDetectorFDI()

    print("==========================================================================")
    print("      CHI-SQUARE SENSOR FAULT DETECTION, ISOLATION & RECOVERY (FDIR)     ")
    print("==========================================================================")

    # 1. Evaluate Nominal Barometer Channel (DOF=1, Threshold = 10.83)
    res_baro_nom = fdi.evaluate_sensor_channel_innovation(
        sensor_name="BAROMETER",
        innovation_residual=np.array([0.25]), # 0.25m error
        innovation_covariance_s=np.array([[0.8**2]])
    )
    print(f"Nominal Barometer Check -> NIS: {res_baro_nom['instantaneous_nis_gamma']:5.2f} (Thresh: {res_baro_nom['chi2_critical_threshold']}) | Status: {res_baro_nom['sensor_health']}")

    # 2. Inject Sudden GPS Step Jump Outlier (+12.0 meters due to multipath reflection)
    res_gps_fault = fdi.evaluate_sensor_channel_innovation(
        sensor_name="GPS_POS",
        innovation_residual=np.array([12.0, -8.5]),
        innovation_covariance_s=np.diag([1.5**2, 1.5**2])
    )
    print(f"Spoofed GPS Multipath   -> NIS: {res_gps_fault['instantaneous_nis_gamma']:5.2f} (Thresh: {res_gps_fault['chi2_critical_threshold']}) | Status: {res_gps_fault['sensor_health']} | Action: {res_gps_fault['fdi_mitigation_action']}")
    print("==========================================================================\n")