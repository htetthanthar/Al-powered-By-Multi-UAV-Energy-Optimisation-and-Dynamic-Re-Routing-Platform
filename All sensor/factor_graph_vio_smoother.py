import math
import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Tuple

class FactorGraphVIOSmoother:
    def __init__(
        self,
        window_size_k: int = 5,
        camera_focal_length_px: float = 520.0,
        camera_principal_point: Tuple[float, float] = (320.0, 240.0),
        imu_accel_noise_std: float = 0.08,
        imu_gyro_noise_std: float = 0.005,
        rtk_pos_noise_std: float = 0.02
    ):
        self.K = window_size_k
        self.fx = camera_focal_length_px
        self.cx, self.cy = camera_principal_point
        self.sigma_imu = imu_accel_noise_std
        self.sigma_cam = 1.5 # 1.5 pixels reprojection error std
        self.sigma_rtk = rtk_pos_noise_std
        self.g = np.array([0.0, 0.0, 9.80665])

    def _project_point(self, p_cam: np.ndarray) -> np.ndarray:
        """Pinhole camera projection: (X, Y, Z) -> (u, v)"""
        z_safe = max(p_cam[2], 0.1)
        u = self.fx * (p_cam[0] / z_safe) + self.cx
        v = self.fx * (p_cam[1] / z_safe) + self.cy
        return np.array([u, v])

    def optimize_sliding_window(
        self,
        initial_poses_ned: List[np.ndarray], # List of [x, y, z] vectors (length K)
        initial_velocities: List[np.ndarray], # List of [vx, vy, vz] vectors (length K)
        imu_preintegrated_deltas: List[Dict], # Preintegrated Delta_p, Delta_v, dt
        visual_feature_tracks: List[Dict],    # Landmark 3D point and observed 2D pixel coordinates
        rtk_gnss_priors: Dict[int, np.ndarray] # Key: node_idx, Value: [x_rtk, y_rtk, z_rtk]
    ) -> Dict:
        """
        Solves nonlinear least squares optimization over the sliding window state vector:
        State theta = [p_0..p_{K-1}, v_0..v_{K-1}] (Size: 6 * K)
        """
        # Pack state vector
        p_flat = np.concatenate(initial_poses_ned)
        v_flat = np.concatenate(initial_velocities)
        theta_0 = np.concatenate([p_flat, v_flat])

        def residual_function(theta: np.ndarray) -> np.ndarray:
            p_nodes = [theta[3*k : 3*k+3] for k in range(self.K)]
            v_nodes = [theta[3*self.K + 3*k : 3*self.K + 3*k+3] for k in range(self.K)]
            residuals = []

            # 1. IMU Preintegration Residuals between consecutive poses
            for k in range(self.K - 1):
                dt = imu_preintegrated_deltas[k]["dt_s"]
                delta_p_imu = imu_preintegrated_deltas[k]["delta_p"]
                delta_v_imu = imu_preintegrated_deltas[k]["delta_v"]

                # Expected kinematics: p_{k+1} = p_k + v_k * dt + 0.5 * g * dt^2 + delta_p
                pred_p = p_nodes[k] + v_nodes[k] * dt + 0.5 * self.g * (dt ** 2) + delta_p_imu
                err_p = (p_nodes[k+1] - pred_p) / (self.sigma_imu * dt)
                residuals.extend(err_p)

                pred_v = v_nodes[k] + self.g * dt + delta_v_imu
                err_v = (v_nodes[k+1] - pred_v) / (self.sigma_imu * math.sqrt(dt))
                residuals.extend(err_v)

            # 2. Visual Reprojection Residuals
            for feat in visual_feature_tracks:
                l_3d = feat["landmark_world_ned"]
                for obs in feat["observations"]:
                    k_idx = obs["node_idx"]
                    uv_meas = obs["pixel_uv"]

                    # Transform landmark into camera frame (downward camera assumption)
                    p_cam = l_3d - p_nodes[k_idx]
                    uv_pred = self._project_point(p_cam)
                    err_pixel = (uv_pred - uv_meas) / self.sigma_cam
                    residuals.extend(err_pixel)

            # 3. RTK GNSS Prior Residuals
            for k_idx, p_rtk in rtk_gnss_priors.items():
                err_rtk = (p_nodes[k_idx] - p_rtk) / self.sigma_rtk
                residuals.extend(err_rtk)

            return np.array(residuals)

        # Execute Levenberg-Marquardt Optimization
        res = least_squares(residual_function, theta_0, method="lm", xtol=1e-4, ftol=1e-4)

        opt_theta = res.x
        opt_p_nodes = [opt_theta[3*k : 3*k+3].tolist() for k in range(self.K)]
        opt_v_nodes = [opt_theta[3*self.K + 3*k : 3*self.K + 3*k+3].tolist() for k in range(self.K)]

        # Calculate Root-Mean-Square Reprojection and Positioning Residuals
        total_chi2 = float(np.sum(res.fun ** 2))
        degrees_of_freedom = max(1, len(res.fun) - len(theta_0))
        reduced_chi2 = total_chi2 / degrees_of_freedom

        return {
            "optimization_converged": bool(res.success),
            "lm_iterations": int(res.nfev),
            "total_residual_chi2": round(total_chi2, 3),
            "reduced_chi2": round(reduced_chi2, 3),
            "smoothed_terminal_position_ned_m": [round(c, 3) for c in opt_p_nodes[-1]],
            "smoothed_terminal_velocity_ned_ms": [round(c, 3) for c in opt_v_nodes[-1]],
            "smoother_confidence": "OPTIMAL_FIXED" if reduced_chi2 <= 1.8 else "HIGH_INNOVATION_DIVERGENCE"
        }

if __name__ == "__main__":
    smoother = FactorGraphVIOSmoother(window_size_k=4)

    # Initial noisy trajectory nodes: 4 nodes at 0.1s intervals (18.05 m/s forward cruise)
    init_p = [np.array([1.805 * i, 0.0, -25.0]) + np.random.normal(0, 0.08, 3) for i in range(4)]
    init_v = [np.array([18.05, 0.0, 0.0]) + np.random.normal(0, 0.05, 3) for i in range(4)]

    # Preintegrated IMU records
    imu_deltas = [
        {"dt_s": 0.1, "delta_p": np.array([0.0, 0.0, -0.5 * 9.80665 * 0.01]), "delta_v": np.array([0.0, 0.0, -9.80665 * 0.1])}
        for _ in range(3)
    ]

    # Visual tracks of ground target at [10.0, 2.0, 0.0]
    vis_tracks = [{
        "landmark_world_ned": np.array([10.0, 2.0, 0.0]),
        "observations": [
            {"node_idx": 0, "pixel_uv": np.array([450.0, 280.0])},
            {"node_idx": 1, "pixel_uv": np.array([420.0, 280.0])},
            {"node_idx": 2, "pixel_uv": np.array([390.0, 280.0])}
        ]
    }]

    # RTK GNSS available at Node 0 and Node 3
    rtk_priors = {
        0: np.array([0.0, 0.0, -25.0]),
        3: np.array([5.415, 0.0, -25.0])
    }

    audit = smoother.optimize_sliding_window(init_p, init_v, imu_deltas, vis_tracks, rtk_priors)

    print("==========================================================================")
    print("      FACTOR GRAPH VISUAL-INERTIAL-GNSS (FGO) SMOOTHER AUDIT              ")
    print("==========================================================================")
    for k, v in audit.items():
        print(f"  • {k:38s}: {v}")
    print("==========================================================================\n")