import math
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple

class DirectLiDARInertialOdometryIEKF:
    def __init__(
        self,
        dt_imu_s: float = 0.005, # 200 Hz IMU
        max_iekf_iterations: int = 4,
        plane_distance_threshold_m: float = 0.08,
        point_cloud_downsample_m: float = 0.25
    ):
        self.dt = dt_imu_s
        self.max_iter = max_iekf_iterations
        self.d_thresh = plane_distance_threshold_m
        self.ds_res = point_cloud_downsample_m
        self.g = np.array([0.0, 0.0, 9.80665])

        # State: [R (3x3), p (3), v (3), b_g (3), b_a (3)]
        self.R = np.eye(3)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.bg = np.zeros(3) # Gyroscope bias
        self.ba = np.zeros(3) # Accelerometer bias

        # Error state covariance P (15x15)
        self.P = np.diag([
            1e-3, 1e-3, 1e-3,  # rot
            1e-2, 1e-2, 1e-2,  # pos
            1e-2, 1e-2, 1e-2,  # vel
            1e-4, 1e-4, 1e-4,  # bg
            1e-3, 1e-3, 1e-3   # ba
        ])

        # Global ikd-Tree planar surface primitives: List of (plane_point, plane_normal)
        self.map_planes: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),  # Ground plane
            (np.array([15.0, 0.0, -5.0]), np.array([-1.0, 0.0, 0.0])), # Building wall
            (np.array([0.0, 15.0, -5.0]), np.array([0.0, -1.0, 0.0]))  # Side wall
        ]

    def forward_propagate_imu(
        self,
        accel_meas: np.ndarray,
        gyro_meas: np.ndarray
    ):
        """Continuous forward integration of IMU measurements on manifold."""
        # Unbiased angular rate and acceleration
        omega = gyro_meas - self.bg
        accel = accel_meas - self.ba

        # State kinematics
        delta_rot = Rotation.from_rotvec(omega * self.dt).as_matrix()
        a_world = self.R @ accel - self.g

        self.p = self.p + self.v * self.dt + 0.5 * a_world * (self.dt ** 2)
        self.v = self.v + a_world * self.dt
        self.R = self.R @ delta_rot

        # Discrete Error State Transition Matrix F_x (15x15)
        Fx = np.eye(15)
        Fx[0:3, 0:3] = delta_rot.T
        Fx[0:3, 9:12] = -np.eye(3) * self.dt
        Fx[3:6, 6:9] = np.eye(3) * self.dt
        Fx[6:9, 0:3] = -self.R @ self._skew(accel) * self.dt
        Fx[6:9, 12:15] = -self.R * self.dt

        # Continuous Noise Covariance Q
        Q = np.eye(15) * 1e-4
        self.P = Fx @ self.P @ Fx.T + Q

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        return np.array([
            [  0.0, -v[2],  v[1]],
            [ v[2],   0.0, -v[0]],
            [-v[1],  v[0],   0.0]
        ])

    def update_point_cloud_scan(
        self,
        raw_point_cloud_body: np.ndarray # Shape (N, 3)
    ) -> Dict:
        """
        Executes Iterated Extended Kalman Filter (IEKF) update using point-to-plane residuals.
        """
        # Downsample point cloud
        pts = raw_point_cloud_body[::4]
        n_pts = len(pts)

        x_propagated_R = self.R.copy()
        x_propagated_p = self.p.copy()

        for iteration in range(self.max_iter):
            residuals = []
            h_rows = []

            for pt_b in pts:
                # Transform point to world frame: p_w = R * p_b + p
                pt_w = self.R @ pt_b + self.p

                # Find nearest planar patch in ikd-tree
                for q_plane, n_plane in self.map_planes:
                    dist = float(np.dot(pt_w - q_plane, n_plane))
                    if abs(dist) <= self.d_thresh:
                        # Valid plane match
                        residuals.append(dist)
                        # Measurement Jacobian: H = [ -n^T * R * (p_b)^skew, n^T, 0, 0, 0 ]
                        h_rot = -n_plane.T @ self.R @ self._skew(pt_b)
                        h_pos = n_plane.T
                        h_full = np.zeros(15)
                        h_full[0:3] = h_rot
                        h_full[3:6] = h_pos
                        h_rows.append(h_full)
                        break

            if len(residuals) < 8:
                break

            r_vec = np.array(residuals)
            H_mat = np.array(h_rows)

            # Measurement covariance matrix R_cov
            sigma_meas2 = (0.02 ** 2) # 2cm LiDAR point noise

            # IEKF Gain: K = (H^T * H + sigma^2 * P^-1)^-1 * H^T
            p_inv = np.linalg.inv(self.P)
            s_mat = H_mat.T @ H_mat + sigma_meas2 * p_inv
            K_gain = np.linalg.inv(s_mat) @ H_mat.T

            # State error update: delta_x = -K * r
            delta_x = -K_gain @ r_vec

            # Update manifold state: x = x boxplus delta_x
            d_rot = delta_x[0:3]
            d_pos = delta_x[3:6]
            d_vel = delta_x[6:9]
            d_bg  = delta_x[9:12]
            d_ba  = delta_x[12:15]

            self.R = self.R @ Rotation.from_rotvec(d_rot).as_matrix()
            self.p = self.p + d_pos
            self.v = self.v + d_vel
            self.bg = self.bg + d_bg
            self.ba = self.ba + d_ba

            if np.linalg.norm(delta_x) < 1e-4:
                break

        # Final Covariance Update
        I_KH = np.eye(15) - K_gain @ H_mat
        self.P = I_KH @ self.P

        return {
            "iekf_iterations": iteration + 1,
            "planar_matches_used": len(residuals),
            "estimated_pos_world_m": (round(float(self.p[0]), 3),
                                      round(float(self.p[1]), 3),
                                      round(float(self.p[2]), 3)),
            "estimated_vel_world_ms": (round(float(self.v[0]), 3),
                                       round(float(self.v[1]), 3),
                                       round(float(self.v[2]), 3)),
            "point_to_plane_rmse_mm": round(float(np.sqrt(np.mean(r_vec**2)) * 1000.0), 2) if len(residuals) > 0 else 0.0,
            "lio_lock_status": "DIRECT_LIO_TRACKING_LOCKED" if len(residuals) >= 8 else "INSUFFICIENT_GEOMETRIC_CONSTRAINTS"
        }

if __name__ == "__main__":
    lio = DirectLiDARInertialOdometryIEKF()

    print("==========================================================================")
    print("      DIRECT ITERATED KALMAN (IEKF) LIDAR-INERTIAL ODOMETRY AUDIT        ")
    print("==========================================================================")

    # 1. Forward propagate IMU across 10 steps (0.05s)
    for _ in range(10):
        lio.forward_propagate_imu(
            accel_meas=np.array([0.0, 0.0, 9.80665]),
            gyro_meas=np.array([0.0, 0.0, 0.0])
        )

    # 2. Synthesize point cloud hits on ground plane and building wall
    np.random.seed(42)
    cloud_ground = np.column_stack([
        np.random.uniform(-5.0, 5.0, 40),
        np.random.uniform(-5.0, 5.0, 40),
        np.zeros(40) + np.random.normal(0, 0.01, 40)
    ])
    cloud_wall = np.column_stack([
        np.full(30, 15.0) + np.random.normal(0, 0.01, 30),
        np.random.uniform(-4.0, 4.0, 30),
        np.random.uniform(-8.0, 0.0, 30)
    ])
    scan_body = np.vstack([cloud_ground, cloud_wall])

    lio_report = lio.update_point_cloud_scan(scan_body)
    for k, v in lio_report.items():
        print(f"  • {k:30s}: {v}")
    print("==========================================================================\n")