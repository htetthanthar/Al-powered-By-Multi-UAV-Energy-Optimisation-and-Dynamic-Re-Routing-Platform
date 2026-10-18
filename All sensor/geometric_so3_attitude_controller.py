import math
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple

class GeometricSO3AttitudeController:
    def __init__(
        self,
        inertia_matrix_diag_kg_m2: Tuple[float, float, float] = (0.082, 0.082, 0.145),
        proportional_gain_kr: float = 8.5,
        derivative_gain_komega: float = 1.65,
        max_torque_nm: float = 3.8
    ):
        self.J = np.diag(inertia_matrix_diag_kg_m2)
        self.J_inv = np.linalg.inv(self.J)
        self.kR = proportional_gain_kr
        self.kOmega = derivative_gain_komega
        self.tau_max = max_torque_nm

    @staticmethod
    def hat_map(vector3: np.ndarray) -> np.ndarray:
        """Skew-symmetric matrix operator (hat map) from R^3 to so(3)."""
        v = vector3.flatten()
        return np.array([
            [  0.0, -v[2],  v[1]],
            [ v[2],   0.0, -v[0]],
            [-v[1],  v[0],   0.0]
        ])

    @staticmethod
    def vee_map(skew_mat: np.ndarray) -> np.ndarray:
        """Inverse skew-symmetric operator (vee map) from so(3) to R^3."""
        return np.array([
            skew_mat[2, 1],
            skew_mat[0, 2],
            skew_mat[1, 0]
        ])

    def compute_rotation_matrix_from_desired_thrust(
        self,
        thrust_vector_ned: np.ndarray,
        desired_yaw_rad: float
    ) -> np.ndarray:
        """
        Synthesizes desired rotation matrix R_d = [b1_d, b2_d, b3_d]
        where b3_d aligns with the commanded thrust vector.
        """
        # b3_d aligns with normalized desired thrust vector (Down/Z body axis points down)
        norm_t = np.linalg.norm(thrust_vector_ned)
        if norm_t < 1e-4:
            b3_d = np.array([0.0, 0.0, 1.0])
        else:
            b3_d = thrust_vector_ned / norm_t

        # Desired heading intermediate vector: c1_d = [cos(psi), sin(psi), 0]
        c1_d = np.array([math.cos(desired_yaw_rad), math.sin(desired_yaw_rad), 0.0])

        # b2_d = (b3_d x c1_d) / ||b3_d x c1_d||
        b2_d_raw = np.cross(b3_d, c1_d)
        norm_b2 = np.linalg.norm(b2_d_raw)
        if norm_b2 < 1e-4:
            b2_d = np.array([0.0, 1.0, 0.0])
        else:
            b2_d = b2_d_raw / norm_b2

        # b1_d = b2_d x b3_d
        b1_d = np.cross(b2_d, b3_d)

        R_d = np.column_stack([b1_d, b2_d, b3_d])
        return R_d

    def compute_control_torques(
        self,
        current_R: np.ndarray,
        current_omega_rad_s: np.ndarray,
        desired_R: np.ndarray,
        desired_omega_rad_s: np.ndarray = np.zeros(3),
        desired_omega_dot_rad_s2: np.ndarray = np.zeros(3)
    ) -> Dict:
        """
        Calculates 3-axis decoupled control moments on SO(3).
        """
        R = current_R
        R_d = desired_R
        omega = current_omega_rad_s
        omega_d = desired_omega_rad_s
        d_omega_d = desired_omega_dot_rad_s2

        # 1. Manifold Attitude Error Vector: e_R = 1/2 * (R_d^T * R - R^T * R_d)^vee
        R_d_T_R = R_d.T @ R
        skew_e_R = 0.5 * (R_d_T_R - R_d_T_R.T)
        e_R = self.vee_map(skew_e_R)

        # 2. Angular Velocity Error Vector: e_Omega = Omega - R^T * R_d * Omega_d
        omega_d_in_body = R_d_T_R.T @ omega_d
        e_Omega = omega - omega_d_in_body

        # 3. Gyroscopic and Feedforward Compensation Terms
        gyroscopic_torque = np.cross(omega, self.J @ omega)
        
        # d/dt (R^T * R_d * Omega_d) = -Omega^hat * R^T * R_d * Omega_d + R^T * R_d * dot(Omega_d)
        feedforward_accel = -np.cross(omega, omega_d_in_body) + (R_d_T_R.T @ d_omega_d)
        feedforward_torque = self.J @ feedforward_accel

        # 4. Geometric Feedback Control Law
        tau_raw = -self.kR * e_R - self.kOmega * e_Omega + gyroscopic_torque + feedforward_torque

        # Torque Saturation
        tau_cmd = np.clip(tau_raw, -self.tau_max, self.tau_max)

        # Normalized Attitude Error Function Psi = 1/2 * tr(I - R_d^T * R)
        psi_error = 0.5 * float(np.trace(np.eye(3) - R_d_T_R))

        return {
            "commanded_torques_nm": (round(float(tau_cmd[0]), 4),
                                     round(float(tau_cmd[1]), 4),
                                     round(float(tau_cmd[2]), 4)),
            "attitude_error_norm": round(float(np.linalg.norm(e_R)), 4),
            "angular_rate_error_norm": round(float(np.linalg.norm(e_Omega)), 4),
            "lyapunov_potential_psi": round(psi_error, 5),
            "torque_saturated": bool(np.any(np.abs(tau_raw) > self.tau_max)),
            "control_stability_status": "ASYMPTOTICALLY_STABLE" if psi_error < 1.0 else "LARGE_ANGLE_ACQUISITION"
        }

if __name__ == "__main__":
    controller = GeometricSO3AttitudeController()

    print("==========================================================================")
    print("      SO(3) LIE GROUP NONLINEAR GEOMETRIC ATTITUDE CONTROLLER AUDIT       ")
    print("==========================================================================")

    # Initial state: Hovering level (Identity)
    R_current = np.eye(3)
    omega_current = np.array([0.05, -0.02, 0.0])

    # Commanded high-speed roll/bank target: 45° bank angle, 10° pitch up
    R_desired = Rotation.from_euler('ZYX', [0.0, math.radians(10.0), math.radians(45.0)]).as_matrix()

    report = controller.compute_control_torques(
        current_R=R_current,
        current_omega_rad_s=omega_current,
        desired_R=R_desired,
        desired_omega_rad_s=np.array([0.0, 0.0, 0.0])
    )

    for k, v in report.items():
        print(f"  • {k:30s}: {v}")
    print("==========================================================================\n")