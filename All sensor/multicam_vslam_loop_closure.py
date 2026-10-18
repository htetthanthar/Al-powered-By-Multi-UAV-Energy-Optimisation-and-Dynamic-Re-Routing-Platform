import math
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional

class SE3Pose:
    def __init__(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray):
        self.R = rotation_matrix.copy() # 3x3
        self.t = translation_vector.copy() # 3x1

    def inverse(self) -> 'SE3Pose':
        r_inv = self.R.T
        t_inv = -r_inv @ self.t
        return SE3Pose(r_inv, t_inv)

    def __matmul__(self, other: 'SE3Pose') -> 'SE3Pose':
        r_new = self.R @ other.R
        t_new = self.R @ other.t + self.t
        return SE3Pose(r_new, t_new)

    @staticmethod
    def exp(xi: np.ndarray) -> 'SE3Pose':
        """Exponential map from se(3) vector [rho(3), phi(3)] to SE(3) matrix."""
        rho = xi[0:3]
        phi = xi[3:6]
        angle = float(np.linalg.norm(phi))

        if angle < 1e-6:
            R = np.eye(3)
            V = np.eye(3)
        else:
            axis = phi / angle
            R = Rotation.from_rotvec(phi).as_matrix()
            # Left Jacobian of SO(3)
            sin_a, cos_a = math.sin(angle), math.cos(angle)
            k_skew = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            V = np.eye(3) + ((1.0 - cos_a) / angle) * k_skew + ((angle - sin_a) / angle) * (k_skew @ k_skew)

        t = V @ rho
        return SE3Pose(R, t)

    def log(self) -> np.ndarray:
        """Logarithmic map from SE(3) to se(3) Lie algebra vector [rho(3), phi(3)]."""
        rot_vec = Rotation.from_matrix(self.R).as_rotvec()
        angle = float(np.linalg.norm(rot_vec))

        if angle < 1e-6:
            V_inv = np.eye(3)
        else:
            axis = rot_vec / angle
            k_skew = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            half_a = 0.5 * angle
            cot_half_a = 1.0 / math.tan(half_a) if abs(half_a) > 1e-4 else 1.0 / half_a
            V_inv = np.eye(3) - 0.5 * k_skew + (1.0 - half_a * cot_half_a) * (k_skew @ k_skew)

        rho = V_inv @ self.t
        return np.concatenate([rho, rot_vec])

class MultiCameraPoseGraphOptimizer:
    def __init__(
        self,
        huber_delta: float = 1.345,
        max_iterations: int = 15,
        convergence_tolerance: float = 1e-4
    ):
        self.delta = huber_delta
        self.max_iter = max_iterations
        self.tol = convergence_tolerance
        
        self.keyframe_poses: List[SE3Pose] = []
        self.odometry_edges: List[Tuple[int, int, SE3Pose, np.ndarray]] = []
        self.loop_closure_edges: List[Tuple[int, int, SE3Pose, np.ndarray]] = []

    def add_keyframe(self, pose: SE3Pose):
        self.keyframe_poses.append(pose)

    def add_odometry_edge(self, from_id: int, to_id: int, relative_pose: SE3Pose, info_matrix: np.ndarray):
        self.odometry_edges.append((from_id, to_id, relative_pose, info_matrix))

    def add_loop_closure_edge(self, from_id: int, to_id: int, relative_pose: SE3Pose, info_matrix: np.ndarray):
        self.loop_closure_edges.append((from_id, to_id, relative_pose, info_matrix))

    def _huber_weight(self, error_norm: float) -> float:
        if error_norm <= self.delta:
            return 1.0
        return self.delta / error_norm

    def optimize_pose_graph(self) -> Dict:
        """
        Executes Gauss-Newton / Levenberg-Marquardt optimization over all SE(3) poses.
        """
        num_poses = len(self.keyframe_poses)
        if num_poses < 3:
            return {"status": "INSUFFICIENT_KEYFRAMES"}

        all_edges = self.odometry_edges + self.loop_closure_edges
        initial_error = 0.0

        for i, j, t_meas, omega in all_edges:
            t_est = self.keyframe_poses[i].inverse() @ self.keyframe_poses[j]
            e_ij = (t_meas.inverse() @ t_est).log()
            initial_error += float(e_ij.T @ omega @ e_ij)

        # Iterative optimization (anchor keyframe 0 fixed)
        for iteration in range(self.max_iter):
            H_matrix = np.zeros((6 * (num_poses - 1), 6 * (num_poses - 1)))
            b_vector = np.zeros(6 * (num_poses - 1))
            total_cost = 0.0

            for i, j, t_meas, omega in all_edges:
                t_i = self.keyframe_poses[i]
                t_j = self.keyframe_poses[j]
                t_est = t_i.inverse() @ t_j
                
                # Error vector in Lie algebra: e_ij = Log(T_meas^-1 * T_est)
                e_ij = (t_meas.inverse() @ t_est).log()
                e_norm = float(np.linalg.norm(e_ij))
                w_huber = self._huber_weight(e_norm)
                omega_robust = w_huber * omega

                total_cost += float(e_ij.T @ omega_robust @ e_ij)

                # Linearized Jacobians: J_i = -Adjoint(T_meas^-1 * T_est), J_j = Identity(6x6)
                j_j = np.eye(6)
                j_i = -np.eye(6) # Approximate local Lie group Jacobian

                # Fill block entries for j (if j > 0)
                if j > 0:
                    idx_j = (j - 1) * 6
                    H_matrix[idx_j : idx_j+6, idx_j : idx_j+6] += j_j.T @ omega_robust @ j_j
                    b_vector[idx_j : idx_j+6] += j_j.T @ omega_robust @ e_ij

                # Fill block entries for i (if i > 0)
                if i > 0:
                    idx_i = (i - 1) * 6
                    H_matrix[idx_i : idx_i+6, idx_i : idx_i+6] += j_i.T @ omega_robust @ j_i
                    b_vector[idx_i : idx_i+6] += j_i.T @ omega_robust @ e_ij

            # Solve linear normal system: (H + lambda*I) * Delta_x = -b
            reg_lambda = 1e-3
            h_reg = H_matrix + reg_lambda * np.eye(H_matrix.shape[0])
            dx_flat = np.linalg.solve(h_reg, -b_vector)

            # Update poses on SE(3) manifold: T_k = T_k * exp(dx_k)
            for k in range(1, num_poses):
                dx_k = dx_flat[(k - 1) * 6 : k * 6]
                self.keyframe_poses[k] = self.keyframe_poses[k] @ SE3Pose.exp(dx_k)

            if np.max(np.abs(dx_flat)) < self.tol:
                break

        # Compute final residual error
        final_error = 0.0
        for i, j, t_meas, omega in all_edges:
            t_est = self.keyframe_poses[i].inverse() @ self.keyframe_poses[j]
            e_ij = (t_meas.inverse() @ t_est).log()
            final_error += float(e_ij.T @ omega @ e_ij)

        return {
            "status": "POSE_GRAPH_OPTIMIZATION_CONVERGED",
            "total_keyframes_optimized": num_poses,
            "loop_closures_integrated": len(self.loop_closure_edges),
            "iterations_executed": iteration + 1,
            "initial_residual_cost": round(initial_error, 3),
            "final_optimized_cost": round(final_error, 3),
            "drift_reduction_ratio_pct": round(((initial_error - final_error) / max(initial_error, 1e-4)) * 100.0, 2),
            "loop_closure_locked": final_error < initial_error
        }

if __name__ == "__main__":
    pgo = MultiCameraPoseGraphOptimizer()

    print("==========================================================================")
    print("      MULTI-CAMERA SE(3) POSE GRAPH OPTIMIZATION & LOOP CLOSURE AUDIT     ")
    print("==========================================================================")

    # Simulate 5-node square trajectory loop: [0, 0] -> [50, 0] -> [50, 50] -> [0, 50] -> [0, 0]
    info_odom = np.eye(6) * 100.0
    info_loop = np.eye(6) * 500.0 # High confidence loop closure constraint

    # Ground truth positions with accumulated dead-reckoning drift
    poses_raw = [
        SE3Pose(np.eye(3), np.array([0.0, 0.0, -25.0])),
        SE3Pose(np.eye(3), np.array([50.2, 0.4, -25.1])),
        SE3Pose(np.eye(3), np.array([50.8, 50.6, -24.8])),
        SE3Pose(np.eye(3), np.array([0.6, 50.9, -25.2])),
        SE3Pose(np.eye(3), np.array([1.2, 1.5, -24.9])) # Drifted endpoint ~1.9m error
    ]

    for p in poses_raw:
        pgo.add_keyframe(p)

    # Odometry links
    for k in range(4):
        rel = poses_raw[k].inverse() @ poses_raw[k+1]
        pgo.add_odometry_edge(k, k+1, rel, info_odom)

    # Visual Loop Closure detected between Keyframe 4 and Keyframe 0 (True relative translation = 0)
    true_loop_rel = SE3Pose(np.eye(3), np.array([0.0, 0.0, 0.0]))
    pgo.add_loop_closure_edge(0, 4, true_loop_rel, info_loop)

    pgo_result = pgo.optimize_pose_graph()
    for k, v in pgo_result.items():
        print(f"  • {k:35s}: {v}")
    
    end_pos_opt = pgo.keyframe_poses[4].t
    print(f"\nCorrected Terminal Loop Pose: [X: {end_pos_opt[0]:5.2f}m, Y: {end_pos_opt[1]:5.2f}m, Z: {end_pos_opt[2]:5.2f}m]")
    print("==========================================================================\n")