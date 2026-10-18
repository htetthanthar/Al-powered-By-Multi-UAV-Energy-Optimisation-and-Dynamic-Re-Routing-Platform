import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class FaultTolerantControlAllocator:
    def __init__(
        self,
        num_rotors: int = 6,  # Hexacopter platform configuration for full single-rotor fault tolerance
        arm_length_m: float = 0.35,
        thrust_coeff_ct: float = 1.2e-5,
        drag_coeff_cq: float = 2.4e-7,
        max_motor_thrust_n: float = 22.0,
        min_motor_thrust_n: float = 1.0
    ):
        self.num_rotors = num_rotors
        self.arm_len = arm_length_m
        self.ct = thrust_coeff_ct
        self.cq = drag_coeff_cq
        self.max_t = max_motor_thrust_n
        self.min_t = min_motor_thrust_n
        
        # Nominal Hexacopter Geometry (Rotor angles in radians: 0, 60, 120, 180, 240, 300 deg)
        # Rotational directions: alternating CW (-1) and CCW (+1)
        self.rotor_angles = [i * (2.0 * math.pi / num_rotors) for i in range(num_rotors)]
        self.rotor_directions = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        
        self.nominal_allocation_matrix = self._build_allocation_matrix(active_rotors=[True] * num_rotors)
        self.active_rotor_mask = [True] * num_rotors

    def _build_allocation_matrix(self, active_rotors: List[bool]) -> np.ndarray:
        """
        Builds the 4 x N control effectiveness matrix B:
          [F_z, tau_phi, tau_theta, tau_psi]^T = B * [T_1, T_2, ..., T_N]^T
        """
        b_matrix = np.zeros((4, self.num_rotors))
        for i in range(self.num_rotors):
            if not active_rotors[i]:
                continue
            theta = self.rotor_angles[i]
            direction = self.rotor_directions[i]
            
            # Row 0: Total Collective Thrust (F_z)
            b_matrix[0, i] = 1.0
            # Row 1: Roll Torque (tau_phi = T * arm * sin(theta))
            b_matrix[1, i] = -self.arm_len * math.sin(theta)
            # Row 2: Pitch Torque (tau_theta = T * arm * cos(theta))
            b_matrix[2, i] = self.arm_len * math.cos(theta)
            # Row 3: Yaw Reaction Torque (tau_psi = direction * (c_q / c_t) * T)
            b_matrix[3, i] = direction * (self.cq / self.ct)

        return b_matrix

    def detect_actuator_failure(
        self,
        commanded_thrusts_n: np.ndarray,
        measured_angular_accel_rad_s2: np.ndarray,
        measured_vertical_accel_ms2: float,
        inertia_matrix: np.ndarray,
        total_mass_kg: float
    ) -> Optional[int]:
        """
        Monitors residual vector: e_tau = J * alpha_meas - B * T_cmd.
        Flags the damaged rotor index if residuals exceed threshold.
        """
        expected_wrench = self.nominal_allocation_matrix @ commanded_thrusts_n
        measured_wrench = np.array([
            total_mass_kg * (measured_vertical_accel_ms2 + 9.81),
            inertia_matrix[0, 0] * measured_angular_accel_rad_s2[0],
            inertia_matrix[1, 1] * measured_angular_accel_rad_s2[1],
            inertia_matrix[2, 2] * measured_angular_accel_rad_s2[2]
        ])

        residual = measured_wrench - expected_wrench
        residual_norm = np.linalg.norm(residual[1:3]) # Focus on roll/pitch disturbance

        if residual_norm > 4.5:  # Significant uncommanded torque mismatch
            # Isolate failed rotor by minimizing projection error
            best_rotor_fault = None
            min_projection_error = float('inf')

            for r_idx in range(self.num_rotors):
                test_mask = [True] * self.num_rotors
                test_mask[r_idx] = False
                b_degraded = self._build_allocation_matrix(test_mask)
                proj_error = np.linalg.norm(measured_wrench - b_degraded @ commanded_thrusts_n)
                if proj_error < min_projection_error:
                    min_projection_error = proj_error
                    best_rotor_fault = r_idx

            return best_rotor_fault

        return None

    def reallocate_control_authority(
        self,
        desired_wrench: np.ndarray,
        failed_rotor_idx: Optional[int]
    ) -> Tuple[np.ndarray, bool]:
        """
        Computes re-allocated individual rotor thrusts using weighted pseudo-inverse.
        Degrades yaw control weight when necessary to prioritize roll, pitch, and altitude.
        """
        if failed_rotor_idx is not None:
            self.active_rotor_mask[failed_rotor_idx] = False

        b_active = self._build_allocation_matrix(self.active_rotor_mask)

        # Priority weights: [F_z=100.0, Roll=80.0, Pitch=80.0, Yaw=2.0 (Relaxed on failure)]
        w_diag = [100.0, 80.0, 80.0, 2.0 if failed_rotor_idx is not None else 30.0]
        w_mat = np.diag(w_diag)

        # Weighted Pseudo-Inverse: B_pinv = (B^T * W * B + lambda * I)^(-1) * B^T * W
        reg_lambda = 1e-4
        h_matrix = b_active.T @ w_mat @ b_active + reg_lambda * np.eye(self.num_rotors)
        b_pinv = np.linalg.inv(h_matrix) @ b_active.T @ w_mat

        # Solve raw motor thrusts
        raw_thrusts = b_pinv @ desired_wrench

        # Enforce zero thrust on disabled motor and clip within saturation bounds
        saturated_thrusts = np.clip(raw_thrusts, self.min_t, self.max_t)
        for i in range(self.num_rotors):
            if not self.active_rotor_mask[i]:
                saturated_thrusts[i] = 0.0

        achieved_wrench = b_active @ saturated_thrusts
        tracking_fidelity = np.linalg.norm(achieved_wrench[0:3] - desired_wrench[0:3]) < 2.5

        return saturated_thrusts, tracking_fidelity

if __name__ == "__main__":
    ftc = FaultTolerantControlAllocator(num_rotors=6)
    
    # Hover wrench demand for 6.14 kg hexacopter: F_z = 60.2 N, Torques = 0
    wrench_demand = np.array([6.14 * 9.80665, 0.0, 0.0, 0.0])
    
    # 1. Nominal Allocation
    nominal_thrusts, ok_nom = ftc.reallocate_control_authority(wrench_demand, failed_rotor_idx=None)

    print("==========================================================================")
    print("      FAULT-TOLERANT CONTROL (FTC) ACTUATOR RE-ALLOCATION TEST            ")
    print("==========================================================================")
    print("Nominal 6-Rotor Hover Thrust Profile:")
    for idx, t in enumerate(nominal_thrusts):
        print(f"  • Motor {idx+1}: {t:5.2f} N")

    # 2. Complete Failure of Rotor #2 (Index 1)
    failed_id = 1
    emergency_thrusts, ok_em = ftc.reallocate_control_authority(wrench_demand, failed_rotor_idx=failed_id)
    
    print(f"\nEmergency Re-Allocated Thrust Profile (Motor #{failed_id+1} Burnout):")
    for idx, t in enumerate(emergency_thrusts):
        status = "DISABLED (0.0N)" if idx == failed_id else f"{t:5.2f} N"
        print(f"  • Motor {idx+1}: {status}")
    print(f"\nAttitude Tracking Feasibility : {'MAINTAINED (STABLE)' if ok_em else 'DEGRADED (CONTROLLED DESCENT)'}")
    print("==========================================================================\n")