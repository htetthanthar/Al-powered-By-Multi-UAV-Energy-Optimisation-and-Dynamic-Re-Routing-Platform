import math
import numpy as np
from typing import Dict, Tuple, Optional

class GNSSMovingBaselineAttitudeEstimator:
    def __init__(
        self,
        antenna_baseline_length_m: float = 0.650, # 65.0 cm physical separation across airframe
        baseline_tolerance_m: float = 0.025,      # +/- 25 mm rigid-body structural tolerance
        carrier_wavelength_l1_m: float = 0.19029, # GPS L1
        phase_noise_std_cycles: float = 0.020
    ):
        self.l_base = antenna_baseline_length_m
        self.tol = baseline_tolerance_m
        self.lambda1 = carrier_wavelength_l1_m
        self.sigma_phi = phase_noise_std_cycles * self.lambda1

        self.last_valid_heading_rad = 0.0
        self.last_valid_pitch_rad = 0.0

    def compute_moving_baseline_attitude(
        self,
        primary_antenna_ecef: np.ndarray,
        secondary_antenna_ecef: np.ndarray,
        reference_lat_deg: float,
        reference_lon_deg: float
    ) -> Dict:
        """
        Transforms ECEF baseline vector into local North-East-Down (NED) coordinates,
        validates the rigid-body baseline length invariant, and computes yaw and pitch.
        """
        delta_ecef = secondary_antenna_ecef - primary_antenna_ecef

        # WGS84 ECEF to NED Rotation Matrix
        lat_rad = math.radians(reference_lat_deg)
        lon_rad = math.radians(reference_lon_deg)

        sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
        sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)

        r_ecef_to_ned = np.array([
            [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
            [          -sin_lon,            cos_lon,      0.0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
        ])

        baseline_ned = r_ecef_to_ned @ delta_ecef
        measured_length = float(np.linalg.norm(baseline_ned))

        # Rigid Body Constraint Invariant Verification
        baseline_length_error = abs(measured_length - self.l_base)
        is_geom_valid = baseline_length_error <= self.tol

        b_n, b_e, b_d = baseline_ned[0], baseline_ned[1], baseline_ned[2]
        horizontal_proj = math.hypot(b_n, b_e)

        if is_geom_valid and horizontal_proj > 1e-4:
            true_heading_rad = math.atan2(b_e, b_n) % (2.0 * math.pi)
            pitch_rad = math.atan2(-b_d, horizontal_proj)

            self.last_valid_heading_rad = true_heading_rad
            self.last_valid_pitch_rad = pitch_rad

            # Theoretical angular standard deviation
            heading_std_deg = math.degrees(self.sigma_phi / (self.l_base * max(math.cos(pitch_rad), 0.1)))
            pitch_std_deg = math.degrees(self.sigma_phi / self.l_base)
            status = "RTK_MOVING_BASELINE_FIXED"
        else:
            true_heading_rad = self.last_valid_heading_rad
            pitch_rad = self.last_valid_pitch_rad
            heading_std_deg = 15.0
            pitch_std_deg = 10.0
            status = "BASELINE_LENGTH_VIOLATION_FALLBACK_DEAD_RECKONING"

        return {
            "status": status,
            "measured_baseline_length_m": round(measured_length, 4),
            "expected_baseline_length_m": round(self.l_base, 4),
            "baseline_error_mm": round(baseline_length_error * 1000.0, 2),
            "rigid_body_constraint_held": is_geom_valid,
            "true_heading_deg": round(math.degrees(true_heading_rad), 2),
            "baseline_pitch_deg": round(math.degrees(pitch_rad), 2),
            "heading_accuracy_std_deg": round(heading_std_deg, 3),
            "pitch_accuracy_std_deg": round(pitch_std_deg, 3),
            "compass_magnetic_immunity": True
        }

if __name__ == "__main__":
    estimator = GNSSMovingBaselineAttitudeEstimator(antenna_baseline_length_m=0.650)

    # Primary antenna at origin
    primary_pos = np.array([ -1498200.000, 6192100.000, 149300.000 ])
    
    # Secondary antenna placed 0.65m away oriented at 045.0° true heading and +3.0° pitch
    psi_sim = math.radians(45.0)
    theta_sim = math.radians(3.0)
    b_n_sim = 0.650 * math.cos(theta_sim) * math.cos(psi_sim)
    b_e_sim = 0.650 * math.cos(theta_sim) * math.sin(psi_sim)
    b_d_sim = -0.650 * math.sin(theta_sim)
    
    # Rotate NED to ECEF
    lat_sg, lon_sg = 1.3521, 103.8198
    lat_r, lon_r = math.radians(lat_sg), math.radians(lon_sg)
    r_ned_to_ecef = np.array([
        [-math.sin(lat_r) * math.cos(lon_r), -math.sin(lon_r), -math.cos(lat_r) * math.cos(lon_r)],
        [-math.sin(lat_r) * math.sin(lon_r),  math.cos(lon_r), -math.cos(lat_r) * math.sin(lon_r)],
        [ math.cos(lat_r),                                 0.0, -math.sin(lat_r)]
    ])
    
    sim_delta_ecef = r_ned_to_ecef @ np.array([b_n_sim, b_e_sim, b_d_sim])
    secondary_pos = primary_pos + sim_delta_ecef + np.random.normal(0, 0.002, 3)

    attitude = estimator.compute_moving_baseline_attitude(primary_pos, secondary_pos, lat_sg, lon_sg)

    print("==========================================================================")
    print("      DUAL-ANTENNA GNSS MOVING BASELINE ATTITUDE DETERMINATION AUDIT     ")
    print("==========================================================================")
    for k, v in attitude.items():
        print(f"  • {k:35s}: {v}")
    print("==========================================================================\n")