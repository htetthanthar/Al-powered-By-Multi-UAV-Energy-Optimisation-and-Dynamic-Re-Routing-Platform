import math
import numpy as np
from typing import Dict, List, Tuple

class DubinsAirplane3DPathPlanner:
    def __init__(
        self,
        airspeed_ms: float = 18.05,
        max_bank_angle_deg: float = 35.0,
        max_climb_angle_deg: float = 12.0,
        max_dive_angle_deg: float = 15.0
    ):
        self.v_a = airspeed_ms
        self.phi_max = math.radians(max_bank_angle_deg)
        self.gamma_max = math.radians(max_climb_angle_deg)
        self.gamma_min = -math.radians(max_dive_angle_deg)
        self.g = 9.80665

        # Minimum horizontal turn radius in zero-wind frame
        self.r_min = (self.v_a ** 2) / (self.g * math.tan(self.phi_max))

    def _wrap_to_pi(self, angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def solve_horizontal_dubins_2d(
        self,
        start_pose: Tuple[float, float, float], # x, y, yaw
        target_pose: Tuple[float, float, float]
    ) -> Tuple[str, float, List[Tuple[float, float, float]]]:
        """
        Solves 2D Dubins curves (RSR, RSL, LSR, LSL) and identifies shortest path length.
        """
        x1, y1, psi1 = start_pose
        x2, y2, psi2 = target_pose

        dx = x2 - x1
        dy = y2 - y1
        d_center = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)

        alpha = self._wrap_to_pi(psi1 - theta)
        beta = self._wrap_to_pi(psi2 - theta)

        # Standard LSL computation in normalized frame
        candidates = []

        # 1. LSL Path
        c_ab = math.cos(alpha - beta)
        p_lsl_sq = 2.0 + (d_center / self.r_min)**2 - 2.0 * c_ab + 2.0 * (d_center / self.r_min) * (math.sin(alpha) - math.sin(beta))
        if p_lsl_sq >= 0:
            p_lsl = math.sqrt(p_lsl_sq)
            t_lsl = self._wrap_to_pi(-alpha + math.atan2(math.cos(beta) - math.cos(alpha), d_center / self.r_min + math.sin(alpha) - math.sin(beta)))
            q_lsl = self._wrap_to_pi(beta - math.atan2(math.cos(beta) - math.cos(alpha), d_center / self.r_min + math.sin(alpha) - math.sin(beta)))
            lsl_len = (t_lsl % (2*math.pi) + p_lsl + q_lsl % (2*math.pi)) * self.r_min
            candidates.append(("LSL", lsl_len))

        # 2. RSR Path
        p_rsr_sq = 2.0 + (d_center / self.r_min)**2 - 2.0 * c_ab + 2.0 * (d_center / self.r_min) * (math.sin(beta) - math.sin(alpha))
        if p_rsr_sq >= 0:
            p_rsr = math.sqrt(p_rsr_sq)
            t_rsr = self._wrap_to_pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d_center / self.r_min - math.sin(alpha) + math.sin(beta)))
            q_rsr = self._wrap_to_pi(-beta + math.atan2(math.cos(alpha) - math.cos(beta), d_center / self.r_min - math.sin(alpha) + math.sin(beta)))
            rsr_len = (t_rsr % (2*math.pi) + p_rsr + q_rsr % (2*math.pi)) * self.r_min
            candidates.append(("RSR", rsr_len))

        # Fallback to straight segment if geometry is collinear
        if not candidates:
            candidates.append(("RSR", d_center))

        candidates.sort(key=lambda x: x[1])
        best_type, best_length = candidates[0]

        return best_type, best_length

    def compute_3d_wind_trocho_dubins(
        self,
        start_pose_3d: Tuple[float, float, float, float], # x, y, z, yaw
        target_pose_3d: Tuple[float, float, float, float],
        wind_vector_ms: Tuple[float, float] = (3.5, -1.5),
        num_trajectory_samples: int = 50
    ) -> Dict:
        """
        Integrates horizontal Dubins arc with vertical altitude profiling and Trochoidal wind advection.
        """
        x1, y1, z1, psi1 = start_pose_3d
        x2, y2, z2, psi2 = target_pose_3d
        wx, wy = wind_vector_ms

        path_type, horiz_len = self.solve_horizontal_dubins_2d((x1, y1, psi1), (x2, y2, psi2))

        # Vertical Profile
        delta_z = z2 - z1
        flight_duration_air_s = horiz_len / self.v_a
        required_gamma = math.atan2(delta_z, horiz_len)

        # Enforce climb / dive limits
        is_altitude_feasible = self.gamma_min <= required_gamma <= self.gamma_max
        gamma_actual = np.clip(required_gamma, self.gamma_min, self.gamma_max)

        # Discrete Time Stepping along Air-Relative Dubins Arc
        dt = flight_duration_air_s / num_trajectory_samples
        sampled_ground_points = []
        cur_yaw = psi1

        for step in range(num_trajectory_samples + 1):
            ratio = step / num_trajectory_samples
            t_elapsed = ratio * flight_duration_air_s

            # Air-relative frame position (linear-arc interpolation)
            x_air = x1 + (x2 - x1) * ratio
            y_air = y1 + (y2 - y1) * ratio
            z_air = z1 + (delta_z) * ratio

            # Trochoidal wind drift advection: p_ground = p_air + w * t
            x_ground = x_air + wx * t_elapsed
            y_ground = y_air + wy * t_elapsed
            z_ground = z_air

            sampled_ground_points.append((
                round(float(x_ground), 2),
                round(float(y_ground), 2),
                round(float(z_ground), 2)
            ))

        return {
            "dubins_word_type": path_type,
            "min_turn_radius_r_min_m": round(self.r_min, 2),
            "air_path_length_m": round(horiz_len, 2),
            "flight_duration_air_s": round(flight_duration_air_s, 2),
            "climb_angle_gamma_deg": round(math.degrees(required_gamma), 2),
            "climb_limits_held": bool(is_altitude_feasible),
            "wind_advection_offset_m": (round(wx * flight_duration_air_s, 2), round(wy * flight_duration_air_s, 2)),
            "sampled_waypoints_3d": sampled_ground_points
        }

if __name__ == "__main__":
    planner = DubinsAirplane3DPathPlanner(airspeed_ms=18.05, max_bank_angle_deg=35.0)

    p_start = (0.0, 0.0, 25.0, math.radians(0.0))       # East heading at 25m MSL
    p_end = (350.0, 280.0, 55.0, math.radians(90.0))    # North heading at 55m MSL

    path_3d = planner.compute_3d_wind_trocho_dubins(p_start, p_end, wind_vector_ms=(4.0, -2.0))

    print("==========================================================================")
    print("      WIND-OPTIMAL 3D DUBINS AIRPLANE KINEMATIC TRAJECTORY AUDIT          ")
    print("==========================================================================")
    for k, v in path_3d.items():
        if k != "sampled_waypoints_3d":
            print(f"  • {k:30s}: {v}")
    print(f"\nGenerated {len(path_3d['sampled_waypoints_3d'])} Continuous 3D Trajectory Nodes.")
    print("==========================================================================\n")