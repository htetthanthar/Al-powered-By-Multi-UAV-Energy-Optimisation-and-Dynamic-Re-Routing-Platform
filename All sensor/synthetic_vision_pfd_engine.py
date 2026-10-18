import math
import numpy as np
from typing import Dict, List, Tuple

class SyntheticVisionPFDEngine:
    def __init__(
        self,
        display_resolution_px: Tuple[int, int] = (1920, 1080),
        horizontal_fov_deg: float = 65.0,
        tunnel_gate_half_width_m: float = 4.0,
        tunnel_gate_half_height_m: float = 3.0
    ):
        self.w_px, self.h_px = display_resolution_px
        self.fov_h = math.radians(horizontal_fov_deg)
        self.g_w = tunnel_gate_half_width_m
        self.g_h = tunnel_gate_half_height_m

        self.cx = self.w_px / 2.0
        self.cy = self.h_px / 2.0
        self.focal_px = (self.w_px / 2.0) / math.tan(self.fov_h / 2.0)

    def _euler_to_rotation_matrix(self, roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
        """Standard Z-Y-X Tait-Bryan rotation matrix from NED to Body frame."""
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

        R = np.array([
            [cp * cy, cp * sy, -sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
            [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp]
        ])
        return R

    def project_3d_point_to_pfd(
        self,
        point_ned: np.ndarray,
        uav_pos_ned: np.ndarray,
        r_ned_to_body: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """Projects 3D NED world point onto 2D PFD screen coordinates."""
        p_rel_ned = point_ned - uav_pos_ned
        p_body = r_ned_to_body @ p_rel_ned # Forward (X), Right (Y), Down (Z)

        # X_body is forward depth along optical axis
        if p_body[0] <= 0.5:
            return None, -1.0 # Point behind aircraft

        # Screen Projection: Camera X is Right (+Y_body), Camera Y is Down (+Z_body)
        u_screen = self.cx + self.focal_px * (p_body[1] / p_body[0])
        v_screen = self.cy + self.focal_px * (p_body[2] / p_body[0])

        return (int(round(u_screen)), int(round(v_screen))), float(p_body[0])

    def render_synthetic_vision_frame(
        self,
        uav_pos_ned: Tuple[float, float, float],
        uav_vel_body_ms: Tuple[float, float, float], # [u_fwd, v_right, w_down]
        uav_attitude_rad: Tuple[float, float, float], # [roll, pitch, yaw]
        trajectory_waypoints_ned: List[Tuple[float, float, float]]
    ) -> Dict:
        """
        Synthesizes complete PFD overlay: Flight Path Vector (FPV), Pitch Ladder rungs,
        and Highway-in-the-Sky (HITS) 3D tunnel bounding gates.
        """
        p_uav = np.array(uav_pos_ned)
        vx_b, vy_b, vz_b = uav_vel_body_ms
        roll, pitch, yaw = uav_attitude_rad

        r_mat = self._euler_to_rotation_matrix(roll, pitch, yaw)

        # 1. Flight Path Vector (FPV) / Velocity Vector Projection
        if vx_b > 1.0:
            fpv_u = int(round(self.cx + self.focal_px * (vy_b / vx_b)))
            fpv_v = int(round(self.cy + self.focal_px * (vz_b / vx_b)))
            fpv_screen = (fpv_u, fpv_v)
        else:
            fpv_screen = (int(self.cx), int(self.cy))

        # 2. Highway-in-the-Sky (HITS) 3D Tunnel Gates
        tunnel_gates = []
        for wp_idx, wp in enumerate(trajectory_waypoints_ned):
            wp_pt = np.array(wp)
            # Create 4 bounding rectangular corners for tunnel gate
            corners = [
                wp_pt + np.array([0.0, -self.g_w, -self.g_h]), # Top-Left
                wp_pt + np.array([0.0,  self.g_w, -self.g_h]), # Top-Right
                wp_pt + np.array([0.0,  self.g_w,  self.g_h]), # Bottom-Right
                wp_pt + np.array([0.0, -self.g_w,  self.g_h])  # Bottom-Left
            ]

            projected_corners = []
            valid = True
            dist_to_gate = 0.0

            for c in corners:
                proj_pt, d_fwd = self.project_3d_point_to_pfd(c, p_uav, r_mat)
                if proj_pt is None or d_fwd > 250.0:
                    valid = False
                    break
                projected_corners.append(proj_pt)
                dist_to_gate = d_fwd

            if valid:
                tunnel_gates.append({
                    "gate_id": wp_idx + 1,
                    "distance_ahead_m": round(dist_to_gate, 1),
                    "screen_polygon_px": projected_corners
                })

        # 3. Artificial Horizon Line Parameters
        horizon_center_y = int(round(self.cy + self.focal_px * math.tan(pitch)))
        roll_angle_deg = math.degrees(roll)

        return {
            "pfd_resolution": (self.w_px, self.h_px),
            "flight_path_vector_px": fpv_screen,
            "artificial_horizon_center_y_px": horizon_center_y,
            "horizon_roll_tilt_deg": round(roll_angle_deg, 2),
            "visible_highway_gates_count": len(tunnel_gates),
            "highway_in_the_sky_gates": tunnel_gates,
            "synthetic_vision_mode": "IFR_DEGRADED_VISIBILITY_TUNNEL_ACTIVE"
        }

if __name__ == "__main__":
    svs = SyntheticVisionPFDEngine(display_resolution_px=(1920, 1080), horizontal_fov_deg=65.0)

    # UAV cruising forward at 18.05 m/s with 3.0° pitch up and 5.0° right bank
    pos_now = (0.0, 0.0, -25.0)
    vel_body = (18.05, 0.5, -0.2) # Slight right drift and gentle climb
    attitude = (math.radians(5.0), math.radians(3.0), math.radians(0.0))

    # Downstream flight trajectory waypoints spaced 30m apart ahead
    future_wps = [
        (30.0, 0.0, -25.0),
        (60.0, 2.0, -26.0),
        (90.0, 5.0, -28.0),
        (120.0, 10.0, -30.0)
    ]

    pfd_frame = svs.render_synthetic_vision_frame(pos_now, vel_body, attitude, future_wps)

    print("==========================================================================")
    print("      SYNTHETIC VISION SYSTEM (SVS) & HIGHWAY-IN-THE-SKY (HITS) AUDIT     ")
    print("==========================================================================")
    print(f"Display Resolution         : {pfd_frame['pfd_resolution'][0]}x{pfd_frame['pfd_resolution'][1]} px")
    print(f"Flight Path Vector (FPV)   : {pfd_frame['flight_path_vector_px']} px")
    print(f"Artificial Horizon Center  : Y = {pfd_frame['artificial_horizon_center_y_px']} px (Roll: {pfd_frame['horizon_roll_tilt_deg']}°)")
    print(f"Visible 3D Tunnel Gates    : {pfd_frame['visible_highway_gates_count']}\n")
    for gate in pfd_frame["highway_in_the_sky_gates"]:
        print(f"  Gate #{gate['gate_id']:02d} (Distance: {gate['distance_ahead_m']:5.1f}m) -> Polygon: {gate['screen_polygon_px']}")
    print("==========================================================================\n")