import math
import numpy as np
from typing import Dict, Tuple, Optional

class GimbalGeoreferencingEngine:
    def __init__(
        self,
        camera_focal_length_px: float = 520.0,
        camera_principal_point: Tuple[float, float] = (320.0, 240.0),
        wgs84_semi_major_a_m: float = 6378137.0,
        wgs84_semi_minor_b_m: float = 6356752.314245
    ):
        self.fx = camera_focal_length_px
        self.fy = camera_focal_length_px
        self.cx, self.cy = camera_principal_point
        self.a = wgs84_semi_major_a_m
        self.b = wgs84_semi_minor_b_m
        self.e2 = 1.0 - (self.b ** 2) / (self.a ** 2)

    def _geodetic_to_ecef(self, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        """Converts WGS-84 geodetic coordinates to Earth-Centered Earth-Fixed (ECEF)."""
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)
        
        n_rad = self.a / math.sqrt(1.0 - self.e2 * (math.sin(lat_rad) ** 2))
        x = (n_rad + alt_m) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (n_rad + alt_m) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (n_rad * (1.0 - self.e2) + alt_m) * math.sin(lat_rad)
        return np.array([x, y, z])

    def _ecef_to_geodetic(self, ecef_pt: np.ndarray) -> Tuple[float, float, float]:
        """Converts ECEF Cartesian point to WGS-84 (Lat, Lon, Alt) via Bowring's method."""
        x, y, z = ecef_pt[0], ecef_pt[1], ecef_pt[2]
        p = math.hypot(x, y)
        lon_rad = math.atan2(y, x)

        theta = math.atan2(z * self.a, p * self.b)
        lat_rad = math.atan2(
            z + (self.e2 * (self.a ** 2) / self.b) * (math.sin(theta) ** 3),
            p - self.e2 * self.a * (math.cos(theta) ** 3)
        )
        n_rad = self.a / math.sqrt(1.0 - self.e2 * (math.sin(lat_rad) ** 2))
        alt_m = p / math.cos(lat_rad) - n_rad
        return math.degrees(lat_rad), math.degrees(lon_rad), alt_m

    def _rotation_matrix_euler(self, roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)

        return np.array([
            [cp * cy, cp * sy, -sp],
            [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
            [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp]
        ])

    def calculate_target_ground_coordinates(
        self,
        uav_geodetic_pos: Tuple[float, float, float], # Lat, Lon, Alt_MSL
        uav_attitude_rad: Tuple[float, float, float], # Roll, Pitch, Yaw
        gimbal_pitch_rad: float,
        gimbal_yaw_rad: float,
        target_pixel_uv: Tuple[float, float]
    ) -> Dict:
        """
        Georeferences target pixel detection onto WGS-84 Earth ellipsoid surface.
        """
        lat_uav, lon_uav, alt_uav = uav_geodetic_pos
        roll_uav, pitch_uav, yaw_uav = uav_attitude_rad
        u_t, v_t = target_pixel_uv

        # 1. Optical Unit Ray in Camera Frame (Z-forward, X-right, Y-down)
        ray_cam = np.array([
            (u_t - self.cx) / self.fx,
            (v_t - self.cy) / self.fy,
            1.0
        ])
        ray_cam_unit = ray_cam / np.linalg.norm(ray_cam)

        # 2. Transform Camera Frame to Gimbal and Body Frame
        # Camera to Gimbal Mount: [X_cam -> Y_gimb, Y_cam -> Z_gimb, Z_cam -> X_gimb]
        r_cam_to_gimbal = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        r_gimbal = self._rotation_matrix_euler(0.0, gimbal_pitch_rad, gimbal_yaw_rad)
        r_body = self._rotation_matrix_euler(roll_uav, pitch_uav, yaw_uav)

        # 3. Transform Body to NED Frame
        r_ned = r_body @ r_gimbal @ r_cam_to_gimbal
        d_ned = r_ned @ ray_cam_unit

        # 4. Transform NED Ray to ECEF Unit Direction
        lat_r, lon_r = math.radians(lat_uav), math.radians(lon_uav)
        r_ned_to_ecef = np.array([
            [-math.sin(lat_r) * math.cos(lon_r), -math.sin(lon_r), -math.cos(lat_r) * math.cos(lon_r)],
            [-math.sin(lat_r) * math.sin(lon_r),  math.cos(lon_r), -math.cos(lat_r) * math.sin(lon_r)],
            [ math.cos(lat_r),                                 0.0, -math.sin(lat_r)]
        ])
        d_ecef = r_ned_to_ecef @ d_ned
        p_uav_ecef = self._geodetic_to_ecef(lat_uav, lon_uav, alt_uav)

        # 5. Ray-Ellipsoid Intersection Quadratic Solve: (x^2 + y^2)/a^2 + z^2/b^2 = 1
        x0, y0, z0 = p_uav_ecef
        dx, dy, dz = d_ecef

        A = (dx**2 + dy**2) / (self.a ** 2) + (dz**2) / (self.b ** 2)
        B = 2.0 * ((x0 * dx + y0 * dy) / (self.a ** 2) + (z0 * dz) / (self.b ** 2))
        C = (x0**2 + y0**2) / (self.a ** 2) + (z0**2) / (self.b ** 2) - 1.0

        discriminant = B**2 - 4.0 * A * C

        if discriminant < 0:
            return {"georeferencing_status": "RAY_POINTS_TOWARDS_SKY_HORIZON"}

        s_dist = (-B - math.sqrt(discriminant)) / (2.0 * A)
        if s_dist <= 0:
            return {"georeferencing_status": "INTERSECTION_BEHIND_CAMERA"}

        p_target_ecef = p_uav_ecef + s_dist * d_ecef
        target_lat, target_lon, target_alt = self._ecef_to_geodetic(p_target_ecef)

        slant_range_m = float(s_dist)
        ground_dist_m = math.hypot(
            (target_lat - lat_uav) * 111000.0,
            (target_lon - lon_uav) * 111000.0 * math.cos(lat_r)
        )

        return {
            "georeferencing_status": "TARGET_GEOREF_LOCKED",
            "target_latitude_wgs84": round(target_lat, 7),
            "target_longitude_wgs84": round(target_lon, 7),
            "target_elevation_msl_m": round(target_alt, 2),
            "optical_slant_range_m": round(slant_range_m, 1),
            "ground_planar_distance_m": round(ground_dist_m, 1),
            "gimbal_depression_angle_deg": round(math.degrees(gimbal_pitch_rad), 2)
        }

if __name__ == "__main__":
    georef = GimbalGeoreferencingEngine()

    print("==========================================================================")
    print("      ACTIVE GIMBAL TARGET GEOREFERENCING & RAY-ELLIPSOID AUDIT           ")
    print("==========================================================================")

    # Drone hovering over Singapore Marina at 120m MSL, camera pitched -45.0° downwards
    uav_pos = (1.3521000, 103.8198000, 120.0)
    uav_att = (math.radians(0.0), math.radians(0.0), math.radians(45.0)) # Heading NE (045°)
    g_pitch = math.radians(-45.0) # -45° depression
    g_yaw = math.radians(0.0)
    pixel_center = (320.0, 240.0) # Target centered in camera crosshairs

    target_fix = georef.calculate_target_ground_coordinates(
        uav_pos, uav_att, g_pitch, g_yaw, pixel_center
    )

    for k, v in target_fix.items():
        print(f"  • {k:32s}: {v}")
    print("==========================================================================\n")