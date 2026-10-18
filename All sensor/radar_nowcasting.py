import math
import numpy as np
import cv2
from typing import Tuple, Dict, List

class RadarPrecipitationNowcaster:
    def __init__(
        self,
        grid_dim: Tuple[int, int] = (64, 64),
        spatial_resolution_km: float = 0.5,
        marshall_palmer_a: float = 200.0,
        marshall_palmer_b: float = 1.6
    ):
        self.grid_rows, self.grid_cols = grid_dim
        self.res_km = spatial_resolution_km
        self.a = marshall_palmer_a
        self.b = marshall_palmer_b
        self.rain_threshold_dbz = 20.0  # ~0.65 mm/h (light rain threshold)

    def dbz_to_rain_rate(self, dbz_grid: np.ndarray) -> np.ndarray:
        """
        Converts radar reflectivity factor Z (dBZ) to rain rate R (mm/h):
        Z_linear = 10^(dBZ / 10) = a * R^b  =>  R = (10^(dBZ / 10) / a)^(1 / b)
        """
        z_linear = np.power(10.0, dbz_grid / 10.0)
        rain_rate = np.power(z_linear / self.a, 1.0 / self.b)
        rain_rate[dbz_grid < self.rain_threshold_dbz] = 0.0
        return rain_rate

    def estimate_advection_field(self, radar_t0: np.ndarray, radar_t1: np.ndarray) -> np.ndarray:
        """
        Calculates spatial velocity vectors (u_flow, v_flow) using dense optical flow.
        """
        # Normalize to 8-bit unsigned integer for OpenCV
        norm_t0 = np.clip((radar_t0 / 70.0) * 255.0, 0, 255).astype(np.uint8)
        norm_t1 = np.clip((radar_t1 / 70.0) * 255.0, 0, 255).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            norm_t0, norm_t1,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        return flow  # Shape: (rows, cols, 2) where flow[..., 0] = dx, flow[..., 1] = dy

    def extrapolate_nowcast(self, radar_current: np.ndarray, flow_field: np.ndarray, lead_steps: int = 6) -> List[np.ndarray]:
        """
        Advects the current radar reflectivity field forward in time using backward semi-Lagrangian mapping.
        """
        h, w = radar_current.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        forecasts = []
        current_frame = radar_current.copy()

        for step in range(1, lead_steps + 1):
            map_x = (grid_x - flow_field[..., 0] * step).astype(np.float32)
            map_y = (grid_y - flow_field[..., 1] * step).astype(np.float32)

            # Interpolate advected field
            extrapolated = cv2.remap(
                current_frame.astype(np.float32),
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0
            )
            forecasts.append(extrapolated)

        return forecasts

    def evaluate_corridor_hazard(
        self,
        flight_waypoints: List[Tuple[int, int]],
        forecast_grids: List[np.ndarray],
        dt_step_min: float = 5.0
    ) -> Dict:
        """
        Evaluates future precipitation hazards along scheduled waypoint coordinates over time.
        """
        hazard_profile = []
        is_safe = True

        for idx, (gx, gy) in enumerate(flight_waypoints):
            forecast_idx = min(idx, len(forecast_grids) - 1)
            dbz_val = forecast_grids[forecast_idx][gy, gx]
            rain_rate = self.dbz_to_rain_rate(np.array([[dbz_val]]))[0, 0]

            threat_level = "NOMINAL"
            if dbz_val >= 35.0:  # Heavy rain / convective gust hazard
                threat_level = "SEVERE_CONVECTIVE_HAZARD"
                is_safe = False
            elif dbz_val >= 20.0:
                threat_level = "MODERATE_RAIN"

            hazard_profile.append({
                "waypoint_idx": idx,
                "projected_time_min": idx * dt_step_min,
                "reflectivity_dbz": round(float(dbz_val), 1),
                "rain_rate_mm_per_h": round(float(rain_rate), 2),
                "threat_status": threat_level
            })

        return {
            "corridor_clear": is_safe,
            "max_reflectivity_dbz": max(h["reflectivity_dbz"] for h in hazard_profile),
            "hazard_trajectory": hazard_profile
        }

if __name__ == "__main__":
    nowcaster = RadarPrecipitationNowcaster(grid_dim=(64, 64))

    # Synthetic convective storm cell drifting northeast
    radar_t0 = np.zeros((64, 64), dtype=np.float32)
    radar_t1 = np.zeros((64, 64), dtype=np.float32)

    # Initial storm core at (x=20, y=20)
    cv2.circle(radar_t0, (20, 20), 10, 42.0, -1)
    # Advected storm core at (x=24, y=23) after 5 minutes
    cv2.circle(radar_t1, (24, 23), 10, 43.5, -1)

    flow = nowcaster.estimate_advection_field(radar_t0, radar_t1)
    future_radar = nowcaster.extrapolate_nowcast(radar_t1, flow, lead_steps=6)

    # Flight path intersecting advection corridor
    flight_path = [(15, 15), (22, 22), (30, 28), (40, 35)]
    assessment = nowcaster.evaluate_corridor_hazard(flight_path, future_radar)

    print("==========================================================================")
    print("      DOPPLER RADAR ADVECTION NOWCASTING & CORRIDOR HAZARD AUDIT          ")
    print("==========================================================================")
    print(f"Corridor Clearance Status : {'CLEARED' if assessment['corridor_clear'] else 'HAZARD INTERCEPT TRIGGERED'}")
    print(f"Peak Reflectivity Along Path: {assessment['max_reflectivity_dbz']} dBZ\n")
    for step in assessment["hazard_trajectory"]:
        print(f"  T+{step['projected_time_min']:02.0f} min (WP-{step['waypoint_idx']}): "
              f"{step['reflectivity_dbz']:4.1f} dBZ | Rain: {step['rain_rate_mm_per_h']:4.2f} mm/h | Status: {step['threat_status']}")
    print("==========================================================================\n")