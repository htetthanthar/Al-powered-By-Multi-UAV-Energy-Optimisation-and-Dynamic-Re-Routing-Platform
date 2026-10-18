import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class Dynamic4DGeocagingServer:
    def __init__(self, horizontal_buffer_m: float = 100.0, vertical_buffer_m: float = 30.0):
        self.h_buf = horizontal_buffer_m
        self.v_buf = vertical_buffer_m
        self.active_tfrs: List[Dict] = []

    def inject_temporary_flight_restriction(
        self,
        tfr_id: str,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        floor_alt_msl_m: float,
        ceiling_alt_msl_m: float,
        start_epoch_s: float,
        duration_s: float,
        expansion_rate_ms: float = 0.0
    ):
        """Registers a dynamic, time-varying 4D cylindrical geocage zone."""
        self.active_tfrs.append({
            "tfr_id": tfr_id,
            "lat": center_lat,
            "lon": center_lon,
            "base_radius_m": radius_m,
            "expansion_rate_ms": expansion_rate_ms,
            "floor_m": floor_alt_msl_m - self.v_buf,
            "ceiling_m": ceiling_alt_msl_m + self.v_buf,
            "start_time": start_epoch_s,
            "end_time": start_epoch_s + duration_s
        })
        print(f"[+] Dynamic 4D TFR Ingested: '{tfr_id}' (Radius: {radius_m}m, Floor: {floor_alt_msl_m}m, Ceiling: {ceiling_alt_msl_m}m)")

    def evaluate_trajectory_breach(
        self,
        planned_4d_trajectory: List[Dict],
        current_epoch_s: float
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Evaluates planned trajectory frames [lat, lon, alt_msl, timestamp_s] against active 4D TFRs.
        """
        for pt in planned_4d_trajectory:
            t_pt = pt["timestamp_s"]
            lat_pt = pt["lat"]
            lon_pt = pt["lon"]
            alt_pt = pt["alt_msl_m"]

            for tfr in self.active_tfrs:
                if tfr["start_time"] <= t_pt <= tfr["end_time"]:
                    # Compute dynamic expanded radius at time t_pt
                    dt = t_pt - tfr["start_time"]
                    current_radius = tfr["base_radius_m"] + (tfr["expansion_rate_ms"] * dt) + self.h_buf

                    # Planar distance calculation
                    dx = (lon_pt - tfr["lon"]) * 111000.0 * math.cos(math.radians(tfr["lat"]))
                    dy = (lat_pt - tfr["lat"]) * 111000.0
                    horizontal_dist = math.hypot(dx, dy)

                    if horizontal_dist <= current_radius:
                        # Altitude containment check
                        if tfr["floor_m"] <= alt_pt <= tfr["ceiling_m"]:
                            breach_info = {
                                "tfr_id": tfr["tfr_id"],
                                "breach_time_s": t_pt,
                                "breach_lat": lat_pt,
                                "breach_lon": lon_pt,
                                "horizontal_penetration_m": round(current_radius - horizontal_dist, 1)
                            }
                            return True, tfr["tfr_id"], breach_info

        return False, None, None

    def compute_tactical_bypass_waypoint(
        self,
        start_wpt: Dict,
        end_wpt: Dict,
        breach_tfr_id: str
    ) -> Dict:
        """
        Computes a lateral detour navigation waypoint offset perpendicular to the TFR boundary.
        """
        tfr = next(t for t in self.active_tfrs if t["tfr_id"] == breach_tfr_id)
        
        # Midpoint tangent displacement
        tfr_lat, tfr_lon = tfr["lat"], tfr["lon"]
        detour_radius_deg = (tfr["base_radius_m"] + self.h_buf * 1.5) / 111000.0

        # Vector from start to end
        dx = end_wpt["lon"] - start_wpt["lon"]
        dy = end_wpt["lat"] - start_wpt["lat"]
        
        # Perpendicular normal vector
        norm_x, norm_y = -dy, dx
        mag = math.hypot(norm_x, norm_y) + 1e-6
        norm_x /= mag
        norm_y /= mag

        detour_lat = tfr_lat + norm_y * detour_radius_deg
        detour_lon = tfr_lon + norm_x * detour_radius_deg

        return {
            "name": f"TACTICAL_DETOUR_{tfr['tfr_id']}",
            "lat": round(detour_lat, 6),
            "lon": round(detour_lon, 6),
            "alt_msl_m": start_wpt.get("alt_msl_m", 25.0)
        }

if __name__ == "__main__":
    caging_server = Dynamic4DGeocagingServer(horizontal_buffer_m=50.0, vertical_buffer_m=15.0)

    now = time.time()
    # Air ambulance dynamic evacuation TFR activated for 30 minutes
    caging_server.inject_temporary_flight_restriction(
        tfr_id="MEDEVAC-HELICOPTER-EMERGENCY",
        center_lat=1.3500,
        center_lon=103.8200,
        radius_m=350.0,
        floor_alt_msl_m=0.0,
        ceiling_alt_msl_m=120.0,
        start_epoch_s=now,
        duration_s=1800.0
    )

    # UAV Scheduled Trajectory passing directly through zone at T+300s
    planned_flight = [
        {"lat": 1.3300, "lon": 103.8200, "alt_msl_m": 25.0, "timestamp_s": now + 100.0},
        {"lat": 1.3500, "lon": 103.8200, "alt_msl_m": 25.0, "timestamp_s": now + 300.0},
        {"lat": 1.3700, "lon": 103.8200, "alt_msl_m": 25.0, "timestamp_s": now + 500.0}
    ]

    has_breach, tfr_name, breach_data = caging_server.evaluate_trajectory_breach(planned_flight, now)

    print("==========================================================================")
    print("      DYNAMIC 4D GEOCAGING & AIRSPACE CONFLICT RESOLUTION AUDIT           ")
    print("==========================================================================")
    print(f"Trajectory Conflict Detected : {has_breach} ({tfr_name})")
    
    if has_breach:
        print(f"Penetration Depth            : {breach_data['horizontal_penetration_m']} meters at T+{breach_data['breach_time_s'] - now:.0f}s")
        detour_wp = caging_server.compute_tactical_bypass_waypoint(planned_flight[0], planned_flight[2], tfr_name)
        print("\nGenerated Lateral Tactical Detour Waypoint:")
        for k, v in detour_wp.items():
            print(f"  • {k:16s}: {v}")
    print("==========================================================================\n")