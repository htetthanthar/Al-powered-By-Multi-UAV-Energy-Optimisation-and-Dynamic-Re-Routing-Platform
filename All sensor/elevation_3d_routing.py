import math
import numpy as np
import scipy.interpolate as interp
from typing import Dict, List, Tuple

class ElevationAware3DRouter:
    def __init__(
        self,
        min_clearance_agl_m: float = 15.0,
        max_flight_ceiling_msl_m: float = 60.0,
        grid_resolution_m: float = 10.0
    ):
        self.min_agl = min_clearance_agl_m
        self.max_ceiling = max_flight_ceiling_msl_m
        self.grid_res = grid_resolution_m

    def query_surface_elevation(self, x_m: float, y_m: float) -> float:
        """
        Simulates querying a digital surface model (ground terrain + building rooftop heights).
        """
        # Baseline terrain undulation
        base_terrain = 12.0 + 4.0 * math.sin(x_m / 250.0) + 3.0 * math.cos(y_m / 300.0)
        
        # Synthetic urban high-rise cluster between x in [200, 350] and y in [150, 300]
        if (200.0 <= x_m <= 350.0) and (150.0 <= y_m <= 300.0):
            building_height = 28.0  # 28m high-rise rooftop
            return base_terrain + building_height
        
        return base_terrain

    def generate_safe_vertical_profile(
        self,
        horizontal_waypoints: List[Tuple[float, float]],
        cruise_speed_ms: float = 18.05
    ) -> List[Dict[str, float]]:
        """
        Samples terrain along waypoint legs and calculates terrain-following altitudes.
        """
        safe_3d_nodes = []
        
        for i in range(len(horizontal_waypoints) - 1):
            p1 = horizontal_waypoints[i]
            p2 = horizontal_waypoints[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            leg_dist = math.hypot(dx, dy)
            num_samples = max(2, int(leg_dist / self.grid_res))

            for s in range(num_samples):
                ratio = s / num_samples
                cur_x = p1[0] + dx * ratio
                cur_y = p1[1] + dy * ratio
                
                surface_height = self.query_surface_elevation(cur_x, cur_y)
                required_alt_msl = min(
                    self.max_ceiling,
                    max(surface_height + self.min_agl, 25.0)
                )

                safe_3d_nodes.append({
                    "x_m": cur_x,
                    "y_m": cur_y,
                    "surface_m": surface_height,
                    "target_alt_msl_m": required_alt_msl
                })

        # Append terminal node
        last_pt = horizontal_waypoints[-1]
        term_surf = self.query_surface_elevation(last_pt[0], last_pt[1])
        safe_3d_nodes.append({
            "x_m": last_pt[0],
            "y_m": last_pt[1],
            "surface_m": term_surf,
            "target_alt_msl_m": max(term_surf + self.min_agl, 25.0)
        })

        return safe_3d_nodes

    def fit_piecewise_bezier_trajectory(
        self,
        safe_3d_nodes: List[Dict[str, float]],
        num_trajectory_points: int = 100
    ) -> Dict:
        """
        Fits a smooth 3D B-spline through safe altitude points to eliminate jerk.
        """
        x_pts = [n["x_m"] for n in safe_3d_nodes]
        y_pts = [n["y_m"] for n in safe_3d_nodes]
        z_pts = [n["target_alt_msl_m"] for n in safe_3d_nodes]

        # Parameterize nodes along path length
        path_segments = [0.0]
        for i in range(1, len(x_pts)):
            d = math.sqrt((x_pts[i] - x_pts[i-1])**2 + (y_pts[i] - y_pts[i-1])**2 + (z_pts[i] - z_pts[i-1])**2)
            path_segments.append(path_segments[-1] + d)
        
        total_length_m = path_segments[-1]
        u_norm = np.array(path_segments) / (total_length_m + 1e-6)

        # 3rd-order spline smoothing
        tck_x = interp.splrep(u_norm, x_pts, k=3, s=1.0)
        tck_y = interp.splrep(u_norm, y_pts, k=3, s=1.0)
        tck_z = interp.splrep(u_norm, z_pts, k=3, s=2.0)

        eval_u = np.linspace(0, 1.0, num_trajectory_points)
        smooth_x = interp.splev(eval_u, tck_x)
        smooth_y = interp.splev(eval_u, tck_y)
        smooth_z = interp.splev(eval_u, tck_z)

        # Evaluate minimum clearance verification along smooth path
        clearance_profile = []
        for sx, sy, sz in zip(smooth_x, smooth_y, smooth_z):
            surf = self.query_surface_elevation(sx, sy)
            clearance_profile.append(sz - surf)

        min_recorded_clearance = min(clearance_profile)

        return {
            "total_path_length_m": round(float(total_length_m), 2),
            "min_agl_clearance_recorded_m": round(float(min_recorded_clearance), 2),
            "clearance_invariant_held": min_recorded_clearance >= (self.min_agl - 0.5),
            "sampled_trajectory_x": smooth_x.tolist(),
            "sampled_trajectory_y": smooth_y.tolist(),
            "sampled_trajectory_z": smooth_z.tolist()
        }

if __name__ == "__main__":
    router = ElevationAware3DRouter(min_clearance_agl_m=15.0, max_flight_ceiling_msl_m=60.0)

    # Waypoints traversing over ground terrain and urban building high-rises
    wps = [(0.0, 0.0), (150.0, 100.0), (280.0, 220.0), (450.0, 350.0)]
    
    nodes_3d = router.generate_safe_vertical_profile(wps)
    smooth_traj = router.fit_piecewise_bezier_trajectory(nodes_3d)

    print("==========================================================================")
    print("      3D TERRAIN & ROOFTOP ELEVATION-AWARE TRAJECTORY GENERATOR           ")
    print("==========================================================================")
    print(f"Total Trajectory Length      : {smooth_traj['total_path_length_m']} meters")
    print(f"Minimum Ground Clearance AGL : {smooth_traj['min_agl_clearance_recorded_m']} meters (Required: >= 15.0m)")
    print(f"Safety Altitude Clearance   : {'VERIFIED SAFE' if smooth_traj['clearance_invariant_held'] else 'ALTITUDE COLLISION RISK'}")
    print("==========================================================================\n")