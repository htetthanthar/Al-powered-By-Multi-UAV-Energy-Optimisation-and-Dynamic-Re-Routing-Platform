import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class EmergencyLandingSiteSelector:
    def __init__(
        self,
        search_radius_m: float = 150.0,
        grid_resolution_m: float = 2.0,
        max_allowable_slope_deg: float = 8.0,
        population_risk_weight: float = 0.55,
        slope_risk_weight: float = 0.25,
        distance_risk_weight: float = 0.20
    ):
        self.r_search = search_radius_m
        self.res = grid_resolution_m
        self.max_slope = max_allowable_slope_deg
        self.w_pop = population_risk_weight
        self.w_slope = slope_risk_weight
        self.w_dist = distance_risk_weight

    def generate_synthetic_emergency_map(
        self,
        grid_size: int = 150
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates local synthetic spatial layers centered at the UAV:
          1. DEM (Elevation map in meters)
          2. Population Density Raster (0.0 = empty field, 1.0 = crowded plaza)
          3. Obstacle / Building Mask (1 = building/tree canopy, 0 = open terrain)
        """
        np.random.seed(42)
        
        # DEM: Baseline terrain with slight grade
        x = np.linspace(-self.r_search, self.r_search, grid_size)
        y = np.linspace(-self.r_search, self.r_search, grid_size)
        xx, yy = np.meshgrid(x, y)
        elevation = 15.0 + 0.02 * xx + 0.015 * yy + np.random.normal(0, 0.1, (grid_size, grid_size))

        # Population Density Map (Pockets of human presence)
        pop_density = np.zeros((grid_size, grid_size))
        # Urban plaza centered at (+40m, +30m)
        r_plaza = np.sqrt((xx - 40.0)**2 + (yy - 30.0)**2)
        pop_density[r_plaza < 35.0] = 0.85
        # Road corridor along x = -50m
        pop_density[np.abs(xx + 50.0) < 6.0] = 0.40

        # Obstacle Mask: Buildings and Water Bodies
        obstacle_mask = np.zeros((grid_size, grid_size), dtype=int)
        # Building cluster at (+40m, -50m)
        obstacle_mask[(xx > 20.0) & (xx < 60.0) & (yy > -70.0) & (yy < -30.0)] = 1
        # Water canal at yy < -120m
        obstacle_mask[yy < -120.0] = 1

        return elevation, pop_density, obstacle_mask

    def calculate_terrain_slope_degrees(self, elevation_grid: np.ndarray) -> np.ndarray:
        """
        Computes spatial gradient magnitude using central finite differences:
        Slope = arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
        """
        dz_dy, dz_dx = np.gradient(elevation_grid, self.res)
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        return np.degrees(slope_rad)

    def select_best_landing_site(
        self,
        current_uav_pos_ned: Tuple[float, float, float]
    ) -> Dict:
        """
        Evaluates candidate landing cells within glide footprint and minimizes composite risk:
        Risk = w_pop * Cost_pop + w_slope * Cost_slope + w_dist * Cost_dist
        """
        grid_dim = int((2.0 * self.r_search) / self.res)
        elevation, pop_density, obstacle_mask = self.generate_synthetic_emergency_map(grid_size=grid_dim)
        slope_grid = self.calculate_terrain_slope_degrees(elevation)

        # Coordinate axes
        x_coords = np.linspace(-self.r_search, self.r_search, grid_dim)
        y_coords = np.linspace(-self.r_search, self.r_search, grid_dim)
        xx, yy = np.meshgrid(x_coords, y_coords)
        dist_grid = np.sqrt(xx**2 + yy**2)

        # Hard Feasibility Mask
        feasible = (
            (dist_grid <= self.r_search) &
            (obstacle_mask == 0) &
            (slope_grid <= self.max_slope) &
            (pop_density < 0.20)
        )

        if not np.any(feasible):
            return {
                "landing_site_found": False,
                "action": "DEPLOY_BALLISTIC_PARACHUTE_TERMINAL_DESCENT",
                "target_ned": current_uav_pos_ned
            }

        # Normalize Cost Metrics to [0, 1]
        norm_pop = pop_density / (np.max(pop_density) + 1e-6)
        norm_slope = slope_grid / self.max_slope
        norm_dist = dist_grid / self.r_search

        composite_risk = (
            self.w_pop * norm_pop +
            self.w_slope * norm_slope +
            self.w_dist * norm_dist
        )
        # Apply penalty to non-feasible cells
        composite_risk[~feasible] = 1e6

        # Identify cell with minimum risk
        min_idx = np.unravel_index(np.argmin(composite_risk), composite_risk.shape)
        best_x_rel = float(xx[min_idx])
        best_y_rel = float(yy[min_idx])
        best_z_ground = float(elevation[min_idx])

        target_north = current_uav_pos_ned[0] + best_x_rel
        target_east = current_uav_pos_ned[1] + best_y_rel

        return {
            "landing_site_found": True,
            "target_relative_x_m": round(best_x_rel, 2),
            "target_relative_y_m": round(best_y_rel, 2),
            "target_ground_elevation_msl_m": round(best_z_ground, 2),
            "distance_to_site_m": round(float(dist_grid[min_idx]), 2),
            "site_terrain_slope_deg": round(float(slope_grid[min_idx]), 2),
            "site_population_density": round(float(pop_density[min_idx]), 3),
            "site_composite_risk_score": round(float(composite_risk[min_idx]), 4),
            "action": "EXECUTE_EMERGENCY_AUTONOMOUS_GLIDE_APPROACH"
        }

if __name__ == "__main__":
    selector = EmergencyLandingSiteSelector(search_radius_m=150.0)
    decision = selector.select_best_landing_site(current_uav_pos_ned=(0.0, 0.0, -25.0))

    print("==========================================================================")
    print("      AUTONOMOUS EMERGENCY FORCED LANDING SITE SELECTION RESULTS          ")
    print("==========================================================================")
    for k, v in decision.items():
        print(f"  • {k:32s}: {v}")
    print("==========================================================================\n")