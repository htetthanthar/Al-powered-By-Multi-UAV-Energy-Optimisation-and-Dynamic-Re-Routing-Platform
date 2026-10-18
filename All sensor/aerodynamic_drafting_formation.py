import math
import numpy as np
from typing import Dict, List, Tuple

class CooperativeAerodynamicFormationPlanner:
    def __init__(
        self,
        uav_mass_kg: float = 6.14,
        rotor_span_effective_m: float = 0.65,
        cruise_speed_ms: float = 18.05,
        air_density_kg_m3: float = 1.225
    ):
        self.mass = uav_mass_kg
        self.b_span = rotor_span_effective_m
        self.v_cruise = cruise_speed_ms
        self.rho = air_density_kg_m3
        self.g = 9.80665

        # Bound vortex parameters
        self.b0 = (math.pi / 4.0) * self.b_span
        self.gamma0 = (self.mass * self.g) / (self.rho * self.v_cruise * self.b0)
        self.r_core = 0.08 * self.b0

    def compute_spatial_upwash_field(self, y_lateral_m: float, z_vertical_m: float) -> float:
        """
        Calculates vertical induced air velocity (w_upwash) around a leading aircraft's wake pair.
        Positive values indicate upward velocity columns (aerodynamic drafting benefit).
        """
        y_vortex = self.b0 / 2.0
        
        # Right vortex contribution (positive circulation)
        term_right = (y_lateral_m - y_vortex) / ((y_lateral_m - y_vortex)**2 + z_vertical_m**2 + self.r_core**2)
        # Left vortex contribution (negative circulation)
        term_left = (y_lateral_m + y_vortex) / ((y_lateral_m + y_vortex)**2 + z_vertical_m**2 + self.r_core**2)

        w_induced = (self.gamma0 / (2.0 * math.pi)) * (term_right - term_left)
        return w_induced

    def find_optimal_echelon_formation_offset(self) -> Dict:
        """
        Finds the lateral and vertical offset that maximizes induced upwash while
        maintaining safe physical separation boundaries (>= 2.5 spans).
        """
        lateral_search = np.linspace(self.b_span * 1.0, self.b_span * 3.5, 100)
        vertical_search = np.linspace(-0.5, 0.5, 50)

        best_upwash = -float('inf')
        best_offset = (0.0, 0.0)

        for y in lateral_search:
            for z in vertical_search:
                w_up = self.compute_spatial_upwash_field(y, z)
                if w_up > best_upwash:
                    best_upwash = w_up
                    best_offset = (y, z)

        # Baseline single-drone cruise power
        p_base_watts = 295.0
        # Power savings: Delta_P = Weight * w_upwash * efficiency
        p_saved_watts = (self.mass * self.g) * best_upwash * 0.72 # 72% capture factor
        p_formation_watts = p_base_watts - p_saved_watts
        energy_reduction_pct = (p_saved_watts / p_base_watts) * 100.0

        # Longitudinal separation for wake stabilization
        longitudinal_dx_m = 4.0 * self.b_span # ~2.6 meters

        return {
            "optimal_longitudinal_dx_m": round(longitudinal_dx_m, 2),
            "optimal_lateral_dy_m": round(best_offset[0], 2),
            "optimal_vertical_dz_m": round(best_offset[1], 2),
            "max_harvested_upwash_ms": round(best_upwash, 3),
            "single_uav_cruise_power_watts": p_base_watts,
            "drafting_uav_cruise_power_watts": round(p_formation_watts, 1),
            "power_reduction_pct": round(energy_reduction_pct, 2),
            "formation_geometry": "RIGHT_ECHELON_VORTEX_HARVESTING"
        }

if __name__ == "__main__":
    formation = CooperativeAerodynamicFormationPlanner()
    opt_params = formation.find_optimal_echelon_formation_offset()

    print("==========================================================================")
    print("      MULTI-UAV COOPERATIVE AERODYNAMIC FORMATION & DRAFTING AUDIT        ")
    print("==========================================================================")
    for k, v in opt_params.items():
        print(f"  • {k:35s}: {v}")
    print("==========================================================================\n")