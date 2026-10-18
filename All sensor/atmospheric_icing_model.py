import math
import numpy as np
from typing import Dict, List, Tuple

class AtmosphericIcingAndDeicingModel:
    def __init__(
        self,
        num_rotors: int = 4,
        blade_chord_m: float = 0.035, # 35mm blade chord
        prop_radius_m: float = 0.1905,
        leading_edge_area_m2_per_blade: float = 0.0045
    ):
        self.num_rotors = num_rotors
        self.chord = blade_chord_m
        self.r_prop = prop_radius_m
        self.a_leading_edge = leading_edge_area_m2_per_blade * 2.0 * num_rotors # 8 blades total
        self.rho_ice = 917.0 # kg/m^3 (glaze/rime ice density)
        self.latent_heat_fusion_j_kg = 3.34e5
        self.cp_air = 1005.0

    def compute_droplet_collection_efficiency(
        self,
        median_volume_diameter_um: float,
        airspeed_ms: float,
        air_temp_k: float
    ) -> float:
        """
        Langmuir-Blodgett droplet impingement efficiency beta_0:
          K = (rho_water * d_drop^2 * V) / (18 * mu_air * c)
          beta_0 = (0.792 * K^0.9) / (1 + 0.792 * K^0.9)
        """
        rho_water = 1000.0
        d_m = median_volume_diameter_um * 1e-6
        # Dynamic viscosity of air via Sutherland's law
        mu_air = 1.458e-6 * (air_temp_k ** 1.5) / (air_temp_k + 110.4)

        inertia_k = (rho_water * (d_m ** 2) * airspeed_ms) / (18.0 * mu_air * self.chord)
        beta_0 = (0.792 * (inertia_k ** 0.9)) / (1.0 + 0.792 * (inertia_k ** 0.9))
        return min(1.0, max(0.0, beta_0))

    def compute_ice_accretion_step(
        self,
        liquid_water_content_g_m3: float,
        median_volume_diameter_um: float,
        airspeed_ms: float,
        ambient_temp_c: float,
        exposure_duration_s: float,
        current_ice_mass_kg: float
    ) -> Dict:
        """
        Calculates mass accretion rate, profile drag amplification, and de-icing thermal demands.
        """
        if ambient_temp_c > 0.0 or liquid_water_content_g_m3 <= 0.01:
            return {
                "icing_active": False,
                "accreted_mass_kg": current_ice_mass_kg,
                "drag_increase_multiplier": 1.0,
                "deicing_power_watts": 0.0
            }

        temp_k = ambient_temp_c + 273.15
        lwc_kg_m3 = liquid_water_content_g_m3 * 1e-3

        # Catch efficiency
        beta = self.compute_droplet_collection_efficiency(median_volume_diameter_um, airspeed_ms, temp_k)

        # Water mass catch rate: m_dot = beta * LWC * V * Area_projected
        projected_area = self.a_leading_edge
        catch_rate_kg_s = beta * lwc_kg_m3 * airspeed_ms * projected_area

        # Freezing fraction (Rime ice regime assumes instantaneous freezing n=1.0)
        new_ice_mass_kg = current_ice_mass_kg + (catch_rate_kg_s * exposure_duration_s)
        ice_thickness_mm = ((new_ice_mass_kg / (self.rho_ice * self.a_leading_edge)) * 1000.0)

        # Aerodynamic penalty: Profile drag increases up to 2.4x with 2mm leading edge ice
        drag_penalty_factor = 1.0 + min(1.40, (ice_thickness_mm / 1.5) * 1.0)

        # Electro-thermal de-icing boot power requirement:
        # P_deice = h * A * (T_target - T_amb) + m_dot * L_fusion
        heat_transfer_coeff_h = 120.0 # Forced convection W/(m^2*K)
        t_target_k = 275.15 # +2°C anti-icing threshold
        
        q_convection = heat_transfer_coeff_h * self.a_leading_edge * (t_target_k - temp_k)
        q_fusion = catch_rate_kg_s * self.latent_heat_fusion_j_kg
        total_deice_power_watts = q_convection + q_fusion

        return {
            "icing_active": True,
            "collection_efficiency_beta": round(beta, 3),
            "water_catch_rate_g_s": round(catch_rate_kg_s * 1000.0, 3),
            "cumulative_ice_mass_kg": round(new_ice_mass_kg, 4),
            "estimated_ice_thickness_mm": round(ice_thickness_mm, 2),
            "drag_increase_multiplier": round(drag_penalty_factor, 3),
            "electrothermal_deice_power_watts": round(total_deice_power_watts, 1),
            "hazard_level": "CRITICAL_MASS_LIMIT" if new_ice_mass_kg > 0.40 else ("MODERATE_DRAG" if new_ice_mass_kg > 0.15 else "TRACE")
        }

if __name__ == "__main__":
    icing = AtmosphericIcingAndDeicingModel()

    # Stratus cloud icing condition: -4°C, 0.45 g/m³ LWC, 20 µm MVD, 18.05 m/s cruise
    print("==========================================================================")
    print("      ATMOSPHERIC ROTOR ICING ACCRETION & DE-ICING POWER AUDIT            ")
    print("==========================================================================")
    
    current_ice = 0.0
    for minute in range(1, 6):
        report = icing.compute_ice_accretion_step(
            liquid_water_content_g_m3=0.45,
            median_volume_diameter_um=20.0,
            airspeed_ms=18.05,
            ambient_temp_c=-4.0,
            exposure_duration_s=60.0,
            current_ice_mass_kg=current_ice
        )
        current_ice = report["cumulative_ice_mass_kg"]
        print(f"Minute {minute:02d}: Ice Mass = {report['cumulative_ice_mass_kg']*1000:5.1f} g | "
              f"Thickness = {report['estimated_ice_thickness_mm']:4.2f} mm | Drag Mult = {report['drag_increase_multiplier']:.3f}x | "
              f"De-Ice Power = {report['electrothermal_deice_power_watts']:5.1f} W | Status = {report['hazard_level']}")
    print("==========================================================================\n")