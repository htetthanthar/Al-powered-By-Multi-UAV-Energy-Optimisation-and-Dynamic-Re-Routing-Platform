import math
import numpy as np
from typing import Dict, List, Tuple

class UrbanThermalEnergyHarvestingModel:
    def __init__(
        self,
        surface_solar_irradiance_w_m2: float = 850.0, # Tropical solar irradiance
        convective_boundary_layer_height_m: float = 800.0,
        ambient_temp_k: float = 305.15 # 32 degrees C
    ):
        self.i_solar = surface_solar_irradiance_w_m2
        self.zi = convective_boundary_layer_height_m
        self.t_amb = ambient_temp_k
        self.g = 9.80665
        self.rho = 1.225
        self.cp = 1005.0 # Specific heat capacity of air (J/kg*K)

    def compute_surface_sensible_heat_flux(self, albedo: float, surface_type: str) -> float:
        """
        Estimates surface sensible heat flux H (W/m^2) based on surface albedo and thermal absorption.
        """
        # Net absorbed radiation: R_net = (1 - albedo) * I_solar
        r_net = (1.0 - albedo) * self.i_solar
        
        # Sensible heat flux fraction
        heat_flux_ratios = {
            "ASPHALT_PARKING": 0.65,
            "INDUSTRIAL_METAL_ROOF": 0.75,
            "CONCRETE_PLAZA": 0.50,
            "VEGETATION_PARK": 0.20,
            "WATER_CANAL": 0.05
        }
        fraction = heat_flux_ratios.get(surface_type, 0.40)
        return r_net * fraction

    def compute_convective_updraft_velocity(
        self,
        heat_flux_w_m2: float,
        altitude_agl_m: float,
        thermal_radius_m: float = 35.0
    ) -> Dict:
        """
        Allen & Deardorff convective boundary layer scaling model:
        Convective velocity scale: w* = ((g * H * zi) / (rho * cp * T_amb))^(1/3)
        Vertical velocity at altitude z: w(z) = w* * 1.34 * (z/zi)^(1/3) * (1 - 1.1 * (z/zi))
        """
        # Convective scale w*
        buoyancy_flux = (self.g * heat_flux_w_m2) / (self.rho * self.cp * self.t_amb)
        w_star = (buoyancy_flux * self.zi) ** (1.0 / 3.0)

        z_norm = max(altitude_agl_m / self.zi, 1e-4)
        if z_norm < 1.0:
            w_core = w_star * 1.34 * (z_norm ** (1.0 / 3.0)) * (1.0 - 1.1 * z_norm)
        else:
            w_core = 0.0

        w_updraft_ms = max(0.0, w_core)

        # Multi-rotor power regeneration / conservation factor:
        # Hover/Cruise power is reduced when climbing in a positive vertical air column:
        # P_saved = Weight * w_updraft
        uav_mass_kg = 6.14
        uav_weight_n = uav_mass_kg * self.g
        power_saved_watts = uav_weight_n * w_updraft_ms * 0.70 # 70% aerodynamic capture efficiency

        return {
            "sensible_heat_flux_w_m2": round(heat_flux_w_m2, 1),
            "convective_velocity_scale_ms": round(w_star, 2),
            "updraft_vertical_velocity_ms": round(w_updraft_ms, 2),
            "instantaneous_power_saved_watts": round(power_saved_watts, 1),
            "soaring_benefit": "ACTIVE_ENERGY_HARVEST" if w_updraft_ms > 0.5 else "NEGLIGIBLE_THERMAL"
        }

if __name__ == "__main__":
    thermal_model = UrbanThermalEnergyHarvestingModel()

    print("==========================================================================")
    print("      URBAN THERMAL UPDRAFT & ENERGY HARVESTING REGENERATION MODEL       ")
    print("==========================================================================")
    
    surfaces = [
        ("ASPHALT_PARKING", 0.12),
        ("INDUSTRIAL_METAL_ROOF", 0.18),
        ("CONCRETE_PLAZA", 0.35),
        ("VEGETATION_PARK", 0.25)
    ]

    flight_altitude = 35.0 # meters AGL

    for s_name, albedo in surfaces:
        h_flux = thermal_model.compute_surface_sensible_heat_flux(albedo, s_name)
        updraft = thermal_model.compute_convective_updraft_velocity(h_flux, flight_altitude)
        print(f"Surface: {s_name:22s} | Flux: {updraft['sensible_heat_flux_w_m2']:5.1f} W/m² | "
              f"Updraft: {updraft['updraft_vertical_velocity_ms']:4.2f} m/s | Power Saved: {updraft['instantaneous_power_saved_watts']:5.1f} W ({updraft['soaring_benefit']})")
    print("==========================================================================\n")