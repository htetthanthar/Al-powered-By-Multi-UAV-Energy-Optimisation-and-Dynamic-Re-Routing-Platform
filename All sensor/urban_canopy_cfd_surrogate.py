import math
import numpy as np
from typing import Dict, Tuple, List

class UrbanCanopyWindSurrogate:
    def __init__(
        self,
        roughness_length_z0_m: float = 1.2,       # High-density urban canopy
        displacement_height_d_m: float = 18.0,    # Mean building height layer
        von_karman_constant: float = 0.40
    ):
        self.z0 = roughness_length_z0_m
        self.d = displacement_height_d_m
        self.kappa = von_karman_constant

    def compute_log_wind_profile(self, altitude_agl_m: float, reference_wind_10m_ms: float) -> float:
        """
        Standard logarithmic wind shear profile in urban atmospheric boundary layers:
        U(z) = U_ref * (ln((z - d) / z0) / ln((z_ref - d) / z0))
        """
        if altitude_agl_m <= self.d:
            # Canopy skim layer (attenuated wind speed with high turbulence intensity)
            return reference_wind_10m_ms * 0.45

        z_effective = max(altitude_agl_m - self.d, self.z0 * 1.05)
        z_ref_effective = max(10.0 + self.d - self.d, self.z0 * 1.05)
        
        u_z = reference_wind_10m_ms * (math.log(z_effective / self.z0) / math.log(z_ref_effective / self.z0))
        return max(0.5, u_z)

    def calculate_canyon_venturi_effect(
        self,
        ambient_wind_u: float,
        ambient_wind_v: float,
        street_heading_deg: float,
        street_width_w_m: float,
        building_height_h_m: float
    ) -> Dict:
        """
        Models along-canyon wind speed acceleration and cross-canyon vortex shedding.
        Aspect ratio: AR = H / W
        """
        ambient_speed = math.hypot(ambient_wind_u, ambient_wind_v)
        ambient_dir_rad = math.atan2(ambient_wind_v, ambient_wind_u)
        street_dir_rad = math.radians(street_heading_deg)

        # Angle between wind vector and street axis
        alpha = abs(ambient_dir_rad - street_dir_rad) % math.pi
        if alpha > math.pi / 2.0:
            alpha = math.pi - alpha

        aspect_ratio = building_height_h_m / max(street_width_w_m, 5.0)

        # Venturi Channeling Factor along canyon axis
        # Maximum acceleration occurs when wind aligns with street (alpha -> 0)
        channeling_factor = 1.0 + (0.45 * math.cos(alpha) * math.tanh(aspect_ratio))
        
        # Turbulence Intensity (TI) amplification factor
        base_ti = 0.18
        ti_canyon = base_ti * (1.0 + 0.8 * math.sin(alpha) * min(aspect_ratio, 2.5))

        effective_canyon_wind = ambient_speed * channeling_factor

        # Aerodynamic power penalty due to gustiness / turbulence:
        # P_turb_penalty = 1 + 3 * TI^2
        turbulence_power_multiplier = 1.0 + 3.0 * (ti_canyon ** 2)

        return {
            "aspect_ratio_h_w": round(aspect_ratio, 2),
            "wind_street_angle_deg": round(math.degrees(alpha), 1),
            "channeling_speed_multiplier": round(channeling_factor, 3),
            "effective_canyon_wind_ms": round(effective_canyon_wind, 2),
            "turbulence_intensity_ti": round(ti_canyon, 3),
            "power_draw_penalty_factor": round(turbulence_power_multiplier, 3)
        }

if __name__ == "__main__":
    canopy_model = UrbanCanopyWindSurrogate()
    
    # Evaluate wind profile at cruising altitudes (15m, 25m, 45m, 60m)
    print("==========================================================================")
    print("      URBAN CANOPY AERODYNAMIC PROFILE & VENTURI CANYON EFFECT            ")
    print("==========================================================================")
    print("Altitude Wind Shear Profile (Base Wind = 4.0 m/s @ 10m):")
    for alt in [15.0, 25.0, 35.0, 50.0, 65.0]:
        u_alt = canopy_model.compute_log_wind_profile(alt, reference_wind_10m_ms=4.0)
        print(f"  • Altitude {alt:4.1f}m AGL : Effective Wind Speed = {u_alt:.2f} m/s ({u_alt*3.6:.1f} km/h)")

    # Canyon Venturi Assessment: 20m wide street flanked by 40m buildings (AR = 2.0)
    canyon_analysis = canopy_model.calculate_canyon_venturi_effect(
        ambient_wind_u=3.5, ambient_wind_v=1.2, street_heading_deg=20.0, street_width_w_m=20.0, building_height_h_m=40.0
    )
    print("\nUrban Canyon Aerodynamic Channeling Diagnostics:")
    for k, v in canyon_analysis.items():
        print(f"  • {k:30s}: {v}")
    print("==========================================================================\n")