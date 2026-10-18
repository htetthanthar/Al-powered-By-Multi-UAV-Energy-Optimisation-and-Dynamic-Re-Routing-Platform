import math
import numpy as np
from typing import Dict, Tuple

class GroundEffectAndVRSSafeguard:
    def __init__(
        self,
        propeller_radius_m: float = 0.1905, # 15-inch props
        num_rotors: int = 4,
        uav_mass_kg: float = 6.14,
        air_density_kg_m3: float = 1.225
    ):
        self.R = propeller_radius_m
        self.num_rotors = num_rotors
        self.m = uav_mass_kg
        self.rho = air_density_kg_m3
        self.g = 9.80665

        # Total multi-rotor disk area
        self.total_disk_area = num_rotors * math.pi * (self.R ** 2)

        # Nominal hover induced velocity: v_h = sqrt(mg / (2 * rho * A))
        self.v_h = math.sqrt((self.m * self.g) / (2.0 * self.rho * self.total_disk_area))

    def compute_ground_effect_thrust_ratio(self, altitude_agl_m: float) -> float:
        """
        Calculates thrust augmentation ratio T_IGE / T_OGE via Cheeseman-Bennett model:
          Ratio = 1 / (1 - (R / (4 * z))^2)
        Valid for z_AGL >= 0.5 * R.
        """
        z_safe = max(altitude_agl_m, 0.55 * self.R)
        
        # Ratio of rotor radius to ground height
        r_over_4z = self.R / (4.0 * z_safe)
        ratio_ige = 1.0 / (1.0 - (r_over_4z ** 2))
        return min(1.35, max(1.0, ratio_ige))

    def evaluate_vrs_boundary_hazard(
        self,
        vertical_velocity_ms: float, # Negative for descent in NED convention
        horizontal_ground_speed_ms: float
    ) -> Dict:
        """
        Evaluates proximity to the Vortex Ring State (VRS) boundary where
        rotors re-ingest their own turbulent tip vortices, causing severe thrust loss.
        """
        vz_descent = -vertical_velocity_ms # Positive downward
        v_horizontal = max(0.0, horizontal_ground_speed_ms)

        # Non-dimensional velocity ratios normalized by v_h
        vz_norm = vz_descent / self.v_h
        vx_norm = v_horizontal / self.v_h

        # Empirical Wolkovitch VRS Boundary (Descent between 0.5*vh and 1.5*vh at low forward speed)
        is_vrs_condition = (0.5 <= vz_norm <= 1.5) and (vx_norm < 1.8)
        
        # Hazard severity scaling [0.0 to 1.0]
        if is_vrs_condition:
            severity = 1.0 - (vx_norm / 1.8)
        else:
            severity = 0.0

        action = "MAINTAIN_DESCENT_PROFILE"
        if severity >= 0.70:
            action = "EMERGENCY_VUICHARD_BREAKOUT_FULL_CYCLIC_PITCH"
        elif severity > 0.0:
            action = "INCREASE_FORWARD_AIRSPEED_EXIT_VORTEX_COLUMN"

        return {
            "hover_induced_velocity_vh_ms": round(self.v_h, 2),
            "descent_velocity_ms": round(vz_descent, 2),
            "descent_ratio_vz_over_vh": round(vz_norm, 2),
            "horizontal_ratio_vx_over_vh": round(vx_norm, 2),
            "vrs_hazard_active": is_vrs_condition,
            "vrs_severity_index": round(severity, 3),
            "flight_safety_action": action
        }

    def compute_touchdown_thrust_correction(
        self,
        nominal_hover_thrust_n: float,
        altitude_agl_m: float,
        vertical_velocity_ms: float,
        horizontal_speed_ms: float
    ) -> Dict:
        """
        Calculates corrected rotor thrust command taking into account ground effect
        cushioning and checking for VRS hazards during precision vertiport touchdown.
        """
        ige_factor = self.compute_ground_effect_thrust_ratio(altitude_agl_m)
        vrs_audit = self.evaluate_vrs_boundary_hazard(vertical_velocity_ms, horizontal_speed_ms)

        # In ground effect, required mechanical thrust is reduced to prevent ballooning:
        # T_cmd = T_nominal / Ratio_IGE
        corrected_thrust_cmd_n = nominal_hover_thrust_n / ige_factor

        return {
            "altitude_agl_m": round(altitude_agl_m, 3),
            "ige_thrust_augmentation_factor": round(ige_factor, 3),
            "nominal_thrust_demand_n": round(nominal_hover_thrust_n, 2),
            "corrected_thrust_command_n": round(corrected_thrust_cmd_n, 2),
            "power_saved_in_ground_effect_pct": round((1.0 - 1.0 / ige_factor) * 100.0, 1),
            "vrs_status": vrs_audit["flight_safety_action"]
        }

if __name__ == "__main__":
    safeguard = GroundEffectAndVRSSafeguard()

    print("==========================================================================")
    print("      IN-GROUND EFFECT (IGE) & VORTEX RING STATE (VRS) AERODYNAMIC AUDIT  ")
    print("==========================================================================")
    
    # 1. In-Ground Effect Cushioning Profile during Final Flare (Altitude 0.1m to 2.0m)
    print(f"Hover Induced Velocity (v_h): {safeguard.v_h:.2f} m/s\n")
    print("Ground Effect Thrust Profile during Descent:")
    for alt in [2.0, 1.0, 0.5, 0.25, 0.12]:
        report = safeguard.compute_touchdown_thrust_correction(
            nominal_hover_thrust_n=60.2, altitude_agl_m=alt, vertical_velocity_ms=-0.4, horizontal_speed_ms=0.2
        )
        print(f"  Alt: {report['altitude_agl_m']:4.2f}m AGL | IGE Factor: {report['ige_thrust_augmentation_factor']:.3f}x | "
              f"Thrust Cmd: {report['corrected_thrust_command_n']:5.2f} N | Power Saved: {report['power_saved_in_ground_effect_pct']:4.1f}%")

    # 2. VRS Hazard Trigger Test during Steep Pure-Vertical Descent (V_z = -4.8 m/s, V_xy = 0.2 m/s)
    print("\nVRS Boundary Test (Steep Axial Descent):")
    vrs_test = safeguard.evaluate_vrs_boundary_hazard(vertical_velocity_ms=-4.8, horizontal_ground_speed_ms=0.2)
    for k, v in vrs_test.items():
        print(f"  • {k:35s}: {v}")
    print("==========================================================================\n")