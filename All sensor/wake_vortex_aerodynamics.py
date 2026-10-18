import math
import numpy as np
from typing import Dict, List, Tuple

class MultiUAVWakeVortexHazardModel:
    def __init__(
        self,
        uav_mass_kg: float = 6.14,
        rotor_diameter_m: float = 0.381, # 15-inch props
        num_rotors: int = 4,
        air_density_kg_m3: float = 1.225
    ):
        self.mass = uav_mass_kg
        self.d_rotor = rotor_diameter_m
        self.num_rotors = num_rotors
        self.rho = air_density_kg_m3
        self.g = 9.80665

        # Effective vortex span: b0 = (pi / 4) * B_effective
        self.span_effective_m = 0.65 # Frame arm diameter
        self.b0 = (math.pi / 4.0) * self.span_effective_m

    def compute_initial_vortex_strength(self, forward_speed_ms: float) -> Dict:
        """
        Calculates initial wake vortex circulation Gamma_0 using the Kutta-Joukowski theorem:
          L = rho * v_air * b0 * Gamma_0  =>  Gamma_0 = (m * g) / (rho * v_air * b0)
        """
        v_air = max(forward_speed_ms, 2.0)
        weight_n = self.mass * self.g

        gamma_0 = weight_n / (self.rho * v_air * self.b0)
        
        # Initial mutual vortex pair descent velocity: w_0 = Gamma_0 / (2 * pi * b0)
        w_descent_0 = gamma_0 / (2.0 * math.pi * self.b0)

        # Characteristic timescale: t_0 = b0 / w_0
        t_0 = self.b0 / max(w_descent_0, 1e-4)

        return {
            "initial_circulation_gamma0_m2_s": round(gamma_0, 3),
            "initial_descent_velocity_w0_ms": round(w_descent_0, 2),
            "characteristic_time_t0_s": round(t_0, 2)
        }

    def compute_induced_rolling_moment(
        self,
        trailing_uav_offset_ned_m: Tuple[float, float, float],
        trailing_uav_roll_inertia_jxx: float,
        time_elapsed_s: float,
        forward_speed_ms: float = 18.05
    ) -> Dict:
        """
        Calculates the tangential velocity field v_theta(r) from Burnham-Hallock decay model:
          v_theta(r) = (Gamma(t) / (2 * pi)) * (r / (r^2 + r_core^2))
        """
        dx, dy, dz = trailing_uav_offset_ned_m
        distance_behind_m = abs(dx)

        # Circulation decay via eddy dissipation (Crow instability model)
        vortex_params = self.compute_initial_vortex_strength(forward_speed_ms)
        gamma_0 = vortex_params["initial_circulation_gamma0_m2_s"]
        t_0 = vortex_params["characteristic_time_t0_s"]

        decay_ratio = math.exp(-0.45 * (time_elapsed_s / t_0))
        gamma_t = gamma_0 * decay_ratio

        # Radial distance to primary vortex core
        r_core = 0.05 * self.b0
        radial_r = math.hypot(dy, dz)

        v_tangential = (gamma_t / (2.0 * math.pi)) * (radial_r / (radial_r**2 + r_core**2))

        # Induced roll acceleration on trailing UAV
        # Differential lift across wingspan / rotor disk creates induced roll torque:
        # tau_roll = 0.5 * rho * v_tan^2 * Area * arm
        disk_area = self.num_rotors * math.pi * ((self.d_rotor / 2.0)**2)
        tau_induced_nm = 0.5 * self.rho * (v_tangential**2) * disk_area * (self.span_effective_m / 2.0)
        
        roll_accel_rad_s2 = tau_induced_nm / trailing_uav_roll_inertia_jxx

        # Safety determination: Quadcopter attitude controller authority limit is ~25 rad/s^2
        is_safe = roll_accel_rad_s2 <= 4.0

        return {
            "vortex_age_seconds": round(time_elapsed_s, 2),
            "attenuated_circulation_m2_s": round(gamma_t, 3),
            "tangential_induced_velocity_ms": round(v_tangential, 2),
            "induced_rolling_torque_nm": round(tau_induced_nm, 3),
            "induced_roll_accel_rad_s2": round(roll_accel_rad_s2, 2),
            "separation_safe": is_safe,
            "required_trailing_separation_m": round(max(35.0, 18.05 * (t_0 * 1.8)), 1)
        }

if __name__ == "__main__":
    hazard_model = MultiUAVWakeVortexHazardModel()
    
    print("==========================================================================")
    print("      MULTI-UAV AERODYNAMIC WAKE VORTEX & DOWNWASH HAZARD AUDIT          ")
    print("==========================================================================")
    
    v_init = hazard_model.compute_initial_vortex_strength(forward_speed_ms=18.05)
    for k, v in v_init.items():
        print(f"  • {k:35s}: {v}")

    # Evaluate trailing drone at varying separation distances behind leading UAV
    print("\nTrailing Aircraft Wake Interaction Profile (Inertia Jxx = 0.082 kg·m²):")
    for distance_m in [10.0, 25.0, 50.0, 80.0]:
        t_lag = distance_m / 18.05
        # Trailing drone 0.8m offset laterally and 0.5m below leading flight path
        offset = (-distance_m, 0.8, 0.5)
        eval_hazard = hazard_model.compute_induced_rolling_moment(
            offset, trailing_uav_roll_inertia_jxx=0.082, time_elapsed_s=t_lag
        )
        print(f"  Separation {distance_m:4.1f}m (Lag: {t_lag:4.2f}s) | V_tan: {eval_hazard['tangential_induced_velocity_ms']:4.2f} m/s | "
              f"Roll Accel: {eval_hazard['induced_roll_accel_rad_s2']:5.2f} rad/s² | Status: {'SAFE' if eval_hazard['separation_safe'] else 'UPSET HAZARD'}")
    print("==========================================================================\n")