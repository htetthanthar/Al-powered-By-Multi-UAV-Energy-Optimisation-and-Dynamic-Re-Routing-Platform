import math
import numpy as np
from typing import Dict, Tuple, List

class DistributedPropulsionBlownWingModel:
    def __init__(
        self,
        wing_span_m: float = 2.20,
        wing_area_m2: float = 0.520,
        num_dep_propellers: int = 8,
        dep_prop_diameter_m: float = 0.165,
        airfoil_cl0: float = 0.32,
        airfoil_cl_alpha: float = 5.10,          # Lift curve slope per radian
        clean_cl_max: float = 1.45,
        flap_deflection_deg: float = 20.0,
        blown_lift_efficiency_eta: float = 0.88,
        air_density_kg_m3: float = 1.225
    ):
        self.b_span = wing_span_m
        self.s_wing = wing_area_m2
        self.n_props = num_dep_propellers
        self.d_prop = dep_prop_diameter_m
        self.cl0 = airfoil_cl0
        self.cl_alpha = airfoil_cl_alpha
        self.cl_max_clean = clean_cl_max
        self.delta_flap = math.radians(flap_deflection_deg)
        self.eta_blown = blown_lift_efficiency_eta
        self.rho = air_density_kg_m3
        self.g = 9.80665

        # Blown wing area: Total span washed by prop slipstreams
        self.blown_span_m = self.n_props * self.d_prop
        self.s_blown = (self.blown_span_m / self.b_span) * self.s_wing
        self.a_disk_total = self.n_props * math.pi * ((self.d_prop / 2.0) ** 2)

    def compute_blown_slipstream_velocity(
        self,
        freestream_speed_ms: float,
        total_dep_thrust_n: float
    ) -> float:
        """
        Calculates actuator-disk slipstream velocity over the wing leading edge:
          V_slip = V_inf * sqrt(1 + 2*T / (rho * A_disk * V_inf^2))
        """
        v_inf = max(freestream_speed_ms, 0.5)
        # Dynamic thrust coefficient over total propeller disk area
        thrust_loading = total_dep_thrust_n / (self.rho * self.a_disk_total * (v_inf ** 2))
        v_slip = v_inf * math.sqrt(1.0 + 2.0 * thrust_loading)
        return float(v_slip)

    def calculate_blown_lift_and_stall(
        self,
        freestream_speed_ms: float,
        angle_of_attack_rad: float,
        total_dep_thrust_n: float,
        uav_mass_kg: float = 7.50
    ) -> Dict:
        """
        Computes blown wing lift amplification, super-circulation gain, and effective stall speed.
        """
        v_inf = max(freestream_speed_ms, 0.5)
        q_inf = 0.5 * self.rho * (v_inf ** 2)

        # 1. Slipstream Velocity and Dynamic Pressure Ratio
        v_slip = self.compute_blown_slipstream_velocity(v_inf, total_dep_thrust_n)
        dynamic_pressure_ratio = (v_slip / v_inf) ** 2

        # 2. Momentum Blowing Coefficient C_mu = Thrust / (q_inf * S_wing)
        c_mu = total_dep_thrust_n / max(q_inf * self.s_wing, 1e-3)

        # 3. Blown Circulation Lift Increment
        # Baseline clean airfoil lift
        cl_clean = self.cl0 + self.cl_alpha * angle_of_attack_rad
        # Slipstream dynamic pressure augmentation factor
        blown_area_ratio = self.s_blown / self.s_wing
        cl_augmented = cl_clean * (1.0 + self.eta_blown * blown_area_ratio * (dynamic_pressure_ratio - 1.0))

        # Super-circulation jet flap effect: Delta_CL = 1.65 * sqrt(C_mu) * sin(alpha + delta_flap)
        delta_cl_supercirculation = 1.65 * math.sqrt(c_mu) * math.sin(angle_of_attack_rad + self.delta_flap)
        cl_total = cl_augmented + delta_cl_supercirculation

        # 4. Maximum Achievable Lift Coefficient C_L,max(C_mu)
        cl_max_blown = self.cl_max_clean * (1.0 + 0.45 * blown_area_ratio * (dynamic_pressure_ratio - 1.0)) + delta_cl_supercirculation

        # 5. Resulting Lift Force and Dynamic Stall Speeds
        total_lift_n = q_inf * self.s_wing * cl_total
        weight_n = uav_mass_kg * self.g

        v_stall_clean = math.sqrt((2.0 * weight_n) / (self.rho * self.s_wing * self.cl_max_clean))
        v_stall_blown = math.sqrt((2.0 * weight_n) / (self.rho * self.s_wing * max(cl_max_blown, 0.1)))

        is_flying = total_lift_n >= weight_n

        return {
            "freestream_airspeed_ms": round(v_inf, 2),
            "dep_slipstream_velocity_ms": round(v_slip, 2),
            "dynamic_pressure_amplification": round(dynamic_pressure_ratio, 3),
            "momentum_blowing_coeff_cmu": round(c_mu, 3),
            "clean_lift_coeff_cl": round(cl_clean, 3),
            "blown_lift_coeff_cl_total": round(cl_total, 3),
            "lift_amplification_multiplier": round(cl_total / max(cl_clean, 1e-3), 2),
            "total_generated_lift_n": round(total_lift_n, 2),
            "clean_stall_speed_ms": round(v_stall_clean, 2),
            "blown_stall_speed_ms": round(v_stall_blown, 2),
            "stall_speed_reduction_pct": round(((v_stall_clean - v_stall_blown) / v_stall_clean) * 100.0, 1),
            "stol_flight_supported": is_flying
        }

if __name__ == "__main__":
    dep_model = DistributedPropulsionBlownWingModel(wing_span_m=2.20, num_dep_propellers=8)

    print("==========================================================================")
    print("      DISTRIBUTED PROPULSION (DEP) BLOWN-WING AEROELASTIC AUDIT           ")
    print("==========================================================================")

    # Simulate slow STOL climbout at 9.0 m/s (below clean stall speed of 13.6 m/s) with 45 N total DEP thrust
    aoa = math.radians(6.0) # 6 degrees Angle of Attack
    report = dep_model.calculate_blown_lift_and_stall(
        freestream_speed_ms=9.0,
        angle_of_attack_rad=aoa,
        total_dep_thrust_n=45.0,
        uav_mass_kg=7.50
    )

    for k, v in report.items():
        print(f"  • {k:35s}: {v}")
    print("==========================================================================\n")