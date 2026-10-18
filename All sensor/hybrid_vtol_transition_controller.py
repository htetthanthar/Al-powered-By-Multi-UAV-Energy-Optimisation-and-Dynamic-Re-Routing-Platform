import math
import numpy as np
from enum import Enum
from typing import Dict, Tuple

class TransitionState(Enum):
    HOVER_ROTARY = "HOVER_ROTARY"
    TRANSITION_FORWARD = "TRANSITION_FORWARD"
    FIXED_WING_CRUISE = "FIXED_WING_CRUISE"
    TRANSITION_BACK = "TRANSITION_BACK"

class HybridVTOLTransitionController:
    def __init__(
        self,
        total_mass_kg: float = 7.80,         # Hybrid VTOL airframe MTOW
        wing_span_m: float = 1.85,
        wing_area_m2: float = 0.420,
        airfoil_cl0: float = 0.28,
        airfoil_cl_alpha: float = 4.85,       # Lift curve slope per radian
        cd0_parasite: float = 0.024,
        oswald_efficiency_e: float = 0.82,
        stall_speed_ms: float = 13.5,         # V_stall in m/s (~48.6 km/h)
        cruise_speed_ms: float = 24.0,        # V_cruise in m/s (~86.4 km/h)
        max_transition_time_s: float = 12.0
    ):
        self.m = total_mass_kg
        self.b_span = wing_span_m
        self.s_area = wing_area_m2
        self.cl0 = airfoil_cl0
        self.cl_alpha = airfoil_cl_alpha
        self.cd0 = cd0_parasite
        self.v_stall = stall_speed_ms
        self.v_cruise = cruise_speed_ms
        self.t_trans_max = max_transition_time_s

        self.rho = 1.225
        self.g = 9.80665
        self.weight_n = total_mass_kg * self.g

        # Induced drag factor: k = 1 / (pi * AR * e)
        aspect_ratio = (self.b_span ** 2) / self.s_area
        self.k_induced = 1.0 / (math.pi * aspect_ratio * oswald_efficiency_e)

        self.mode = TransitionState.HOVER_ROTARY
        self.transition_timer_s = 0.0

    def compute_aerodynamic_coefficients(self, angle_of_attack_rad: float) -> Tuple[float, float]:
        """Calculates wing lift and drag coefficients: C_L(alpha), C_D(alpha)."""
        alpha_eff = np.clip(angle_of_attack_rad, math.radians(-4.0), math.radians(12.0))
        c_l = self.cl0 + self.cl_alpha * alpha_eff
        c_d = self.cd0 + self.k_induced * (c_l ** 2)
        return float(c_l), float(c_d)

    def execute_forward_transition_step(
        self,
        dt_seconds: float,
        current_airspeed_ms: float,
        pitch_angle_rad: float = math.radians(3.5),
        target_climb_rate_ms: float = 0.0
    ) -> Dict:
        """
        Calculates progressive motor thrust blending from quad rotors to forward pusher motor.
        """
        self.transition_timer_s += dt_seconds
        v_air = max(current_airspeed_ms, 0.5)

        # 1. Calculate Aerodynamic Wing Lift and Drag
        c_l, c_d = self.compute_aerodynamic_coefficients(pitch_angle_rad)
        q_dynamic = 0.5 * self.rho * (v_air ** 2)
        wing_lift_n = q_dynamic * self.s_area * c_l
        wing_drag_n = q_dynamic * self.s_area * c_d

        # 2. Lift Blending Ratio: eta = 0 (Pure Hover) -> 1 (Pure Wing Flight)
        blend_ratio = np.clip(
            (v_air ** 2 - 0.0) / (self.v_cruise ** 2 - 0.0), 0.0, 1.0
        )

        # 3. Rotary Vertical Thrust Demand (Compensating weight deficit)
        vertical_thrust_needed = max(0.0, self.weight_n - wing_lift_n * math.cos(pitch_angle_rad))
        
        # 4. Pusher Forward Motor Thrust Demand: T_pusher = Drag + m * a_desired
        desired_accel_ms2 = (self.v_cruise - v_air) / max(self.t_trans_max - self.transition_timer_s, 1.0)
        desired_accel_ms2 = float(np.clip(desired_accel_ms2, 0.5, 2.5))
        pusher_thrust_needed = wing_drag_n + (self.m * desired_accel_ms2)

        # 5. State Transition Logic
        if v_air >= self.v_cruise and self.mode == TransitionState.TRANSITION_FORWARD:
            self.mode = TransitionState.FIXED_WING_CRUISE
            vertical_thrust_needed = 0.0 # Shutdown VTOL lift motors
            status = "TRANSITION_COMPLETE_FIXED_WING_ENGAGED"
        elif v_air > self.v_stall:
            self.mode = TransitionState.TRANSITION_FORWARD
            status = "CORRIDOR_SAFE_ACCELERATING_ABOVE_STALL"
        else:
            self.mode = TransitionState.TRANSITION_FORWARD
            status = "SUB_STALL_VTOL_THRUST_ASSISTED"

        # Electrical power consumption across all motors
        p_vtol_electrical = (vertical_thrust_needed ** 1.5) / math.sqrt(2.0 * self.rho * (4.0 * math.pi * 0.1905**2)) / 0.80
        p_pusher_electrical = (pusher_thrust_needed * v_air) / 0.75 + 35.0
        total_power_watts = p_vtol_electrical + p_pusher_electrical

        return {
            "transition_mode": self.mode.value,
            "airspeed_ms": round(v_air, 2),
            "wing_lift_n": round(wing_lift_n, 2),
            "wing_lift_share_pct": round((wing_lift_n / self.weight_n) * 100.0, 1),
            "vtol_quad_thrust_command_n": round(vertical_thrust_needed, 2),
            "pusher_forward_thrust_command_n": round(pusher_thrust_needed, 2),
            "lift_blending_factor_eta": round(float(blend_ratio), 3),
            "instantaneous_total_power_watts": round(total_power_watts, 1),
            "corridor_status": status,
            "vtol_motors_active": vertical_thrust_needed > 1.0
        }

if __name__ == "__main__":
    vtol_ctrl = HybridVTOLTransitionController()

    print("==========================================================================")
    print("      HYBRID VTOL TRANSITIONAL FLIGHT CORRIDOR & LIFT BLENDING AUDIT      ")
    print("==========================================================================")
    
    # Simulate a 10-second forward transition from hover to 24 m/s cruise
    for step in range(1, 11):
        # Forward speed accelerates from 2 m/s up to 24 m/s
        v_current = 2.0 + (step * 2.2)
        audit = vtol_ctrl.execute_forward_transition_step(
            dt_seconds=1.0, current_airspeed_ms=v_current, pitch_angle_rad=math.radians(3.5)
        )
        print(f"Step {step:02d} (Airspeed: {audit['airspeed_ms']:4.1f} m/s) | Lift Share: {audit['wing_lift_share_pct']:5.1f}% | "
              f"VTOL Thrust: {audit['vtol_quad_thrust_command_n']:5.1f} N | Pusher: {audit['pusher_forward_thrust_command_n']:5.1f} N | "
              f"Power: {audit['instantaneous_total_power_watts']:5.1f} W | State: {audit['corridor_status']}")
    print("==========================================================================\n")