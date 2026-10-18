import math
import numpy as np
from enum import Enum
from typing import Dict, Tuple, List

class AutorotationPhase(Enum):
    ENTRY_PITCH_DOWN = "ENTRY_PITCH_DOWN"
    STEADY_GLIDE = "STEADY_GLIDE"
    DECELERATION_FLARE = "DECELERATION_FLARE"
    TOUCHDOWN_CUSHION = "TOUCHDOWN_CUSHION"
    POST_LANDING_STOP = "POST_LANDING_STOP"

class AutorotationFlareController:
    def __init__(
        self,
        uav_mass_kg: float = 6.80,
        rotor_radius_m: float = 0.280,            # Variable-pitch rotor radius
        num_rotors: int = 4,
        rotor_inertia_j_kg_m2: float = 0.045,     # Polar moment of inertia per rotor
        nominal_rotor_speed_rpm: float = 4500.0,
        min_recoverable_rpm: float = 2800.0,
        max_blade_pitch_deg: float = 16.0,
        min_autorotation_pitch_deg: float = -2.5, # Negative pitch for windmill spin-up
        touchdown_target_vz_ms: float = 0.85
    ):
        self.m = uav_mass_kg
        self.R = rotor_radius_m
        self.num_rotors = num_rotors
        self.J_rot = rotor_inertia_j_kg_m2
        self.nominal_rpm = nominal_rotor_speed_rpm
        self.min_rpm = min_recoverable_rpm
        self.theta_max = math.radians(max_blade_pitch_deg)
        self.theta_min = math.radians(min_autorotation_pitch_deg)
        self.vz_target = touchdown_target_vz_ms

        self.rho = 1.225
        self.g = 9.80665
        self.total_disk_area = num_rotors * math.pi * (self.R ** 2)

        self.phase = AutorotationPhase.ENTRY_PITCH_DOWN
        self.current_omega_rad_s = (self.nominal_rpm / 60.0) * 2.0 * math.pi

    def compute_autorotative_aerodynamics(
        self,
        omega_rad_s: float,
        descent_speed_vz_ms: float, # Positive downward in autorotation frame
        blade_pitch_rad: float
    ) -> Tuple[float, float]:
        """
        Calculates instantaneous aerodynamic thrust (T) and shaft torque (Q).
        In autorotation: negative collective pitch produces positive torque to spin rotor.
        """
        omega = max(omega_rad_s, 50.0)
        v_tip = omega * self.R
        
        # Non-dimensional inflow ratio lambda = (vz - vi) / v_tip
        # Windmill brake state approximation
        v_inflow_net = descent_speed_vz_ms * 0.75
        lambda_i = v_inflow_net / v_tip

        # Sectional angle of attack alpha = theta - arctan(lambda)
        phi_inflow = math.atan2(v_inflow_net, v_tip)
        alpha = blade_pitch_rad - phi_inflow

        # Empirical lift/drag polar for symmetrical NACA 0012 variable-pitch blade
        cl = 5.7 * alpha
        cd = 0.011 + 0.95 * (math.sin(alpha) ** 2)

        # Aerodynamic force coefficients
        ct = (cl * math.cos(phi_inflow) - cd * math.sin(phi_inflow)) * 0.08
        cq = (cl * math.sin(phi_inflow) + cd * math.cos(phi_inflow)) * 0.08

        # Total thrust and driving torque across all rotors
        # Positive Q_aero accelerates rotor (when phi_inflow drives blades forward)
        thrust_n = self.num_rotors * (0.5 * self.rho * (v_tip ** 2) * math.pi * (self.R ** 2) * ct)
        torque_nm = self.num_rotors * (0.5 * self.rho * (v_tip ** 2) * math.pi * (self.R ** 3) * cq)

        return float(thrust_n), float(-torque_nm) # Inverting torque sign: driving torque is positive

    def calculate_flare_initiation_height(
        self,
        descent_speed_vz_ms: float,
        current_rpm: float
    ) -> float:
        """
        Calculates the height AGL required to arrest vertical descent velocity.
        """
        omega = (current_rpm / 60.0) * 2.0 * math.pi
        
        # Peak burst thrust achievable at maximum collective pitch
        t_max_burst, _ = self.compute_autorotative_aerodynamics(omega, descent_speed_vz_ms, self.theta_max)
        max_decel_az = (t_max_burst / self.m) - self.g
        max_decel_az = max(max_decel_az, 1.5) # Minimum 1.5 m/s^2 deceleration capability

        h_flare = (descent_speed_vz_ms ** 2 - self.vz_target ** 2) / (2.0 * max_decel_az)
        return float(max(1.5, h_flare))

    def step_autorotation_control(
        self,
        dt_seconds: float,
        altitude_agl_m: float,
        vertical_descent_rate_ms: float, # Positive downwards
        horizontal_airspeed_ms: float
    ) -> Dict:
        """
        Executes multi-phase autorotation state machine and computes optimal collective blade pitch.
        """
        current_rpm = (self.current_omega_rad_s / (2.0 * math.pi)) * 60.0
        h_flare_trigger = self.calculate_flare_initiation_height(vertical_descent_rate_ms, current_rpm)

        # 1. Phase Transition Logic
        if altitude_agl_m <= 0.15:
            self.phase = AutorotationPhase.POST_LANDING_STOP
            cmd_pitch = 0.0
        elif altitude_agl_m <= 1.2 and self.phase == AutorotationPhase.DECELERATION_FLARE:
            self.phase = AutorotationPhase.TOUCHDOWN_CUSHION
            # Maximum collective pitch to convert all remaining rotor inertia into cushioning thrust
            cmd_pitch = self.theta_max
        elif altitude_agl_m <= h_flare_trigger and self.phase in [AutorotationPhase.ENTRY_PITCH_DOWN, AutorotationPhase.STEADY_GLIDE]:
            self.phase = AutorotationPhase.DECELERATION_FLARE
            cmd_pitch = self.theta_max * 0.75
        elif self.phase == AutorotationPhase.ENTRY_PITCH_DOWN:
            # Low/negative pitch to preserve and spin up rotor RPM
            cmd_pitch = self.theta_min
            if current_rpm >= (self.nominal_rpm * 0.90):
                self.phase = AutorotationPhase.STEADY_GLIDE
        else:
            # Steady glide: Closed-loop RPM governor regulating collective pitch
            rpm_error = (self.nominal_rpm * 0.95) - current_rpm
            cmd_pitch = math.radians(1.5) + 0.00008 * rpm_error
            cmd_pitch = float(np.clip(cmd_pitch, self.theta_min, math.radians(6.0)))

        # 2. Integrate Rotor RPM Dynamics: J * dOmega/dt = Q_aero - Q_friction
        thrust_n, torque_nm = self.compute_autorotative_aerodynamics(
            self.current_omega_rad_s, vertical_descent_rate_ms, cmd_pitch
        )
        q_bearing_friction = 0.08 * (self.current_omega_rad_s / 400.0)
        d_omega_dt = (torque_nm - q_bearing_friction) / (self.num_rotors * self.J_rot)
        
        self.current_omega_rad_s = max(50.0, self.current_omega_rad_s + d_omega_dt * dt_seconds)
        new_rpm = (self.current_omega_rad_s / (2.0 * math.pi)) * 60.0

        # Stored rotor kinetic energy: E = 0.5 * J * Omega^2
        rotor_kinetic_energy_j = 0.5 * (self.num_rotors * self.J_rot) * (self.current_omega_rad_s ** 2)

        return {
            "autorotation_phase": self.phase.value,
            "altitude_agl_m": round(altitude_agl_m, 2),
            "rotor_speed_rpm": round(new_rpm, 1),
            "commanded_blade_pitch_deg": round(math.degrees(cmd_pitch), 2),
            "instantaneous_thrust_n": round(thrust_n, 2),
            "calculated_flare_height_m": round(h_flare_trigger, 2),
            "stored_rotor_energy_joules": round(rotor_kinetic_energy_j, 1),
            "rpm_recoverable": new_rpm >= self.min_rpm,
            "flare_ready": altitude_agl_m <= h_flare_trigger
        }

if __name__ == "__main__":
    auto_ctrl = AutorotationFlareController()

    print("==========================================================================")
    print("      UNPOWERED AUTOROTATION DESCENT & GROUND-EFFECT FLARE AUDIT          ")
    print("==========================================================================")
    
    # Simulate unpowered emergency descent from 40m AGL (Vertical descent rate = 9.5 m/s)
    alt = 40.0
    vz = 9.5
    dt = 0.5

    for step in range(1, 13):
        telemetry = auto_ctrl.step_autorotation_control(
            dt_seconds=dt, altitude_agl_m=alt, vertical_descent_rate_ms=vz, horizontal_airspeed_ms=4.0
        )
        print(f"T+{(step*dt):4.1f}s | Alt: {telemetry['altitude_agl_m']:5.2f}m | Phase: {telemetry['autorotation_phase']:19s} | "
              f"RPM: {telemetry['rotor_speed_rpm']:6.1f} | Pitch: {telemetry['commanded_blade_pitch_deg']:+5.2f}° | "
              f"Thrust: {telemetry['instantaneous_thrust_n']:5.1f}N | Stored E: {telemetry['stored_rotor_energy_joules']:6.1f}J")
        
        # Update descent profile
        if telemetry["autorotation_phase"] == "DECELERATION_FLARE":
            vz = max(1.2, vz - 3.2 * dt)
        elif telemetry["autorotation_phase"] == "TOUCHDOWN_CUSHION":
            vz = 0.85
        alt = max(0.0, alt - vz * dt)
        if alt <= 0.05:
            break
    print("==========================================================================\n")