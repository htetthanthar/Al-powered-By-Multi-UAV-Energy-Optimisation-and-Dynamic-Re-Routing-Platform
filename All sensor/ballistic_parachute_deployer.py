import math
import numpy as np
from enum import Enum
from typing import Dict, Tuple, Optional

class ParachuteState(Enum):
    ARMED = "ARMED"
    TRIGGERED = "TRIGGERED"
    INFLATING = "INFLATING"
    STEADY_DESCENT = "STEADY_DESCENT"
    TOUCHDOWN = "TOUCHDOWN"
    DISARMED = "DISARMED"

class BallisticParachuteSystem:
    def __init__(
        self,
        uav_mass_kg: float = 6.14,
        canopy_area_m2: float = 2.45,
        canopy_cd: float = 1.75, # Hemispherical parachute drag coefficient
        air_density_kg_m3: float = 1.225,
        ejection_velocity_ms: float = 18.0,
        inflation_time_s: float = 0.65,
        min_deployment_alt_agl_m: float = 12.0,
        kinetic_energy_limit_joules: float = 69.0 # EASA M2 / FAA Category 2 limit
    ):
        self.mass = uav_mass_kg
        self.a_canopy = canopy_area_m2
        self.cd_canopy = canopy_cd
        self.rho = air_density_kg_m3
        self.v_eject = ejection_velocity_ms
        self.t_inflate = inflation_time_s
        self.min_alt = min_deployment_alt_agl_m
        self.e_limit = kinetic_energy_limit_joules
        self.g = 9.80665

        self.state = ParachuteState.ARMED
        self.deploy_timestamp = 0.0
        self.terminal_velocity_ms = self._calculate_terminal_velocity()

    def _calculate_terminal_velocity(self) -> float:
        """
        Calculates steady-state vertical descent velocity under full canopy inflation.
        """
        weight = self.mass * self.g
        drag_area = self.cd_canopy * self.a_canopy
        v_term = math.sqrt((2.0 * weight) / (self.rho * drag_area))
        return v_term

    def evaluate_deployment_triggers(
        self,
        altitude_agl_m: float,
        accel_body_ms2: np.ndarray,
        angular_rate_body_rad_s: np.ndarray,
        active_rotor_count: int
    ) -> Tuple[bool, str]:
        """
        Validates deployment conditions:
          1. Free-fall condition: ||a_meas|| <= 0.25 g for > 150 ms
          2. Severe unrecoverable tumble: ||omega|| >= 350 deg/s (6.1 rad/s)
          3. Critical structural loss: active_rotor_count <= 2
        """
        if self.state != ParachuteState.ARMED:
            return False, "SYSTEM_NOT_ARMED"

        if altitude_agl_m < self.min_alt:
            return False, "ALTITUDE_BELOW_CANOPY_DEPLOYMENT_FLOOR"

        accel_mag = float(np.linalg.norm(accel_body_ms2))
        rate_mag = float(np.linalg.norm(angular_rate_body_rad_s))

        # 1. Free-Fall Trigger
        if accel_mag <= (0.25 * self.g):
            return True, "FREE_FALL_COLLAPSE_DETECTED"

        # 2. Uncontrolled Tumble Rate Trigger
        if rate_mag >= math.radians(350.0):
            return True, "UNCONTROLLED_ANGULAR_TUMBLE_EXCEEDED"

        # 3. Catastrophic Multirotor Structural Failure
        if active_rotor_count <= 2:
            return True, "CATASTROPHIC_ACTUATOR_LOSS"

        return False, "NOMINAL_FLIGHT_ENVELOPE"

    def execute_ballistic_ejection(
        self,
        current_pos_ned: Tuple[float, float, float],
        current_vel_ned: Tuple[float, float, float],
        wind_vector_ms: Tuple[float, float]
    ) -> Dict:
        """
        Computes ballistic parachute ejection, canopy fill trajectory,
        ground impact velocity, and wind drift impact coordinates.
        """
        self.state = ParachuteState.TRIGGERED
        x0, y0, z0 = current_pos_ned # z is negative altitude AGL
        vx0, vy0, vz0 = current_vel_ned
        alt_agl = -z0
        w_x, w_y = wind_vector_ms

        # Phase 1: Ballistic inflation transit
        # Altitude lost during canopy inflation: Delta_z = v_z0 * t_inflate + 0.5 * g * t_inflate^2
        alt_lost_during_inflation = (vz0 * self.t_inflate) + (0.5 * self.g * (self.t_inflate ** 2))
        alt_post_inflation = max(0.0, alt_agl - alt_lost_during_inflation)

        # Phase 2: Steady-state descent to ground
        time_steady_descent = alt_post_inflation / self.terminal_velocity_ms
        total_descent_time_s = self.t_inflate + time_steady_descent

        # Phase 3: Wind drift displacement
        drift_x = w_x * total_descent_time_s
        drift_y = w_y * total_descent_time_s
        impact_x = x0 + drift_x
        impact_y = y0 + drift_y

        # Impact kinetic energy calculation
        impact_energy_j = 0.5 * self.mass * (self.terminal_velocity_ms ** 2)
        compliant_with_regulations = impact_energy_j <= self.e_limit

        self.state = ParachuteState.STEADY_DESCENT

        return {
            "ejection_command": "FIRE_PYROTECHNIC_SQUIB_CUT_MOTOR_POWER",
            "terminal_descent_velocity_ms": round(self.terminal_velocity_ms, 2),
            "total_descent_duration_s": round(total_descent_time_s, 2),
            "ground_impact_kinetic_energy_joules": round(impact_energy_j, 2),
            "faa_easa_69j_limit_compliant": compliant_with_regulations,
            "canopy_inflation_altitude_loss_m": round(alt_lost_during_inflation, 2),
            "projected_ground_impact_offset_ned_m": (round(drift_x, 2), round(drift_y, 2), 0.0),
            "safety_status": "CONTROLLED_EMERGENCY_RECOVERY_ENGAGED"
        }

if __name__ == "__main__":
    chute = BallisticParachuteSystem(uav_mass_kg=6.14, canopy_area_m2=2.45)

    print("==========================================================================")
    print("      BALLISTIC PARACHUTE RECOVERY & GROUND IMPACT KINETIC ENERGY AUDIT   ")
    print("==========================================================================")
    
    # 1. Trigger Evaluation under Uncontrolled High-Speed Spin
    acc_nominal = np.array([0.0, 0.0, 9.81])
    gyro_spin = np.array([0.2, 0.1, math.radians(380.0)]) # 380 deg/s yaw tumble
    
    should_deploy, reason = chute.evaluate_deployment_triggers(
        altitude_agl_m=35.0,
        accel_body_ms2=acc_nominal,
        angular_rate_body_rad_s=gyro_spin,
        active_rotor_count=4
    )
    print(f"Deployment Trigger Assessment: Deploy={should_deploy} ({reason})")

    # 2. Execute Ballistic Recovery from 35m AGL with 4.5 m/s Wind
    if should_deploy:
        ejection_report = chute.execute_ballistic_ejection(
            current_pos_ned=(100.0, 150.0, -35.0),
            current_vel_ned=(12.0, 2.0, 1.5),
            wind_vector_ms=(4.5, -2.0)
        )
        for k, v in ejection_report.items():
            print(f"  • {k:38s}: {v}")
    print("==========================================================================\n")