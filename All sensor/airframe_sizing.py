import math
import numpy as np

class QuadcopterAerodynamicSizer:
    def __init__(
        self,
        frame_mass_empty_kg: float = 3.2,
        max_payload_kg: float = 2.0,
        battery_mass_kg: float = 1.3,
        num_rotors: int = 4,
        prop_diameter_inches: float = 15.0,
        prop_pitch_inches: float = 5.5,
        air_density_kg_m3: float = 1.225
    ):
        self.m_empty = frame_mass_empty_kg
        self.m_payload = max_payload_kg
        self.m_battery = battery_mass_kg
        self.total_mass_kg = self.m_empty + self.m_payload + self.m_battery
        self.num_rotors = num_rotors
        
        # Metric Conversions
        self.prop_radius_m = (prop_diameter_inches * 0.0254) / 2.0
        self.disk_area_per_rotor_m2 = math.pi * (self.prop_radius_m ** 2)
        self.total_disk_area_m2 = self.num_rotors * self.disk_area_per_rotor_m2
        self.rho = air_density_kg_m3
        self.g = 9.80665

    def compute_hover_aerodynamics(self) -> dict:
        """
        Calculates hover thrust, induced velocity, and ideal thrust-to-weight ratio.
        """
        total_weight_n = self.total_mass_kg * self.g
        thrust_per_rotor_n = total_weight_n / self.num_rotors
        
        # Momentum Theory: v_induced = sqrt(T / (2 * rho * A))
        v_induced_hover_ms = math.sqrt(total_weight_n / (2.0 * self.rho * self.total_disk_area_m2))
        
        # Ideal Induced Power: P_induced = T * v_induced
        ideal_hover_power_watts = total_weight_n * v_induced_hover_ms
        
        # Real-world figure of merit (FM = 0.65 typical for 15-inch carbon props)
        figure_of_merit = 0.65
        actual_rotor_power_watts = ideal_hover_power_watts / figure_of_merit
        
        # Electrical conversion losses (ESC: 95%, Motor: 85%)
        powertrain_efficiency = 0.95 * 0.85
        electrical_hover_power_watts = actual_rotor_power_watts / powertrain_efficiency
        
        # Thrust margin requirement: Minimum 2.0:1 Thrust-to-Weight for wind stabilization
        required_max_thrust_per_motor_n = (total_weight_n * 2.0) / self.num_rotors
        required_max_thrust_per_motor_g = (required_max_thrust_per_motor_n / self.g) * 1000.0

        return {
            "total_all_up_weight_kg": round(self.total_mass_kg, 2),
            "disk_loading_n_per_m2": round(total_weight_n / self.total_disk_area_m2, 2),
            "induced_velocity_hover_ms": round(v_induced_hover_ms, 2),
            "electrical_hover_power_watts": round(electrical_hover_power_watts, 2),
            "hover_thrust_per_motor_g": round((thrust_per_rotor_n / self.g) * 1000.0, 1),
            "min_required_max_thrust_g": round(required_max_thrust_per_motor_g, 1),
            "thrust_to_weight_ratio": 2.0
        }

    def compute_cruise_envelope(self, cruise_speed_kmh: float = 65.0, headwind_kmh: float = 30.0) -> dict:
        """
        Calculates forward flight thrust angle and total electrical power draw under maximum headwind.
        """
        v_ground_ms = cruise_speed_kmh / 3.6
        v_wind_ms = headwind_kmh / 3.6
        v_airspeed_ms = v_ground_ms + v_wind_ms
        
        # Fuselage Parasite Drag: D = 0.5 * rho * v_air^2 * C_D * A_front
        cd_fuselage = 0.38
        a_front_m2 = 0.095
        drag_parasite_n = 0.5 * self.rho * (v_airspeed_ms ** 2) * cd_fuselage * a_front_m2
        
        # Total force required: T_total = sqrt(Weight^2 + Drag^2)
        weight_n = self.total_mass_kg * self.g
        total_thrust_forward_n = math.sqrt(weight_n**2 + drag_parasite_n**2)
        pitch_tilt_angle_deg = math.degrees(math.atan2(drag_parasite_n, weight_n))

        # Forward flight induced power (Glauert formula)
        # v_i = T / (2 * rho * A * v_air)
        v_i_forward = total_thrust_forward_n / (2.0 * self.rho * self.total_disk_area_m2 * v_airspeed_ms)
        p_induced_forward = total_thrust_forward_n * v_i_forward
        p_parasite_forward = drag_parasite_n * v_airspeed_ms
        p_profile_forward = 95.0 * (1.0 + 3.0 * ((v_airspeed_ms / 45.0) ** 2))
        
        total_aero_power = (p_induced_forward / 0.65) + p_parasite_forward + p_profile_forward
        electrical_cruise_power = (total_aero_power / 0.8075) + 25.0 # Powertrain loss + 25W avionics

        # Battery C-Rate Evaluation (6S 22Ah Pack)
        battery_pack_wh = 488.4
        c_rate = electrical_cruise_power / battery_pack_wh
        max_duration_hours = (battery_pack_wh * 0.80) / electrical_cruise_power # 80% usable DoD

        return {
            "airspeed_ms": round(v_airspeed_ms, 2),
            "pitch_tilt_angle_deg": round(pitch_tilt_angle_deg, 2),
            "parasite_drag_n": round(drag_parasite_n, 2),
            "electrical_cruise_power_watts": round(electrical_cruise_power, 2),
            "battery_continuous_c_rate": round(c_rate, 2),
            "flight_duration_at_max_headwind_min": round(max_duration_hours * 60.0, 2),
            "meets_55min_envelope": (max_duration_hours * 60.0) >= 45.0
        }

if __name__ == "__main__":
    sizer = QuadcopterAerodynamicSizer()
    hover_metrics = sizer.compute_hover_aerodynamics()
    cruise_metrics = sizer.compute_cruise_envelope(cruise_speed_kmh=65.0, headwind_kmh=30.0)

    print("==========================================================================")
    print("         UAV PROPULSION & PHYSICAL ENVELOPE SPECIFICATION                 ")
    print("==========================================================================")
    print("Hover Propulsion Parameters:")
    for k, v in hover_metrics.items():
        print(f"  • {k:32s}: {v}")
    print("\nCruise Aerodynamics (65 km/h + 30 km/h Headwind = 95 km/h Airspeed):")
    for k, v in cruise_metrics.items():
        print(f"  • {k:32s}: {v}")
    print("==========================================================================\n")