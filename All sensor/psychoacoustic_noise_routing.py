import math
import numpy as np
from typing import Dict, List, Tuple

class RotorAeroAcousticModel:
    def __init__(
        self,
        num_rotors: int = 4,
        blade_count_per_rotor: int = 2,
        prop_radius_m: float = 0.1905,  # 15-inch propeller
        sound_ref_pressure_pa: float = 20e-6, # 20 uPa (0 dB threshold of human hearing)
        air_density_kg_m3: float = 1.225,
        speed_of_sound_ms: float = 343.0,
        ground_absorption_db_per_100m: float = 0.55 # A-weighted atmospheric absorption
    ):
        self.b = blade_count_per_rotor
        self.num_rotors = num_rotors
        self.r_prop = prop_radius_m
        self.p_ref = sound_ref_pressure_pa
        self.rho = air_density_kg_m3
        self.c_sound = speed_of_sound_ms
        self.alpha_absorb = ground_absorption_db_per_100m / 100.0

    def compute_acoustic_emission(
        self,
        thrust_per_rotor_n: float,
        electrical_power_watts: float,
        tip_speed_ratio: float = 0.45 # M_tip = v_tip / c
    ) -> Dict:
        """
        Estimates acoustic source power level (PWL) based on Gutin-Lowson rotational loading
        and broadband turbulent vortex shedding formulations.
        """
        # Tip Mach Number & Blade Tip Velocity
        m_tip = tip_speed_ratio
        v_tip = m_tip * self.c_sound
        rpm = (v_tip / (2.0 * math.pi * self.r_prop)) * 60.0
        blade_pass_freq_hz = (rpm / 60.0) * self.b

        # 1. Rotational Loading Acoustic Power (Lowson's Formulation)
        # W_rotational proportional to (Thrust^2 * M_tip^2) / (rho * c * A)
        disk_area = math.pi * (self.r_prop ** 2)
        w_rotational = (thrust_per_rotor_n ** 2 * (m_tip ** 2)) / (self.rho * self.c_sound * disk_area * 12.0)

        # 2. Broadband Turbulent Ingestion Noise (proportional to electrical power input)
        acoustic_efficiency = 1.2e-4  # ~0.012% electrical-to-acoustic power conversion
        w_broadband = (electrical_power_watts / self.num_rotors) * acoustic_efficiency

        total_acoustic_power_watts = (w_rotational + w_broadband) * self.num_rotors
        total_acoustic_power_watts = max(total_acoustic_power_watts, 1e-12)

        # Acoustic Sound Power Level (SWL, dB re 1 pW)
        swl_db = 120.0 + 10.0 * math.log10(total_acoustic_power_watts)

        return {
            "propeller_rpm": round(rpm, 1),
            "blade_pass_frequency_hz": round(blade_pass_freq_hz, 1),
            "total_acoustic_power_watts": f"{total_acoustic_power_watts:.4e}",
            "sound_power_level_swl_db": round(swl_db, 2)
        }

    def compute_ground_spl_footprint(
        self,
        swl_db: float,
        drone_pos_ned: Tuple[float, float, float],
        ground_receiver_ned: Tuple[float, float, float]
    ) -> float:
        """
        Computes ground Sound Pressure Level (SPL, in dBA) considering hemispherical geometric
        spreading, atmospheric attenuation, and A-weighting spectral corrections.
        """
        dx = ground_receiver_ned[0] - drone_pos_ned[0]
        dy = ground_receiver_ned[1] - drone_pos_ned[1]
        dz = ground_receiver_ned[2] - drone_pos_ned[2]  # dz is positive down
        
        distance_r_m = math.sqrt(dx**2 + dy**2 + dz**2)
        distance_r_m = max(distance_r_m, 1.0)

        # Hemispherical free-field propagation: SPL = SWL - 20*log10(r) - 8.0 dB
        geometric_divergence = 20.0 * math.log10(distance_r_m) + 8.0
        
        # Atmospheric absorption loss
        atmospheric_loss = self.alpha_absorb * distance_r_m

        # A-Weighting correction factor for ~120 Hz multirotor BPF fundamental
        a_weighting_correction = -16.1

        spl_ground_dba = swl_db - geometric_divergence - atmospheric_loss + a_weighting_correction
        return max(20.0, spl_ground_dba)

    def evaluate_route_noise_penalty(
        self,
        trajectory_waypoints: List[Dict],
        noise_sensitive_zones: List[Dict],
        nominal_swl_db: float = 88.5
    ) -> Dict:
        """
        Integrates acoustic sound dose exposure (dBA * duration) over municipal sensitive zones.
        """
        total_exposure_dose = 0.0
        violations = []

        for zone in noise_sensitive_zones:
            zone_center = (zone["x"], zone["y"], 0.0)
            max_limit_dba = zone.get("max_spl_dba", 55.0)

            for wp in trajectory_waypoints:
                drone_ned = (wp["x"], wp["y"], -wp["alt_m"])
                spl_at_zone = self.compute_ground_spl_footprint(nominal_swl_db, drone_ned, zone_center)

                if spl_at_zone > max_limit_dba:
                    violations.append({
                        "zone_name": zone["name"],
                        "recorded_spl_dba": round(spl_at_zone, 1),
                        "threshold_limit_dba": max_limit_dba,
                        "drone_alt_m": wp["alt_m"]
                    })

                # Acoustic energy accumulation (Leq equivalent metric)
                total_exposure_dose += 10.0 ** (spl_at_zone / 10.0) * wp.get("duration_s", 2.0)

        leq_equivalent_dba = 10.0 * math.log10(max(total_exposure_dose, 1.0))

        return {
            "cumulative_noise_dose_leq_dba": round(leq_equivalent_dba, 2),
            "noise_abatement_violations": len(violations),
            "violation_details": violations,
            "meets_noise_regulations": len(violations) == 0
        }

if __name__ == "__main__":
    acoustic_engine = RotorAeroAcousticModel()
    
    # 1. Calculate Source Acoustic Power under 6.14 kg AUW Cruise
    thrust_n_per_motor = (6.14 * 9.81) / 4.0
    emission = acoustic_engine.compute_acoustic_emission(
        thrust_per_rotor_n=thrust_n_per_motor,
        electrical_power_watts=295.0,
        tip_speed_ratio=0.38
    )

    print("==========================================================================")
    print("      AERO-ACOUSTIC ROTOR NOISE SOURCE & GROUND FOOTPRINT MODEL           ")
    print("==========================================================================")
    for k, v in emission.items():
        print(f"  • {k:30s}: {v}")

    # 2. Evaluate Ground SPL Footprint at Varying Flight Altitudes
    swl = emission["sound_power_level_swl_db"]
    print("\nGround Sound Pressure Level (SPL) vs Flight Altitude (Directly Overhead):")
    for altitude in [15.0, 25.0, 35.0, 50.0, 75.0]:
        spl = acoustic_engine.compute_ground_spl_footprint(
            swl_db=swl,
            drone_pos_ned=(0.0, 0.0, -altitude),
            ground_receiver_ned=(0.0, 0.0, 0.0)
        )
        print(f"  • Altitude {altitude:4.1f} m AGL : Ground Noise = {spl:.2f} dBA")

    # 3. Check Sensitive Quiet Zone (Hospital / School Threshold: 55 dBA)
    zones = [{"name": "Central District Hospital Quiet Zone", "x": 100.0, "y": 100.0, "max_spl_dba": 55.0}]
    waypoints = [
        {"x": 100.0, "y": 100.0, "alt_m": 25.0, "duration_s": 5.0}, # Low pass
        {"x": 100.0, "y": 100.0, "alt_m": 45.0, "duration_s": 5.0}  # Noise-abated high pass
    ]
    audit = acoustic_engine.evaluate_route_noise_penalty(waypoints, zones, nominal_swl_db=swl)
    print(f"\nNoise Abatement Zone Compliance: {'PASSED' if audit['meets_noise_regulations'] else 'VIOLATION DETECTED'}")
    print("==========================================================================\n")