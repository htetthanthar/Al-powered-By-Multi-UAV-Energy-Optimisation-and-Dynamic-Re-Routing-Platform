import math
import numpy as np
from typing import Dict, List, Tuple

class ElectromagneticInterferenceAuditor:
    def __init__(
        self,
        gnss_carrier_freq_hz: float = 1575.42e6,  # GPS L1
        cellular_5g_freq_hz: float = 3500.0e6,    # 5G n78 band
        foc_switching_freq_hz: float = 24000.0,   # 24 kHz inverter switching
        max_acceptable_cn0_drop_db: float = 2.5,
        antenna_loop_area_m2: float = 0.00045     # 4.5 cm^2 GNSS patch antenna effective loop
    ):
        self.f_gnss = gnss_carrier_freq_hz
        self.f_5g = cellular_5g_freq_hz
        self.f_sw = foc_switching_freq_hz
        self.max_cn0_drop = max_acceptable_cn0_drop_db
        self.a_loop = antenna_loop_area_m2

        self.mu0 = 4.0 * math.pi * 1e-7 # H/m
        self.k_boltzmann = 1.380649e-23
        self.t0_kelvin = 290.0

        # Physical separation distances on airframe (meters)
        self.dist_pdb_to_gnss_m = 0.220  # 22 cm mast separation
        self.dist_pdb_to_5g_m = 0.160    # 16 cm separation

    def compute_radiated_b_field_microtesla(
        self,
        bus_current_amps: float,
        current_ripple_ratio: float = 0.08,
        separation_distance_m: float = 0.220
    ) -> float:
        """
        Biot-Savart magnetic near-field calculation for high-current bus bars:
          B = (mu_0 * I_ripple) / (2 * pi * r)
        """
        i_ripple = bus_current_amps * current_ripple_ratio
        r_safe = max(separation_distance_m, 0.02)
        b_field_tesla = (self.mu0 * i_ripple) / (2.0 * math.pi * r_safe)
        return float(b_field_tesla * 1e6) # Return in micro-Tesla (uT)

    def calculate_rf_desense_budget(
        self,
        bus_current_amps: float,
        pwm_rise_time_ns: float = 35.0, # Fast GaN/MOSFET switching edge
        gnss_noise_figure_linear: float = 2.0, # 3 dB Noise Figure
        gnss_bandwidth_hz: float = 2.046e6
    ) -> Dict:
        """
        Calculates harmonic coupling, EMI noise power, and C/N0 desense degradation.
        """
        # 1. Harmonic Frequency Multiplier
        # High-order harmonic amplitude falls as 1/n for square wave switching edges
        harmonic_n = int(self.f_gnss / self.f_sw)
        # Slew rate attenuation factor: sinc(pi * f * t_rise)
        slew_attenuation = abs(math.sin(math.pi * self.f_gnss * (pwm_rise_time_ns * 1e-9)) / 
                               (math.pi * self.f_gnss * (pwm_rise_time_ns * 1e-9) + 1e-9))

        # 2. Magnetic Field at GNSS Mast
        b_ut = self.compute_radiated_b_field_microtesla(
            bus_current_amps, current_ripple_ratio=0.08, separation_distance_m=self.dist_pdb_to_gnss_m
        )
        b_tesla = b_ut * 1e-6

        # 3. Induced RF Noise Voltage and Coupled Power
        # V_ind = A * 2 * pi * f * B * (1/n) * slew_atten
        v_induced = self.a_loop * 2.0 * math.pi * self.f_gnss * b_tesla * (1.0 / harmonic_n) * slew_attenuation
        p_emi_watts = (v_induced ** 2) / 50.0 # 50-ohm RF antenna impedance

        # 4. Thermal Noise Floor: P_noise = k * T * B * NF
        p_thermal_noise = self.k_boltzmann * self.t0_kelvin * gnss_bandwidth_hz * gnss_noise_figure_linear

        # 5. Receiver Desense & Carrier-to-Noise Ratio (C/N0) Drop
        desense_factor = 1.0 + (p_emi_watts / p_thermal_noise)
        cn0_degradation_db = 10.0 * math.log10(max(desense_factor, 1.0))

        # Dynamic Mitigation Command
        is_desense_critical = cn0_degradation_db > self.max_cn0_drop
        recommended_pwm_slew_ns = pwm_rise_time_ns if not is_desense_critical else (pwm_rise_time_ns * 2.5)

        return {
            "bus_current_amps": round(bus_current_amps, 1),
            "gnss_antenna_b_field_ut": round(b_ut, 3),
            "nearest_gnss_harmonic_order": harmonic_n,
            "coupled_emi_power_dbm": round(10.0 * math.log10(max(p_emi_watts * 1000.0, 1e-15)), 2),
            "thermal_noise_floor_dbm": round(10.0 * math.log10(p_thermal_noise * 1000.0), 2),
            "gnss_cn0_degradation_db": round(cn0_degradation_db, 2),
            "max_allowable_cn0_drop_db": self.max_cn0_drop,
            "rf_desense_status": "CRITICAL_GNSS_LINK_MARGIN_DEGRADED" if is_desense_critical else "RF_EMISSION_COMPLIANT",
            "active_mitigation_command": "INJECT_FOC_PWM_SLEW_RATE_BACKOFF" if is_desense_critical else "MAINTAIN_NOMINAL_SWITCHING"
        }

if __name__ == "__main__":
    auditor = ElectromagneticInterferenceAuditor()

    print("==========================================================================")
    print("      ELECTROMAGNETIC INTERFERENCE (EMI) & RF DESENSE AUDIT               ")
    print("==========================================================================")
    
    # Evaluate RF Desense during Cruise (35A bus current) vs Maximum Climb (85A bus current)
    current_levels = [15.0, 35.0, 60.0, 85.0]
    for current in current_levels:
        report = auditor.calculate_rf_desense_budget(
            bus_current_amps=current, pwm_rise_time_ns=30.0
        )
        print(f"Bus Current: {report['bus_current_amps']:4.1f}A | B-Field: {report['gnss_antenna_b_field_ut']:5.3f} uT | "
              f"EMI Power: {report['coupled_emi_power_dbm']:6.1f} dBm | C/N0 Loss: {report['gnss_cn0_degradation_db']:4.2f} dB | "
              f"Status: {report['rf_desense_status']} ({report['active_mitigation_command']})")
    print("==========================================================================\n")