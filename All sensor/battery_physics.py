import math
import numpy as np

class BatteryElectrochemicalModel:
    def __init__(
        self,
        nominal_capacity_ah: float = 22.0,  # 6S LiPo Pack (22.2V nominal, ~488 Wh)
        nominal_voltage_v: float = 22.2,
        internal_resistance_base_ohm: float = 0.015,
        ambient_temp_c: float = 28.0,
        thermal_mass_j_per_k: float = 850.0,
        cooling_coeff_w_per_k: float = 12.5
    ):
        self.capacity_coulombs = nominal_capacity_ah * 3600.0
        self.nominal_voltage = nominal_voltage_v
        self.r_base = internal_resistance_base_ohm
        self.temp_k = ambient_temp_c + 273.15
        self.ambient_temp_k = ambient_temp_c + 273.15
        self.c_thermal = thermal_mass_j_per_k
        self.h_cooling = cooling_coeff_w_per_k
        
        self.remaining_coulombs = self.capacity_coulombs
        self.cycle_count = 120  # Historical flight cycles
        self.soh = self._calculate_initial_soh()

    def _calculate_initial_soh(self) -> float:
        """Estimates State of Health based on cumulative cycle wear (Arrhenius aging model)."""
        aging_factor = 1.0 - (0.00025 * self.cycle_count)
        return max(0.70, aging_factor)

    def compute_internal_resistance(self) -> float:
        """
        Calculates temperature-dependent internal resistance using Arrhenius relation:
        R_int(T) = R_base * exp(E_a / R * (1/T - 1/T_ref)) / SOH
        """
        e_a_over_r = 1850.0  # Activation energy ratio
        t_ref_k = 298.15     # 25 degrees C reference
        r_temp = self.r_base * math.exp(e_a_over_r * (1.0 / self.temp_k - 1.0 / t_ref_k))
        return (r_temp / self.soh)

    def step(self, power_demand_watts: float, dt_seconds: float) -> dict:
        """
        Advances the battery state over dt_seconds under a specified power draw.
        """
        r_int = self.compute_internal_resistance()
        
        # Open Circuit Voltage (OCV) curve approximation as a function of SOC
        soc = self.remaining_coulombs / (self.capacity_coulombs * self.soh)
        ocv = 18.0 + (self.nominal_voltage - 18.0) * (0.85 * soc + 0.15) + (1.2 * math.log(max(soc, 1e-4) + 0.05))
        
        # Solve terminal current: P = V_term * I = (OCV - I * R_int) * I  =>  R_int * I^2 - OCV * I + P = 0
        discriminant = max(0.0, ocv**2 - 4 * r_int * power_demand_watts)
        current_amps = (ocv - math.sqrt(discriminant)) / (2 * r_int) if r_int > 0 else power_demand_watts / ocv
        terminal_voltage = ocv - current_amps * r_int

        # Coulomb counting
        coulombs_discharged = current_amps * dt_seconds
        self.remaining_coulombs = max(0.0, self.remaining_coulombs - coulombs_discharged)
        updated_soc = self.remaining_coulombs / (self.capacity_coulombs * self.soh)

        # Thermal dynamics: Q_gen = I^2 * R_int, Q_diss = h * (T_batt - T_amb)
        q_gen = (current_amps ** 2) * r_int
        q_diss = self.h_cooling * (self.temp_k - self.ambient_temp_k)
        delta_temp_k = ((q_gen - q_diss) / self.c_thermal) * dt_seconds
        self.temp_k += delta_temp_k

        return {
            "soc_pct": round(updated_soc * 100.0, 2),
            "terminal_voltage_v": round(terminal_voltage, 2),
            "current_amps": round(current_amps, 2),
            "internal_resistance_m_ohm": round(r_int * 1000.0, 3),
            "temperature_c": round(self.temp_k - 273.15, 2),
            "heat_generated_w": round(q_gen, 2),
            "soh_pct": round(self.soh * 100.0, 2)
        }

if __name__ == "__main__":
    battery = BatteryElectrochemicalModel(ambient_temp_c=32.0)
    print("========================================================")
    print("       ELECTROCHEMICAL BATTERY DISCHARGE PROFILE        ")
    print("========================================================")
    print(f"Initial SOH: {battery.soh * 100:.1f}% | Ambient Temp: 32.0°C")
    
    for minute in range(1, 11):
        # 10 minutes under 320 W flight cruise power draw
        telemetry = battery.step(power_demand_watts=320.0, dt_seconds=60.0)
        print(f"Minute {minute:02d}: SOC = {telemetry['soc_pct']:5.1f}% | Voltage = {telemetry['terminal_voltage_v']:4.1f}V | "
              f"Current = {telemetry['current_amps']:4.1f}A | Temp = {telemetry['temperature_c']:4.1f}°C | "
              f"R_int = {telemetry['internal_resistance_m_ohm']:.2f} mΩ")
    print("========================================================\n")