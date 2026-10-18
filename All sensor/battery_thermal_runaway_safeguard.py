import math
import numpy as np
from typing import Dict, Tuple

class BatteryThermalRunawaySafeguard:
    def __init__(
        self,
        cell_mass_kg: float = 0.485,             # 22Ah individual pouch cell
        cell_heat_capacity_j_kg_k: float = 1050.0,
        cell_surface_area_m2: float = 0.042,
        convective_heat_transfer_h: float = 35.0, # Forced rotor downwash cooling
        pyro_isolation_temp_threshold_c: float = 78.0,
        venting_pressure_limit_kpa: float = 280.0
    ):
        self.m_cell = cell_mass_kg
        self.cp = cell_heat_capacity_j_kg_k
        self.area = cell_surface_area_m2
        self.h_conv = convective_heat_transfer_h
        self.t_cutoff = pyro_isolation_temp_threshold_c
        self.p_vent_limit = venting_pressure_limit_kpa

        # Universal Gas Constant
        self.R_gas = 8.31446 # J/(mol*K)

        # Kinetic Parameters (Arrhenius constants: A [1/s], Ea [J/mol], H [J/kg])
        # 1. SEI Layer Breakdown (onset ~ 80°C - 120°C)
        self.a_sei = 1.667e15
        self.ea_sei = 1.35e5
        self.h_sei = 2.57e5

        # 2. Anode-Solvent Reaction (onset ~ 120°C - 140°C)
        self.a_anode = 2.50e13
        self.ea_anode = 1.40e5
        self.h_anode = 1.71e6

        # 3. Cathode Active Oxygen Decomposition (onset ~ 160°C - 220°C)
        self.a_cathode = 6.67e13
        self.ea_cathode = 1.45e5
        self.h_cathode = 3.14e5

        # State Variables: [T_cell_K, c_sei, c_anode, c_cathode, internal_gas_moles]
        self.T_k = 308.15 # Initial 35°C
        self.c_sei = 1.0
        self.c_anode = 1.0
        self.c_cathode = 1.0
        self.n_gas = 0.005 # 5 mmol baseline headspace gas
        self.cell_void_volume_m3 = 3.5e-5 # 35 cm^3 pouch headspace

    def evaluate_thermal_runaway_step(
        self,
        dt_seconds: float,
        joule_heating_power_watts: float,
        ambient_temp_c: float = 32.0
    ) -> Dict:
        """
        Integrates coupled Arrhenius chemical kinetics and thermal energy conservation equations.
        """
        t_amb_k = ambient_temp_c + 273.15

        # 1. Reaction Rates (Arrhenius Equations)
        r_sei = self.a_sei * self.c_sei * math.exp(-self.ea_sei / (self.R_gas * self.T_k))
        r_anode = self.a_anode * self.c_anode * math.exp(-self.ea_anode / (self.R_gas * self.T_k))
        r_cathode = self.a_cathode * self.c_cathode * math.exp(-self.ea_cathode / (self.R_gas * self.T_k))

        # 2. Chemical Exothermic Heat Generation Rates (Watts)
        q_sei = self.h_sei * self.m_cell * r_sei
        q_anode = self.h_anode * self.m_cell * r_anode
        q_cathode = self.h_cathode * self.m_cell * r_cathode
        q_chemical_total = q_sei + q_anode + q_cathode

        # 3. Convective Heat Dissipation
        q_cooling = self.h_conv * self.area * (self.T_k - t_amb_k)

        # 4. Energy Conservation: m * cp * (dT/dt) = Q_joule + Q_chem - Q_cooling
        q_net = joule_heating_power_watts + q_chemical_total - q_cooling
        d_temp_dt = q_net / (self.m_cell * self.cp)

        # 5. Flammable Gas Generation Rate: dn/dt ~ (m_active / M_mol) * (r_anode + r_cathode)
        d_gas_dt = 0.12 * (r_anode + r_cathode) # mol/s
        
        # Euler state update
        self.T_k += d_temp_dt * dt_seconds
        self.c_sei = max(0.0, self.c_sei - r_sei * dt_seconds)
        self.c_anode = max(0.0, self.c_anode - r_anode * dt_seconds)
        self.c_cathode = max(0.0, self.c_cathode - r_cathode * dt_seconds)
        self.n_gas += d_gas_dt * dt_seconds

        # Pouch Internal Pressure via Ideal Gas Law: P = (n * R * T) / V
        p_internal_kpa = ((self.n_gas * self.R_gas * self.T_k) / self.cell_void_volume_m3) / 1000.0
        current_temp_c = self.T_k - 273.15

        # Safety Triggers
        is_venting = p_internal_kpa >= self.p_vent_limit
        is_pyro_isolation_triggered = current_temp_c >= self.t_cutoff

        status = "CELL_ELECTROCHEMICALLY_STABLE"
        if current_temp_c >= 130.0:
            status = "CATASTROPHIC_UNSTOPPABLE_THERMAL_RUNAWAY"
        elif is_pyro_isolation_triggered:
            status = "CRITICAL_PYRO_CONTACTORS_OPENED_BATTERY_ISOLATED"
        elif q_chemical_total > 5.0:
            status = "WARNING_EXOTHERMIC_SEI_BREAKDOWN_ACTIVE"

        return {
            "cell_temperature_c": round(current_temp_c, 2),
            "self_heating_rate_c_per_min": round(d_temp_dt * 60.0, 2),
            "chemical_exothermic_power_w": round(q_chemical_total, 3),
            "joule_heating_power_w": round(joule_heating_power_watts, 2),
            "internal_pouch_pressure_kpa": round(p_internal_kpa, 1),
            "pouch_burst_venting_active": is_venting,
            "pyrotechnic_cutoff_triggered": is_pyro_isolation_triggered,
            "electrochemical_safety_status": status
        }

if __name__ == "__main__":
    safeguard = BatteryThermalRunawaySafeguard()

    print("==========================================================================")
    print("      BATTERY ELECTROCHEMICAL THERMAL RUNAWAY & VENTING AUDIT            ")
    print("==========================================================================")

    # Simulate localized internal short-circuit (55 Watts continuous internal Joule dissipation)
    for step in range(1, 11):
        dt = 5.0 # 5-second steps
        report = safeguard.evaluate_thermal_runaway_step(dt_seconds=dt, joule_heating_power_watts=55.0)
        print(f"T+{step*5:02d}s | Temp: {report['cell_temperature_c']:6.2f}°C | Self-Heating: {report['self_heating_rate_c_per_min']:+6.2f}°C/min | "
              f"P_chem: {report['chemical_exothermic_power_w']:6.2f}W | Pressure: {report['internal_pouch_pressure_kpa']:5.1f} kPa | Status: {report['electrochemical_safety_status']}")
    print("==========================================================================\n")