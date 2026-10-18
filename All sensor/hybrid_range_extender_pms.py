import math
import numpy as np
from typing import Dict, Tuple

class HybridRangeExtenderPowerManager:
    def __init__(
        self,
        generator_max_power_w: float = 850.0,
        generator_idle_power_w: float = 120.0,
        battery_capacity_wh: float = 240.0,   # High-rate buffer battery
        target_soc_sustaining_pct: float = 65.0,
        fuel_energy_density_wh_kg: float = 11900.0 # Jet-A1 fuel
    ):
        self.p_gen_max = generator_max_power_w
        self.p_gen_idle = generator_idle_power_w
        self.e_batt_wh = battery_capacity_wh
        self.soc_target = target_soc_sustaining_pct
        self.h_fuel = fuel_energy_density_wh_kg

        # BSFC Polynomial: Fuel Flow (g/s) = a0 + a1*P + a2*P^2
        self.alpha0 = 0.045  # Idle consumption (g/s)
        self.alpha1 = 0.00032
        self.alpha2 = 1.2e-7

        # Battery Pack (6S buffer: 22.2V nominal, 10.8 Ah)
        self.q_batt_as = (self.e_batt_wh / 22.2) * 3600.0

    def compute_fuel_burn_rate_g_s(self, p_gen_watts: float) -> float:
        """Calculates instantaneous turbine fuel burn rate."""
        if p_gen_watts <= 5.0:
            return 0.0 # Generator shutdown
        p = np.clip(p_gen_watts, self.p_gen_idle, self.p_gen_max)
        flow_g_s = self.alpha0 + self.alpha1 * p + self.alpha2 * (p ** 2)
        return float(flow_g_s)

    def optimize_power_split_pmp(
        self,
        bus_demand_power_watts: float,
        current_soc_pct: float,
        co_state_lambda: float = 0.85
    ) -> Dict:
        """
        Solves instantaneous Pontryagin's Minimum Principle Hamiltonian optimization.
        """
        p_demand = bus_demand_power_watts
        soc = current_soc_pct

        # Dynamic co-state adjustment based on SOC deviation
        delta_soc = (self.soc_target - soc) / 100.0
        lambda_effective = co_state_lambda * (1.0 + 3.5 * delta_soc)

        # Discretize search space across generator operating range
        p_gen_candidates = np.linspace(0.0, self.p_gen_max, 50)
        best_hamiltonian = float('inf')
        best_p_gen = 0.0

        v_ocv = 22.2 + 2.5 * (soc / 100.0) # Estimated pack voltage

        for p_gen in p_gen_candidates:
            if 0.0 < p_gen < self.p_gen_idle:
                continue # Below minimum operating idle

            p_batt = p_demand - p_gen
            m_dot_fuel = self.compute_fuel_burn_rate_g_s(p_gen)

            # Battery SOC rate of change: dSOC/dt = -P_batt / (V * Q)
            d_soc_dt = -p_batt / (v_ocv * self.q_batt_as)

            # Hamiltonian H = Fuel_Rate - lambda * dSOC/dt
            hamiltonian = m_dot_fuel - lambda_effective * (d_soc_dt * 1e5)

            if hamiltonian < best_hamiltonian:
                best_hamiltonian = hamiltonian
                best_p_gen = p_gen

        p_batt_opt = p_demand - best_p_gen
        fuel_flow = self.compute_fuel_burn_rate_g_s(best_p_gen)

        # Generator thermal state efficiency
        thermal_eff_pct = (best_p_gen / max(fuel_flow * (self.h_fuel * 3.6), 1.0)) * 100.0 if best_p_gen > 0 else 0.0

        return {
            "bus_power_demand_watts": round(p_demand, 1),
            "optimal_generator_power_w": round(best_p_gen, 1),
            "optimal_battery_power_w": round(p_batt_opt, 1),
            "battery_charging_active": p_batt_opt < 0.0,
            "turbine_fuel_consumption_g_s": round(fuel_flow, 4),
            "generator_thermal_efficiency_pct": round(thermal_eff_pct, 1),
            "battery_state_of_charge_pct": round(soc, 2),
            "hybrid_mode": "CHARGE_SUSTAINING_OPTIMAL" if abs(delta_soc) < 0.05 else ("RAPID_BATTERY_RECHARGE" if delta_soc > 0 else "PEAK_SHAVING_DISCHARGE")
        }

if __name__ == "__main__":
    pms = HybridRangeExtenderPowerManager()

    print("==========================================================================")
    print("      HYBRID RANGE-EXTENDER TURBINE/BATTERY OPTIMAL POWER SPLIT AUDIT    ")
    print("==========================================================================")

    # 1. Cruise Transit (P_bus = 340 W, SOC = 64% -> generator sustains load)
    report_cruise = pms.optimize_power_split_pmp(bus_demand_power_watts=340.0, current_soc_pct=64.0)
    print("Cruise Transit State (340 W Demand):")
    for k, v in report_cruise.items():
        print(f"  • {k:35s}: {v}")

    # 2. Maximum Climb Surge (P_bus = 980 W, SOC = 55% -> combined peak boost)
    report_climb = pms.optimize_power_split_pmp(bus_demand_power_watts=980.0, current_soc_pct=55.0)
    print("\nMaximum Climb Surge State (980 W Demand):")
    for k, v in report_climb.items():
        print(f"  • {k:35s}: {v}")
    print("==========================================================================\n")