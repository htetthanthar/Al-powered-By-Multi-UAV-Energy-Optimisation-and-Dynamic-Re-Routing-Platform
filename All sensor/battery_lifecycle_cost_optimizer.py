import math
import numpy as np
from typing import Dict, List, Tuple

class BatteryLifeCycleCostValuator:
    def __init__(
        self,
        pack_replacement_cost_usd: float = 650.0, # 6S 22Ah High-Discharge LiPo
        nominal_capacity_ah: float = 22.0,
        eol_soh_threshold: float = 0.80,          # 20% capacity loss threshold
        activation_energy_ea_over_r: float = 2850.0 # Arrhenius thermal degradation constant
    ):
        self.cost_pack = pack_replacement_cost_usd
        self.c_nom = nominal_capacity_ah
        self.soh_eol = eol_soh_threshold
        self.e_over_r = activation_energy_ea_over_r
        self.usable_capacity_loss_budget = 1.0 - eol_soh_threshold # 0.20

    def extract_rainflow_cycles(self, soc_history_pct: np.ndarray) -> List[Tuple[float, float]]:
        """
        ASTM E1049-85 Rainflow Cycle Counting algorithm on battery Depth-of-Discharge (DoD) series.
        Returns list of (range_dod_pct, mean_soc_pct).
        """
        # Simplify signal to local turning points (extrema)
        extrema = [soc_history_pct[0]]
        for i in range(1, len(soc_history_pct) - 1):
            d1 = soc_history_pct[i] - soc_history_pct[i - 1]
            d2 = soc_history_pct[i + 1] - soc_history_pct[i]
            if d1 * d2 <= 0.0:
                extrema.append(soc_history_pct[i])
        extrema.append(soc_history_pct[-1])

        # Rainflow stack pairing
        stack = []
        cycles = []

        for p in extrema:
            stack.append(p)
            while len(stack) >= 3:
                s0, s1, s2 = stack[-3], stack[-2], stack[-1]
                range1 = abs(s1 - s0)
                range2 = abs(s2 - s1)

                if range1 <= range2:
                    mean_val = (s0 + s1) / 2.0
                    cycles.append((range1, mean_val))
                    stack.pop(-2)
                    stack.pop(-2)
                else:
                    break

        # Remaining half-cycles in stack
        while len(stack) >= 2:
            p1 = stack.pop(0)
            p2 = stack[0]
            cycles.append((abs(p2 - p1), (p1 + p2) / 2.0))

        return cycles

    def compute_mission_degradation_cost(
        self,
        soc_profile_pct: np.ndarray,
        mean_battery_temp_c: float,
        mean_discharge_current_amps: float,
        mission_flight_distance_km: float
    ) -> Dict:
        """
        Calculates cumulative electrochemical capacity degradation and financial amortization.
        """
        temp_k = mean_battery_temp_c + 273.15
        t_ref_k = 298.15 # 25°C reference
        
        # Arrhenius thermal acceleration factor
        arrhenius_factor = math.exp(self.e_over_r * (1.0 / t_ref_k - 1.0 / temp_k))

        # C-rate mechanical stress factor: (I / I_ref)^0.65
        c_rate = mean_discharge_current_amps / self.c_nom
        c_stress = (max(c_rate, 0.2)) ** 0.65

        # Rainflow decomposition
        cycles = self.extract_rainflow_cycles(soc_profile_pct)
        total_fractional_loss = 0.0

        for dod_range_pct, mean_soc in cycles:
            dod_fraction = (dod_range_pct / 100.0)
            # Semi-empirical power law: Loss ~ (DoD)^1.45
            cycle_loss = 0.00012 * (dod_fraction ** 1.45) * arrhenius_factor * c_stress
            total_fractional_loss += cycle_loss

        # Financial cost calculation
        consumed_soh_fraction = total_fractional_loss / self.usable_capacity_loss_budget
        degradation_cost_usd = consumed_soh_fraction * self.cost_pack
        cost_per_km = degradation_cost_usd / max(mission_flight_distance_km, 0.1)

        projected_lifetime_missions = int(1.0 / max(consumed_soh_fraction, 1e-6))

        return {
            "mission_distance_km": round(mission_flight_distance_km, 2),
            "pack_temperature_c": round(mean_battery_temp_c, 1),
            "rainflow_subcycles_extracted": len(cycles),
            "soh_consumed_this_mission_pct": round(total_fractional_loss * 100.0, 5),
            "marginal_battery_wear_cost_usd": round(degradation_cost_usd, 3),
            "battery_amortization_cost_per_km_usd": round(cost_per_km, 4),
            "projected_pack_mission_lifetime": projected_lifetime_missions,
            "economic_health_status": "OPTIMAL_LIFECYCLE_EXPENSE" if cost_per_km < 0.05 else "HIGH_THERMAL_WEAR_PENALTY"
        }

if __name__ == "__main__":
    valuator = BatteryLifeCycleCostValuator(pack_replacement_cost_usd=650.0)

    # Simulated flight: SOC drops from 98% down to 24% with minor regenerative updraft fluctuations
    time_series_soc = np.array([
        98.0, 92.0, 85.0, 78.0, 80.0, # Minor thermal updraft charge
        72.0, 65.0, 54.0, 42.0, 31.0, 24.0
    ])

    audit = valuator.compute_mission_degradation_cost(
        soc_profile_pct=time_series_soc,
        mean_battery_temp_c=36.5,
        mean_discharge_current_amps=14.8,
        mission_flight_distance_km=28.4
    )

    print("==========================================================================")
    print("      RAINFLOW BATTERY WEAR & LIFE-CYCLE FINANCIAL VALUATION AUDIT        ")
    print("==========================================================================")
    for k, v in audit.items():
        print(f"  • {k:40s}: {v}")
    print("==========================================================================\n")