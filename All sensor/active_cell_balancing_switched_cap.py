import math
import numpy as np
from typing import Dict, List, Tuple

class SwitchedCapacitorActiveBalancer:
    def __init__(
        self,
        num_cells: int = 6,
        nominal_cell_capacity_ah: float = 22.0,
        flying_capacitor_uf: float = 47.0,     # 47 uF resonant MLCC capacitor
        switching_frequency_hz: float = 120000.0, # 120 kHz zero-current switching (ZCS)
        mosfet_on_resistance_ohms: float = 0.008,
        resonant_transfer_efficiency: float = 0.94
    ):
        self.n_cells = num_cells
        self.q_nom_as = nominal_cell_capacity_ah * 3600.0 # Ampere-seconds
        self.c_fly = flying_capacitor_uf * 1e-6
        self.f_sw = switching_frequency_hz
        self.r_on = mosfet_on_resistance_ohms
        self.eta = resonant_transfer_efficiency

        # State Variables: SOC array (0.0 to 1.0) for each of the 6 cells
        self.soc_cells = np.array([0.92, 0.915, 0.885, 0.925, 0.905, 0.890])

    def ocv_from_soc(self, soc: float) -> float:
        """Empirical OCV(SOC) curve for LiNiMnCoO2 chemistry."""
        soc_clamped = np.clip(soc, 0.0, 1.0)
        v_ocv = 3.42 + 0.70 * soc_clamped + 0.12 * (soc_clamped ** 2) - 0.08 * math.exp(-25.0 * soc_clamped)
        return float(v_ocv)

    def step_active_balancing_cycle(
        self,
        dt_seconds: float,
        pack_load_current_a: float
    ) -> Dict:
        """
        Calculates bidirectional charge redistribution between adjacent and global cell pairs.
        """
        # 1. Compute individual cell voltages
        cell_voltages = np.array([self.ocv_from_soc(s) for s in self.soc_cells])
        v_mean = float(np.mean(cell_voltages))
        v_max = float(np.max(cell_voltages))
        v_min = float(np.min(cell_voltages))
        delta_v = v_max - v_min

        # 2. Charge Transfer Rates
        # I_transfer = f_sw * C_fly * (V_i - V_j) * eta
        balancing_currents = np.zeros(self.n_cells)
        total_transfer_power_watts = 0.0

        for i in range(self.n_cells):
            v_diff = cell_voltages[i] - v_mean
            # If cell voltage is above mean, export current; if below, import current
            i_shuttle = self.f_sw * self.c_fly * v_diff * self.eta
            balancing_currents[i] = -i_shuttle # Negative means losing charge
            total_transfer_power_watts += abs(i_shuttle * v_diff)

        # 3. Update SOC Differential Equations: dSOC/dt = (-I_load + I_bal) / Q_nom
        d_soc_dt = (-pack_load_current_a + balancing_currents) / self.q_nom_as
        self.soc_cells = np.clip(self.soc_cells + d_soc_dt * dt_seconds, 0.0, 1.0)

        # Updated voltages post-transfer
        new_voltages = np.array([self.ocv_from_soc(s) for s in self.soc_cells])
        new_delta_v = float(np.max(new_voltages) - np.min(new_voltages))

        return {
            "initial_cell_imbalance_mv": round(delta_v * 1000.0, 2),
            "final_cell_imbalance_mv": round(new_delta_v * 1000.0, 2),
            "cell_voltages_v": [round(float(v), 3) for v in new_voltages],
            "max_balancing_current_a": round(float(np.max(np.abs(balancing_currents))), 3),
            "active_shuttling_power_w": round(total_transfer_power_watts, 3),
            "pack_load_current_a": round(pack_load_current_a, 2),
            "balancing_active": delta_v > 0.015,
            "cell_balancing_status": "OPTIMAL_VOLTAGE_EQUALIZATION" if new_delta_v <= 0.015 else "ACTIVE_CHARGE_SHUTTLING"
        }

if __name__ == "__main__":
    balancer = SwitchedCapacitorActiveBalancer()

    print("==========================================================================")
    print("      SWITCHED-CAPACITOR ACTIVE BATTERY CELL BALANCING AUDIT              ")
    print("==========================================================================")
    
    # Simulate 30 seconds of high-power climb with initial 40 mV cell imbalance
    print("Initial Cell SOCs: ", [f"{s*100:.1f}%" for s in balancer.soc_cells])
    for step in range(1, 7):
        # 5-second integration intervals under 18.5A cruise load
        audit = balancer.step_active_balancing_cycle(dt_seconds=5.0, pack_load_current_a=18.5)
        print(f"T+{step*5:02d}s | Imbalance: {audit['final_cell_imbalance_mv']:5.1f} mV | Max Bal Current: {audit['max_balancing_current_a']:5.3f} A | "
              f"Voltages: {audit['cell_voltages_v']} | Status: {audit['cell_balancing_status']}")
    print("==========================================================================\n")