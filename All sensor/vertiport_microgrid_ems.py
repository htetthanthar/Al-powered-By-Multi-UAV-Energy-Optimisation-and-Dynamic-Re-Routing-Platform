import math
import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Tuple

class VertiportMicrogridEMS:
    def __init__(
        self,
        transformer_capacity_kw: float = 75.0,
        bess_capacity_kwh: float = 120.0,
        bess_max_charge_kw: float = 30.0,
        bess_max_discharge_kw: float = 30.0,
        bess_efficiency: float = 0.92,
        charger_efficiency: float = 0.94
    ):
        self.p_trans_max = transformer_capacity_kw
        self.bess_cap = bess_capacity_kwh
        self.p_bess_ch_max = bess_max_charge_kw
        self.p_bess_dis_max = bess_max_discharge_kw
        self.eta_bess = bess_efficiency
        self.eta_ch = charger_efficiency

    def optimize_charging_schedule(
        self,
        time_horizon_hours: int,
        dt_hours: float,
        solar_pv_forecast_kw: np.ndarray,
        grid_tou_tariff_per_kwh: np.ndarray,
        drone_demand_kwh: np.ndarray,
        initial_bess_soc_pct: float = 50.0
    ) -> Dict:
        """
        Solves LP for optimal power flows over horizon T:
          min sum_t ( Tariff(t) * P_grid(t) * dt )
          s.t.
            P_grid(t) + P_pv(t) + P_bess_dis(t) >= P_drone(t)/eta_ch + P_bess_ch(t)
            P_grid(t) <= P_trans_max
            BESS SOC dynamics and boundary capacity limits
        """
        T = time_horizon_hours
        # Decision variables per timestep: [P_grid, P_bess_ch, P_bess_dis] => Total 3*T variables
        num_vars = 3 * T

        # Cost vector c: Only P_grid incurs purchase cost
        c = np.zeros(num_vars)
        for t in range(T):
            c[3 * t] = grid_tou_tariff_per_kwh[t] * dt_hours

        # Inequality constraints: A_ub * x <= b_ub
        a_ub = []
        b_ub = []

        # 1. Power Balance: -P_grid - P_bess_dis + P_bess_ch <= P_pv - (P_drone / eta_ch)
        for t in range(T):
            row = np.zeros(num_vars)
            row[3 * t] = -1.0                    # -P_grid
            row[3 * t + 1] = 1.0                 # +P_bess_ch
            row[3 * t + 2] = -1.0                # -P_bess_dis
            a_ub.append(row)
            
            p_drone_load = drone_demand_kwh[t] / dt_hours
            b_ub.append(solar_pv_forecast_kw[t] - (p_drone_load / self.eta_ch))

        # 2. Transformer Ingress Capacity: P_grid(t) <= P_trans_max
        for t in range(T):
            row = np.zeros(num_vars)
            row[3 * t] = 1.0
            a_ub.append(row)
            b_ub.append(self.p_trans_max)

        # 3. BESS Maximum Capacity Constraint:
        # E_bess(t) = E_0 + sum_{tau=0}^t [ eta * P_ch(tau) - (1/eta) * P_dis(tau) ] * dt <= E_max
        e_init = self.bess_cap * (initial_bess_soc_pct / 100.0)
        
        for t in range(T):
            # Upper capacity bound
            row_upper = np.zeros(num_vars)
            for tau in range(t + 1):
                row_upper[3 * tau + 1] = self.eta_bess * dt_hours
                row_upper[3 * tau + 2] = -(1.0 / self.eta_bess) * dt_hours
            a_ub.append(row_upper)
            b_ub.append(self.bess_cap * 0.95 - e_init) # 95% max SOC

            # Lower capacity bound: -E_bess(t) <= -E_min
            row_lower = -row_upper
            a_ub.append(row_lower)
            b_ub.append(e_init - self.bess_cap * 0.15) # 15% min reserve

        # Variable Bounds: [0, P_max]
        bounds = []
        for t in range(T):
            bounds.append((0.0, self.p_trans_max))      # P_grid
            bounds.append((0.0, self.p_bess_ch_max))    # P_bess_ch
            bounds.append((0.0, self.p_bess_dis_max))   # P_bess_dis

        # Solve Linear Program
        res = linprog(c, A_ub=np.array(a_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")

        if not res.success:
            return {"status": "OPTIMIZATION_FAILED", "message": res.message}

        x_opt = res.x
        grid_power = x_opt[0::3]
        bess_ch = x_opt[1::3]
        bess_dis = x_opt[2::3]

        # Calculate BESS SOC profile
        bess_energy = [e_init]
        for t in range(T):
            delta_e = (self.eta_bess * bess_ch[t] - (1.0 / self.eta_bess) * bess_dis[t]) * dt_hours
            bess_energy.append(bess_energy[-1] + delta_e)

        soc_profile = [(e / self.bess_cap) * 100.0 for e in bess_energy[1:]]

        return {
            "status": "OPTIMAL_SCHEDULE_SOLVED",
            "total_electricity_cost_usd": round(float(res.fun), 2),
            "peak_grid_demand_kw": round(float(np.max(grid_power)), 2),
            "solar_energy_utilized_kwh": round(float(np.sum(solar_pv_forecast_kw) * dt_hours), 2),
            "hourly_grid_power_kw": [round(float(p), 1) for p in grid_power],
            "hourly_bess_net_power_kw": [round(float(bess_dis[t] - bess_ch[t]), 1) for t in range(T)],
            "final_bess_soc_pct": round(soc_profile[-1], 1)
        }

if __name__ == "__main__":
    ems = VertiportMicrogridEMS()

    # 12-hour operational profile (08:00 to 20:00)
    T = 12
    dt = 1.0
    solar_pv = np.array([5.0, 15.0, 28.0, 42.0, 48.0, 45.0, 38.0, 25.0, 12.0, 4.0, 0.0, 0.0])
    # Peak TOU tariff ($0.38/kWh) between 12:00 and 17:00 (indices 4..9)
    tou_tariff = np.array([0.18, 0.18, 0.18, 0.22, 0.38, 0.38, 0.38, 0.38, 0.38, 0.22, 0.18, 0.18])
    # Continuous fleet battery swap demand (kWh required per hour)
    drone_demand = np.array([12.0, 18.0, 24.0, 32.0, 35.0, 38.0, 35.0, 30.0, 28.0, 22.0, 16.0, 10.0])

    schedule = ems.optimize_charging_schedule(
        time_horizon_hours=T,
        dt_hours=dt,
        solar_pv_forecast_kw=solar_pv,
        grid_tou_tariff_per_kwh=tou_tariff,
        drone_demand_kwh=drone_demand,
        initial_bess_soc_pct=60.0
    )

    print("==========================================================================")
    print("      VERTIPORT MICROGRID & BESS OPTIMAL DISPATCH BENCHMARK               ")
    print("==========================================================================")
    for k, v in schedule.items():
        if not isinstance(v, list):
            print(f"  • {k:32s}: {v}")
    print("\nHourly Energy Dispatch Vector:")
    for h in range(T):
        print(f"  Hour {h+8:02d}:00 | Grid: {schedule['hourly_grid_power_kw'][h]:4.1f} kW | "
              f"BESS Net: {schedule['hourly_bess_net_power_kw'][h]:+5.1f} kW | PV: {solar_pv[h]:4.1f} kW | Tariff: ${tou_tariff[h]:.2f}")
    print("==========================================================================\n")