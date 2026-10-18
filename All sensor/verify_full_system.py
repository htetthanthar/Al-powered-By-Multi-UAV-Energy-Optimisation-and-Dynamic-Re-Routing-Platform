import sys
import time
import torch
import numpy as np
import pandas as pd

# Core Subsystem Imports
from dynamic_planner import EnergySurrogateMLP, process_order_batch, evaluate_route_feasibility, optimize_stop_sequence
from train_weather import WeatherLSTM
from spatial_geometry import AirspaceGeometryResolver
from battery_physics import BatteryElectrochemicalModel
from state_machine import MissionExecutionFSM, UAVState
from telemetry_logger import MissionDataLake

def execute_system_smoke_test():
    print("==========================================================================")
    print("          AUTONOMOUS UAV ENGINE: END-TO-END PIPELINE SMOKE TEST          ")
    print("==========================================================================")
    checks = []

    # 1. Verify Neural Model Checkpoints & Inferences
    try:
        energy_model = EnergySurrogateMLP()
        energy_model.eval()
        dummy_feat = torch.tensor([[18.05, 1.85, 3.5, 0.52, 0.0]], dtype=torch.float32)
        power_out = energy_model(dummy_feat).item()
        assert 150.0 <= power_out <= 450.0
        checks.append(("Neural Energy Surrogate (MLP)", "PASSED", f"Power: {power_out:.1f} W"))
    except Exception as e:
        checks.append(("Neural Energy Surrogate (MLP)", "FAILED", str(e)))

    try:
        weather_model = WeatherLSTM()
        weather_model.eval()
        dummy_seq = torch.randn(1, 12, 5)
        wind_pred, rain_pred = weather_model(dummy_seq)
        assert wind_pred.shape == (1, 2)
        checks.append(("Weather Forecaster (Bi-LSTM)", "PASSED", f"Wind UV: {wind_pred.detach().numpy()[0]}"))
    except Exception as e:
        checks.append(("Weather Forecaster (Bi-LSTM)", "FAILED", str(e)))

    # 2. Verify Airspace Triage & Visibility Geometry
    try:
        geo_resolver = AirspaceGeometryResolver()
        p_start, p_end = (103.86, 1.32), (103.95, 1.37)
        intersects, zone_name = geo_resolver.check_trajectory_conflict(p_start, p_end)
        detour = geo_resolver.compute_tangent_detour(p_start, p_end)
        assert intersects is True
        assert len(detour) >= 3
        checks.append(("Geofence & Visibility Detour", "PASSED", f"Detour nodes: {len(detour)}"))
    except Exception as e:
        checks.append(("Geofence & Visibility Detour", "FAILED", str(e)))

    # 3. Verify Combinatorial Routing & Constraint Feasibility
    try:
        depot = {"lat": 1.3000, "lon": 103.8000, "name": "Depot"}
        orders = [
            {"id": "SMK-01", "lat": 1.3840, "lon": 103.7470, "weight_kg": 0.65},
            {"id": "SMK-02", "lat": 1.4304, "lon": 103.8354, "weight_kg": 0.70},
            {"id": "SMK-03", "lat": 1.3532, "lon": 103.9447, "weight_kg": 0.50}
        ]
        opt_route, eta, energy = optimize_stop_sequence(orders, depot, energy_model, wind_speed_ms=3.0, wind_angle_rad=0.0)
        assert eta <= 55.0
        assert sum(o["weight_kg"] for o in orders) <= 2.0
        checks.append(("Graph Optimization (GNN/2-Opt)", "PASSED", f"ETA: {eta:.2f} min (Energy: {energy:.1f} Wh)"))
    except Exception as e:
        checks.append(("Graph Optimization (GNN/2-Opt)", "FAILED", str(e)))

    # 4. Verify Electrochemical Battery State Estimator
    try:
        battery = BatteryElectrochemicalModel()
        batt_state = battery.step(power_demand_watts=300.0, dt_seconds=60.0)
        assert 95.0 <= batt_state["soc_pct"] <= 100.0
        checks.append(("Electrochemical Battery ECM", "PASSED", f"SOC: {batt_state['soc_pct']}% (V: {batt_state['terminal_voltage_v']}V)"))
    except Exception as e:
        checks.append(("Electrochemical Battery ECM", "FAILED", str(e)))

    # 5. Verify In-Flight State Machine & Data Lake
    try:
        fsm = MissionExecutionFSM("SMK-MSN", total_stops=3)
        state_nominal = fsm.process_telemetry_frame({"wind_kmh": 12.0, "rain_flag": False, "battery_soc": 98.0, "alt_m": 25.0})
        fsm.current_state = UAVState.EN_ROUTE
        state_shock = fsm.process_telemetry_frame({"wind_kmh": 34.0, "rain_flag": True, "battery_soc": 90.0, "alt_m": 25.0})
        assert state_shock == UAVState.HOVER_HOLD
        checks.append(("Finite State Machine (FSM)", "PASSED", f"Transitioned -> {state_shock.value}"))
    except Exception as e:
        checks.append(("Finite State Machine (FSM)", "FAILED", str(e)))

    try:
        lake = MissionDataLake("data/smoke_test_lake.duckdb")
        lake.log_mission("SMK-MSN", 3, 1.85, eta, eta, energy, 14.5, "SUCCESS")
        kpi = lake.query_mission_kpis()
        assert len(kpi) == 1
        checks.append(("DuckDB Analytical Lake", "PASSED", f"Recorded 1 mission entry."))
    except Exception as e:
        checks.append(("DuckDB Analytical Lake", "FAILED", str(e)))

    # Render Summary Table
    df_checks = pd.DataFrame(checks, columns=["Subsystem Component", "Status", "Diagnostic Details"])
    print(df_checks.to_string(index=False))
    print("==========================================================================")

    all_passed = all(c[1] == "PASSED" for c in checks)
    if all_passed:
        print("[+] ALL SYSTEM SMOKE TESTS PASSED: READY FOR PRODUCTION FLIGHT.")
        sys.exit(0)
    else:
        print("[!] SMOKE TEST DETECTED CRITICAL PIPELINE FAILURES.")
        sys.exit(1)

if __name__ == "__main__":
    execute_system_smoke_test()