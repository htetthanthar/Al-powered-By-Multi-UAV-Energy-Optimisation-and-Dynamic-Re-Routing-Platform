import sys
import time
import numpy as np
import pandas as pd
from dynamic_planner import EnergySurrogateMLP, optimize_stop_sequence, evaluate_route_feasibility

def execute_regression_harness(num_episodes=25, stops_per_mission=4):
    print("==========================================================================")
    print(f"[*] INITIATING SITL HEADLESS REGRESSION HARNESS ({num_episodes} EPISODES)")
    print("==========================================================================")
    
    energy_model = EnergySurrogateMLP()
    energy_model.eval()
    depot = {"name": "Central Depot", "lat": 1.3000, "lon": 103.8000}

    passed_episodes = 0
    failed_episodes = 0
    audit_logs = []

    for episode in range(1, num_episodes + 1):
        # Generate randomized mission parameters
        np.random.seed(episode * 17)
        orders = [
            {
                "id": f"ORD-EP{episode:02d}-{i:02d}",
                "lat": 1.3000 + np.random.uniform(-0.06, 0.06),
                "lon": 103.8000 + np.random.uniform(-0.06, 0.06),
                "weight_kg": round(np.random.uniform(0.3, 0.6), 2)
            }
            for i in range(stops_per_mission)
        ]

        wind_speed_ms = np.random.uniform(1.0, 7.5) # Up to 27 km/h
        wind_angle_rad = np.random.uniform(0.0, 2 * np.pi)

        t_start = time.perf_counter()
        opt_route, eta, energy = optimize_stop_sequence(
            orders, depot, energy_model, wind_speed_ms, wind_angle_rad
        )
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Assertions
        payload_valid = sum(o["weight_kg"] for o in orders) <= 2.0
        time_valid = eta <= 55.0
        latency_valid = latency_ms <= 100.0

        if payload_valid and time_valid and latency_valid:
            passed_episodes += 1
            status = "PASS"
        else:
            failed_episodes += 1
            status = "FAIL"

        audit_logs.append({
            "Episode": f"EP-{episode:02d}",
            "Payload (kg)": sum(o["weight_kg"] for o in orders),
            "Wind (km/h)": round(wind_speed_ms * 3.6, 1),
            "ETA (min)": round(eta, 2),
            "Energy (Wh)": round(energy, 2),
            "Latency (ms)": round(latency_ms, 2),
            "Status": status
        })

    df_audit = pd.DataFrame(audit_logs)
    print(df_audit.to_string(index=False))
    print("==========================================================================")
    print(f"REGRESSION RESULTS: {passed_episodes}/{num_episodes} Passed | {failed_episodes} Failed")
    print("==========================================================================")

    if failed_episodes > 0:
        print("[!] REGRESSION FAILED: One or more operational boundaries were violated.")
        sys.exit(1)
    else:
        print("[+] ALL REGRESSION TEST CASES PASSED SUCCESSFULLY.")
        sys.exit(0)

if __name__ == "__main__":
    execute_regression_harness()