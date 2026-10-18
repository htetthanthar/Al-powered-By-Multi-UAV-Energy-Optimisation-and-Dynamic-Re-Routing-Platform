import sys
import time
import argparse
import pandas as pd
import numpy as np
from state_machine import MissionExecutionFSM, UAVState
from dynamic_planner import EnergySurrogateMLP

def replay_telemetry_stream(log_csv_path: str, playback_rate_multiplier: float = 5.0):
    print("==========================================================================")
    print(f"[*] REPLAYING RECORDED TELEMETRY LOG: {log_csv_path}")
    print(f"[*] PLAYBACK SPEED: {playback_rate_multiplier}x Real-Time")
    print("==========================================================================")

    try:
        df = pd.read_csv(log_csv_path)
    except FileNotFoundError:
        print(f"[!] Log file not found: {log_csv_path}. Generating synthetic test replay stream...")
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-01 10:00", periods=60, freq="s"),
            "ground_speed_ms": [18.05] * 60,
            "battery_soc_pct": np.linspace(98.0, 75.0, 60),
            "power_watts": [280.0] * 30 + [420.0] * 10 + [290.0] * 20,
            "wind_speed_kmh": [12.0] * 35 + [34.5] * 10 + [14.0] * 15,
            "rain_flag": [0] * 38 + [1] * 8 + [0] * 14
        })

    fsm = MissionExecutionFSM(mission_id="REPLAY-MSN-01", total_stops=4)
    fsm.current_state = UAVState.EN_ROUTE

    anomalies_detected = 0

    for idx, row in df.iterrows():
        t_frame = {
            "wind_kmh": float(row["wind_speed_kmh"]),
            "rain_flag": bool(row["rain_flag"]),
            "battery_soc": float(row["battery_soc_pct"]),
            "alt_m": 25.0
        }

        prev_state = fsm.current_state
        new_state = fsm.process_telemetry_frame(t_frame)

        if new_state != prev_state:
            anomalies_detected += 1
            print(f"  [FRAME {idx:03d} | ALERT] State Shift: {prev_state.value} -> {new_state.value} | "
                  f"Wind: {t_frame['wind_kmh']} km/h, Rain: {t_frame['rain_flag']}, SOC: {t_frame['battery_soc']}%")
        else:
            if idx % 10 == 0:
                print(f"  [FRAME {idx:03d}] Nominal Cruise | Speed: {row.get('ground_speed_ms', 18.05):.1f} m/s | "
                      f"Power: {row['power_watts']:.1f} W | SOC: {t_frame['battery_soc']:.1f}%")

        time.sleep(1.0 / (playback_rate_multiplier * 10.0))

    print("==========================================================================")
    print(f"[+] REPLAY COMPLETE: {len(df)} Frames Ingested | {anomalies_detected} Dynamic Interventions Executed.")
    print("==========================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Telemetry Replay & Post-Mortem Analyzer")
    parser.add_argument("--log", type=str, default="data/flight_telemetry.csv", help="Path to telemetry CSV log")
    parser.add_argument("--speed", type=float, default=20.0, help="Playback speed multiplier")
    args = parser.parse_args()

    replay_telemetry_stream(args.log, args.speed)