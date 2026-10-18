import os
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

DATABASE_FILE = "data/uav_mission_data.duckdb"
OUTPUT_DIR = "reports/blackbox_audits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BlackboxFlightAnalyzer:
    def __init__(self, db_path: str = DATABASE_FILE):
        self.conn = duckdb.connect(db_path)

    def analyze_mission(self, mission_id: str) -> Dict[str, Any]:
        """
        Extracts telemetry frames for a specific mission and performs structural diagnostics.
        """
        query = f"""
            SELECT * FROM in_flight_telemetry 
            WHERE mission_id = '{mission_id}'
            ORDER BY timestamp ASC;
        """
        df = self.conn.execute(query).df()
        
        if df.empty:
            print(f"[!] No recorded telemetry found for mission ID: {mission_id}")
            return {}

        # 1. Kinematic & Tracking Diagnostics
        speeds = df["ground_speed_ms"].values
        mean_speed = float(np.mean(speeds))
        max_speed = float(np.max(speeds))
        speed_variance = float(np.var(speeds))

        # 2. Electrical & Battery Discharge Diagnostics
        soc = df["battery_soc_pct"].values
        power = df["power_watts"].values
        initial_soc = soc[0]
        final_soc = soc[-1]
        total_soc_depletion = initial_soc - final_soc
        mean_power_draw = float(np.mean(power))
        peak_power_draw = float(np.max(power))

        # 3. Anomaly & Safety Threshold Evaluation
        over_power_events = int(np.sum(power > 400.0))  # Threshold: 400 Watts peak
        wind_speeds = df["wind_speed_kmh"].values
        gust_violations = int(np.sum(wind_speeds > 30.0)) # Threshold: 30 km/h wind limit

        # Export Diagnostic Profile Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        time_index = np.arange(len(df))

        ax1.plot(time_index, power, color="#e26d5c", label="Instantaneous Power (Watts)")
        ax1.axhline(400.0, color="red", linestyle="--", alpha=0.7, label="Peak Structural Limit (400W)")
        ax1.set_ylabel("Power (Watts)")
        ax1.set_title(f"Blackbox Telemetry Audit: Mission {mission_id}")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2.plot(time_index, soc, color="#2b5c8f", label="Battery SOC (%)")
        ax2.axhline(20.0, color="red", linestyle=":", alpha=0.7, label="Emergency Reserve (20%)")
        ax2.set_xlabel("Recorded Telemetry Frames (Time Steps)")
        ax2.set_ylabel("Battery State of Charge (%)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plot_path = os.path.join(OUTPUT_DIR, f"{mission_id}_diagnostic.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

        report = {
            "mission_id": mission_id,
            "total_frames_analyzed": len(df),
            "mean_ground_speed_ms": round(mean_speed, 2),
            "peak_ground_speed_ms": round(max_speed, 2),
            "mean_power_watts": round(mean_power_draw, 2),
            "peak_power_watts": round(peak_power_draw, 2),
            "total_soc_depletion_pct": round(total_soc_depletion, 2),
            "power_spike_anomalies": over_power_events,
            "gust_violations_recorded": gust_violations,
            "airworthiness_status": "COMPLIANT" if (over_power_events == 0 and final_soc >= 20.0) else "FLAGGED_FOR_INSPECTION",
            "plot_saved_to": plot_path
        }
        return report

if __name__ == "__main__":
    analyzer = BlackboxFlightAnalyzer()
    
    # Generate mock flight data if empty
    tables = analyzer.conn.execute("SHOW TABLES").df()
    if "in_flight_telemetry" not in tables["name"].values or analyzer.conn.execute("SELECT COUNT(*) FROM in_flight_telemetry").fetchone()[0] == 0:
        from simulate_fleet_mission import run_simulation_cluster
        import asyncio
        asyncio.run(run_simulation_cluster())

    audit_result = analyzer.analyze_mission("SIM-MISSION-01")
    print("========================================================")
    print("             POST-FLIGHT BLACKBOX AUDIT REPORT          ")
    print("========================================================")
    for k, v in audit_result.items():
        print(f"  • {k:26s}: {v}")
    print("========================================================")