import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

DATABASE_FILE = "data/uav_mission_data.duckdb"

class MissionDataLake:
    def __init__(self, db_path=DATABASE_FILE):
        self.conn = duckdb.connect(db_path)
        self._initialize_schema()

    def _initialize_schema(self):
        """Creates analytical tables for missions, live telemetry, and dynamic events."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mission_records (
                mission_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP,
                total_stops INTEGER,
                payload_kg DOUBLE,
                planned_eta_min DOUBLE,
                actual_flight_time_min DOUBLE,
                energy_consumed_wh DOUBLE,
                savings_vs_baseline_min DOUBLE,
                status VARCHAR
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS in_flight_telemetry (
                entry_id BIGINT PRIMARY KEY,
                mission_id VARCHAR,
                timestamp TIMESTAMP,
                lat DOUBLE,
                lon DOUBLE,
                altitude_m DOUBLE,
                ground_speed_ms DOUBLE,
                battery_soc_pct DOUBLE,
                power_watts DOUBLE,
                wind_speed_kmh DOUBLE
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS intervention_events (
                event_id VARCHAR PRIMARY KEY,
                mission_id VARCHAR,
                timestamp TIMESTAMP,
                trigger_type VARCHAR,
                action_taken VARCHAR,
                pre_event_soc DOUBLE,
                reroute_latency_ms DOUBLE
            );
        """)
        print("[+] DuckDB Analytical Data Lake initialized.")

    def log_mission(self, mission_id, stops, payload, eta, flight_time, energy, savings, status):
        self.conn.execute("""
            INSERT INTO mission_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (mission_id, datetime.now(), stops, payload, eta, flight_time, energy, savings, status))

    def log_telemetry_batch(self, df_telemetry):
        """High-speed vectorized insert for streamed telemetry frames."""
        self.conn.register("df_temp", df_telemetry)
        self.conn.execute("""
            INSERT INTO in_flight_telemetry SELECT * FROM df_temp
        """)
        self.conn.unregister("df_temp")

    def log_intervention(self, event_id, mission_id, trigger_type, action, soc, latency_ms):
        self.conn.execute("""
            INSERT INTO intervention_events VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (event_id, mission_id, datetime.now(), trigger_type, action, soc, latency_ms))

    def query_mission_kpis(self):
        """Calculates executive KPI metrics across all logged flights."""
        query = """
            SELECT 
                COUNT(*) AS total_missions,
                AVG(actual_flight_time_min) AS mean_flight_duration,
                AVG(energy_consumed_wh) AS mean_energy_wh,
                AVG(savings_vs_baseline_min) AS mean_time_savings_min,
                SUM(CASE WHEN actual_flight_time_min > 55.0 THEN 1 ELSE 0 END) AS battery_violations
            FROM mission_records;
        """
        return self.conn.execute(query).df()

if __name__ == "__main__":
    lake = MissionDataLake()
    
    # Insert simulated test mission
    lake.log_mission(
        mission_id="MSN-2026-0801",
        stops=4,
        payload=1.85,
        eta=42.1,
        flight_time=41.8,
        energy=176.4,
        savings=14.3,
        status="COMPLETED_SUCCESS"
    )

    kpi_summary = lake.query_mission_kpis()
    print("========================================================")
    print("            DATA LAKE HISTORICAL KPI AUDIT              ")
    print("========================================================")
    print(kpi_summary.to_string(index=False))
    print("========================================================")