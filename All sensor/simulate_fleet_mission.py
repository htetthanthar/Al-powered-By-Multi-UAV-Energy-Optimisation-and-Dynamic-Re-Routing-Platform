# simulate_fleet_mission.py
import os
import sys
import asyncio
import json
import time
import argparse
import websockets
from dynamic_planner import haversine_km

# Resolve WebSocket broadcaster URL (CLI arg > ENV var > localhost default)
DEFAULT_WS_URL = os.getenv("WS_TELEMETRY_URL", "ws://127.0.0.1:8765")


class FleetMissionSimulator:
    def __init__(self, ws_url: str = DEFAULT_WS_URL):
        self.ws_url = ws_url
        self.depot = {"lat": 1.3000, "lon": 103.8000, "name": "Hub Depot"}
        self.drones = [
            {
                "id": "UAV-ALPHA",
                "lat": 1.3000,
                "lon": 103.8000,
                "target_lat": 1.3500,
                "target_lon": 103.8200,
                "altitude_m": 45.0,
                "speed_kmh": 65.0,
                "soc_pct": 100.0,
                "status": "EN_ROUTE"
            },
            {
                "id": "UAV-BRAVO",
                "lat": 1.3000,
                "lon": 103.8000,
                "target_lat": 1.3800,
                "target_lon": 103.7900,
                "altitude_m": 50.0,
                "speed_kmh": 62.5,
                "soc_pct": 98.5,
                "status": "EN_ROUTE"
            }
        ]

    async def run_simulation(self, total_steps: int = 20):
        print("=" * 80)
        print("         REAL-TIME MULTI-UAV FLEET TELEMETRY SIMULATOR")
        print("=" * 80)
        print(f"Target WebSocket Broadcaster: {self.ws_url}")
        print("-" * 80)

        ws = None
        try:
            ws = await websockets.connect(self.ws_url)
            print("[+] Successfully connected to Live WebSocket Telemetry Broadcaster.")
        except Exception:
            print("[!] WebSocket broadcaster not reachable. Running simulation in local loop mode.")

        for step in range(1, total_steps + 1):
            telemetry_payload = {
                "timestamp": time.time(),
                "step": step,
                "active_fleet_count": len(self.drones),
                "fleet": []
            }

            for drone in self.drones:
                drone["lat"] += (drone["target_lat"] - drone["lat"]) * 0.1
                drone["lon"] += (drone["target_lon"] - drone["lon"]) * 0.1
                drone["soc_pct"] = max(15.0, drone["soc_pct"] - 0.45)
                dist_rem_km = haversine_km(drone["lat"], drone["lon"], drone["target_lat"], drone["target_lon"])

                if dist_rem_km < 0.1:
                    drone["status"] = "PAYLOAD_DROP_COMPLETED"

                telemetry_payload["fleet"].append({
                    "drone_id": drone["id"],
                    "lat": round(drone["lat"], 5),
                    "lon": round(drone["lon"], 5),
                    "alt_m": drone["altitude_m"],
                    "speed_kmh": drone["speed_kmh"],
                    "battery_soc": round(drone["soc_pct"], 1),
                    "distance_to_target_km": round(dist_rem_km, 3),
                    "status": drone["status"]
                })

            if ws:
                try:
                    await ws.send(json.dumps(telemetry_payload))
                except Exception:
                    ws = None

            d1 = telemetry_payload["fleet"][0]
            d2 = telemetry_payload["fleet"][1]
            print(f"[Step {step:02d}/{total_steps:02d}] "
                  f"{d1['drone_id']}: Lat={d1['lat']:.4f}, Lon={d1['lon']:.4f}, SOC={d1['battery_soc']}% | "
                  f"{d2['drone_id']}: Lat={d2['lat']:.4f}, Lon={d2['lon']:.4f}, SOC={d2['battery_soc']}%")

            await asyncio.sleep(0.5)

        if ws:
            await ws.close()

        print("=" * 80)
        print("[+] Fleet mission simulation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-UAV Fleet Telemetry Simulator")
    parser.add_argument("--ws-url", type=str, default=DEFAULT_WS_URL, help="WebSocket URL for telemetry broadcast")
    parser.add_argument("--steps", type=int, default=20, help="Number of telemetry simulation steps")
    args = parser.parse_args()

    sim = FleetMissionSimulator(ws_url=args.ws_url)
    asyncio.run(sim.run_simulation(total_steps=args.steps))