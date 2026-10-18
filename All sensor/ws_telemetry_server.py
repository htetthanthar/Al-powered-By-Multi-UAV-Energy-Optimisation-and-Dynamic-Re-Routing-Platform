import asyncio
import json
import random
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from typing import List

app = FastAPI(title="UAV Telemetry WebSocket Streamer")

telemetry_frames_total = Counter(
    "uav_telemetry_frames_total",
    "Total telemetry frames produced by the fleet telemetry service.",
    ["drone_id"],
)
battery_soc_pct = Gauge("uav_battery_soc_percent", "Current battery state of charge.", ["drone_id"])
power_watts = Gauge("uav_power_watts", "Current electrical power draw.", ["drone_id"])
ground_speed_kmh = Gauge("uav_ground_speed_kmh", "Current ground speed.", ["drone_id"])
wind_speed_kmh = Gauge("uav_wind_speed_kmh", "Current wind magnitude.", ["drone_id"])
altitude_meters = Gauge("uav_altitude_meters", "Current altitude above ground.", ["drone_id"])
telemetry_clients = Gauge("uav_telemetry_connected_clients", "Connected WebSocket dashboard clients.")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        telemetry_clients.set(len(self.active_connections))
        print(f"[+] Client connected. Active streams: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        telemetry_clients.set(len(self.active_connections))
        print(f"[-] Client disconnected. Active streams: {len(self.active_connections)}")

    async def broadcast_json(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus scrape endpoint for Grafana's live fleet dashboard."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.websocket("/ws/telemetry")
async def telemetry_stream_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Background task receiving messages/commands from frontend
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            print(f"[*] Received command from client: {cmd}")
            if cmd.get("action") == "FORCE_REROUTE":
                # Acknowledge reroute command
                await websocket.send_json({
                    "event": "REROUTE_ACK",
                    "status": "CALCULATING_TRAJECTORY",
                    "timestamp": asyncio.get_event_loop().time()
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background telemetry generator broadcasting to all clients
async def simulated_telemetry_loop():
    soc = 100.0
    while True:
        await asyncio.sleep(1.0)
        soc = max(0.0, soc - 0.005)
        telemetry_payload = {
            "drone_id": "UAV-ALPHA",
            "timestamp": time.time(),
            "lat": 1.3000 + random.uniform(-0.001, 0.001),
            "lon": 103.8000 + random.uniform(-0.001, 0.001),
            "alt_m": 25.0 + random.uniform(-0.2, 0.2),
            "ground_speed_kmh": 65.0 + random.uniform(-1.0, 1.0),
            "battery_soc_pct": round(soc, 2),
            "power_watts": round(278.4 + random.uniform(-5.0, 5.0), 1),
            "wind_magnitude_kmh": round(13.8 + random.uniform(-1.5, 1.5), 1),
            "flight_mode": "AUTO_MISSION",
            "fsm_state": "EN_ROUTE"
        }
        drone_id = telemetry_payload["drone_id"]
        telemetry_frames_total.labels(drone_id).inc()
        battery_soc_pct.labels(drone_id).set(telemetry_payload["battery_soc_pct"])
        power_watts.labels(drone_id).set(telemetry_payload["power_watts"])
        ground_speed_kmh.labels(drone_id).set(telemetry_payload["ground_speed_kmh"])
        wind_speed_kmh.labels(drone_id).set(telemetry_payload["wind_magnitude_kmh"])
        altitude_meters.labels(drone_id).set(telemetry_payload["alt_m"])
        if manager.active_connections:
            await manager.broadcast_json(telemetry_payload)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulated_telemetry_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
