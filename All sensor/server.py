# server.py
import os
import math
import uvicorn
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from dynamic_planner import optimize_stop_sequence, evaluate_route_feasibility
from formal_invariants_verifier import RouteSafetySMTVerifier
from fleet_scheduler import FleetScheduler

app = FastAPI(
    title="Autonomous UAV Ground Station REST Gateway",
    version="1.0.0",
    description="Micro-climate drone route optimization, flight dispatch, and SMT formal verification API."
)

scheduler = FleetScheduler()
verifier = RouteSafetySMTVerifier()

# Service Discovery Configuration
WEATHER_SERVICE_URL = os.getenv("WEATHER_SERVICE_URL", "http://127.0.0.1:8002")
MATRIX_SERVICE_URL = os.getenv("MATRIX_SERVICE_URL", "http://127.0.0.1:8001")


class OrderItem(BaseModel):
    id: str
    lat: float = Field(..., ge=-90.0, le=90.0, allow_inf_nan=False)
    lon: float = Field(..., ge=-180.0, le=180.0, allow_inf_nan=False)
    weight_kg: float = Field(..., gt=0.0, le=2.0, allow_inf_nan=False)


class DispatchRequest(BaseModel):
    orders: List[OrderItem] = Field(..., min_length=1, max_length=20)
    wind_speed_ms: float = Field(default=3.5, ge=0.0, le=8.33, allow_inf_nan=False)
    wind_angle_rad: float = Field(default=0.0, ge=-math.pi, le=math.pi, allow_inf_nan=False)
    cruise_speed_ms: float = Field(default=18.05, ge=2.0, le=40.0, allow_inf_nan=False)


class MissionResponse(BaseModel):
    status: str
    safety_clearance: bool
    flight_time_min: float
    total_energy_wh: float
    optimized_route: List[Dict[str, Any]]


@app.get("/")
def root():
    index_file = Path("index.html")
    if index_file.exists():
        return FileResponse(index_file)
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "engine": "Drone On AI", "version": "1.0.0"}


@app.get("/api/v1/mission/weather-status")
async def get_weather_status(lat: float = 1.3000, lon: float = 103.8000):
    """Asynchronously queries the weather microservice for localized wind and climate vectors."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{WEATHER_SERVICE_URL}/api/v1/wind-forecast?lat={lat}&lon={lon}")
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Weather service returned error: {resp.text}"
                )
            return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Weather microservice unreachable at {WEATHER_SERVICE_URL}: {str(exc)}"
        )


@app.post("/api/v1/dispatch", response_model=MissionResponse)
def dispatch_mission(request: DispatchRequest):
    depot = {"lat": 1.3000, "lon": 103.8000, "name": "Central Hub Depot"}
    orders_dicts = [o.model_dump() for o in request.orders]

    route, eta, energy = optimize_stop_sequence(
        orders=orders_dicts,
        depot=depot,
        wind_speed_ms=request.wind_speed_ms,
        wind_angle_rad=request.wind_angle_rad,
        cruise_speed_ms=request.cruise_speed_ms
    )
    feasibility = evaluate_route_feasibility(
        route,
        wind_speed_ms=request.wind_speed_ms,
        wind_angle_rad=request.wind_angle_rad,
        cruise_speed_ms=request.cruise_speed_ms
    )
    safety_audit = verifier.verify_mission_invariants(
        orders_dicts,
        total_distance_km=feasibility["total_distance_km"],
        projected_wind_kmh=request.wind_speed_ms * 3.6,
        cruise_speed_kmh=request.cruise_speed_ms * 3.6
    )
    if not feasibility["feasible"] or not safety_audit.get("mathematical_safety_clearance", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Mission violates the operational safety envelope.",
                "route_feasibility": feasibility,
                "smt_status": safety_audit["smt_solver_status"]
            }
        )

    return MissionResponse(
        status="DISPATCHED",
        safety_clearance=True,
        flight_time_min=round(eta, 2),
        total_energy_wh=round(energy, 2),
        optimized_route=route
    )


@app.post("/api/v1/fleet/schedule")
def schedule_fleet(orders: List[OrderItem]):
    orders_dicts = [o.model_dump() for o in orders]
    return scheduler.generate_fleet_schedule(orders_dicts)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
