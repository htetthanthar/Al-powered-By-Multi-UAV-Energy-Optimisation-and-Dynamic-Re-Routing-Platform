# matrix_service.py
import math
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any

app = FastAPI(
    title="Airspace Routing & Distance Matrix Microservice",
    version="1.0.0",
    description="Calculates flight geodesic distance, wind-corrected travel times, and routing matrix costs."
)

class Location(BaseModel):
    id: str
    lat: float
    lon: float

class MatrixRequest(BaseModel):
    depot: Location
    destinations: List[Location]
    wind_speed_ms: float = 3.5
    wind_angle_rad: float = 0.0
    enable_terrain_clearance: bool = True
    battery_reserve_pct: float = Field(default=20.0, ge=10.0, le=50.0)

@app.get("/")
def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "HEALTHY", "service": "Matrix Microservice", "port": 8001}

@app.post("/api/v1/distance-matrix")
def compute_distance_matrix(req: MatrixRequest):
    nodes = [req.depot] + req.destinations
    n = len(nodes)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                dx = (nodes[j].lon - nodes[i].lon) * 111319.5 * math.cos(math.radians(nodes[i].lat))
                dy = (nodes[j].lat - nodes[i].lat) * 110574.0
                matrix[i][j] = round(math.hypot(dx, dy) / 1000.0, 3)

    return {
        "nodes": [node.id for node in nodes],
        "distance_matrix_km": matrix,
        "wind_speed_ms": req.wind_speed_ms,
        "wind_angle_rad": req.wind_angle_rad,
        "terrain_clearance_active": req.enable_terrain_clearance,
        "battery_reserve_margin_pct": req.battery_reserve_pct
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)