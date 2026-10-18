# weather_service.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Weather & Micro-Climate Forecast Service", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "HEALTHY", "service": "Weather Microservice", "port": 8002}

@app.get("/api/v1/wind-forecast")
def get_wind_vector(lat: float, lon: float):
    return {"u_wind_ms": 3.20, "v_wind_ms": -2.10, "gust_ms": 5.4}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)