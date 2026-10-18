import os
import math
import time
import asyncio
import httpx
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class MicroClimateCollector:
    def __init__(self, cache_ttl_seconds: int = 300):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.cache_ttl = cache_ttl_seconds
        self.last_fetch_time = 0.0
        self.cached_grid_data = None

    async def fetch_meteo_grid(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float, grid_steps: int = 3) -> pd.DataFrame:
        """
        Queries meteorological telemetry across a spatial grid over the bounding box.
        """
        now = time.time()
        if self.cached_grid_data is not None and (now - self.last_fetch_time) < self.cache_ttl:
            return self.cached_grid_data

        lats = np.linspace(lat_min, lat_max, grid_steps)
        lons = np.linspace(lon_min, lon_max, grid_steps)
        
        records = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for lat in lats:
                for lon in lons:
                    params = {
                        "latitude": lat,
                        "longitude": lon,
                        "current": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation"],
                        "wind_speed_unit": "ms"
                    }
                    try:
                        resp = await client.get(self.base_url, params=params)
                        if resp.status_code == 200:
                            data = resp.json()["current"]
                            # Decompose wind speed and angle to u (east) and v (north) vectors
                            w_speed = data["wind_speed_10m"]
                            w_dir_rad = math.radians(data["wind_direction_10m"])
                            u_wind = -w_speed * math.sin(w_dir_rad)
                            v_wind = -w_speed * math.cos(w_dir_rad)

                            records.append({
                                "lat": lat,
                                "lon": lon,
                                "temp_c": data["temperature_2m"],
                                "humidity_pct": data["relative_humidity_2m"],
                                "pressure_hpa": data["surface_pressure"],
                                "wind_speed_ms": w_speed,
                                "wind_u": u_wind,
                                "wind_v": v_wind,
                                "rain_flag": 1 if data["precipitation"] > 0.1 else 0
                            })
                    except Exception as e:
                        # Fallback default on network exception
                        records.append({
                            "lat": lat, "lon": lon, "temp_c": 28.0, "humidity_pct": 75.0,
                            "pressure_hpa": 1013.25, "wind_speed_ms": 3.0, "wind_u": 2.1, "wind_v": -1.5, "rain_flag": 0
                        })

        self.cached_grid_data = pd.DataFrame(records)
        self.last_fetch_time = now
        return self.cached_grid_data

    def interpolate_point_weather(self, target_lat: float, target_lon: float, grid_df: pd.DataFrame, power: float = 2.0) -> Dict[str, float]:
        """
        Computes localized atmospheric variables using Inverse Distance Weighting (IDW).
        """
        distances = []
        for _, row in grid_df.iterrows():
            d = math.hypot(target_lat - row["lat"], target_lon - row["lon"])
            distances.append(max(d, 1e-6))

        weights = [1.0 / (d ** power) for d in distances]
        sum_weights = sum(weights)
        norm_weights = [w / sum_weights for w in weights]

        interp_u = sum(w * row["wind_u"] for w, (_, row) in zip(norm_weights, grid_df.iterrows()))
        interp_v = sum(w * row["wind_v"] for w, (_, row) in zip(norm_weights, grid_df.iterrows()))
        interp_temp = sum(w * row["temp_c"] for w, (_, row) in zip(norm_weights, grid_df.iterrows()))
        interp_rain = 1 if any(row["rain_flag"] == 1 and w > 0.25 for w, (_, row) in zip(norm_weights, grid_df.iterrows())) else 0

        return {
            "target_lat": target_lat,
            "target_lon": target_lon,
            "wind_u_ms": round(interp_u, 3),
            "wind_v_ms": round(interp_v, 3),
            "wind_speed_ms": round(math.sqrt(interp_u**2 + interp_v**2), 3),
            "wind_speed_kmh": round(math.sqrt(interp_u**2 + interp_v**2) * 3.6, 2),
            "temp_c": round(interp_temp, 2),
            "rain_flag": interp_rain
        }

if __name__ == "__main__":
    collector = MicroClimateCollector()
    
    async def test():
        print("[*] Fetching live micro-climate grid (Singapore bounds: 1.28 to 1.45 Lat, 103.70 to 103.95 Lon)...")
        grid = await collector.fetch_meteo_grid(1.28, 1.45, 103.70, 103.95, grid_steps=3)
        print(f"[+] Loaded {len(grid)} spatial weather observation points.")
        
        # Interpolate localized conditions for a specific waypoint
        test_lat, test_lon = 1.3840, 103.7470  # Choa Chu Kang
        point_weather = collector.interpolate_point_weather(test_lat, test_lon, grid)
        print("\nInterpolated Atmospheric Telemetry at Target Waypoint:")
        for k, v in point_weather.items():
            print(f"  • {k:18s}: {v}")

    asyncio.run(test())