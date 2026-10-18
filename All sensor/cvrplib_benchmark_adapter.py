import math
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class CVRPLIBBenchmarkAdapter:
    def __init__(self, output_dir: str = "data/benchmarks"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_vrp_instance(self, raw_vrp_text: str) -> Dict:
        """
        Parses standard TSPLIB/CVRPLIB formatted text into structured coordinate and demand arrays.
        """
        lines = [line.strip() for line in raw_vrp_text.strip().split("\n") if line.strip()]
        node_coords = []
        demands = []
        section = None

        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                section = "COORDS"
                continue
            elif line.startswith("DEMAND_SECTION"):
                section = "DEMANDS"
                continue
            elif line.startswith("DEPOT_SECTION") or line.startswith("EOF"):
                section = None
                continue

            parts = line.split()
            if section == "COORDS" and len(parts) >= 3:
                node_coords.append((float(parts[1]), float(parts[2])))
            elif section == "DEMANDS" and len(parts) >= 2:
                demands.append(float(parts[1]))

        return {
            "coords": np.array(node_coords),
            "demands": np.array(demands)
        }

    def adapt_to_uav_instance(
        self,
        instance_name: str,
        coords: np.ndarray,
        demands: np.ndarray,
        center_lat: float = 1.3521,
        center_lon: float = 103.8198,
        scale_km: float = 15.0
    ) -> pd.DataFrame:
        """
        Normalizes arbitrary integer grids to Singapore geographic coordinate bounds
        and scales demand weights to the 0.1 kg - 0.8 kg per-stop delivery envelope.
        """
        # Normalize coordinates to [-1, 1] then project to GPS
        c_min, c_max = coords.min(axis=0), coords.max(axis=0)
        norm_coords = (coords - c_min) / (c_max - c_min + 1e-6) * 2.0 - 1.0

        # Degree scaling factor: 1 deg lat ~= 111 km
        lat_offset = (norm_coords[:, 1] * scale_km) / 111.0
        lon_offset = (norm_coords[:, 0] * scale_km) / (111.0 * math.cos(math.radians(center_lat)))

        gps_lats = center_lat + lat_offset
        gps_lons = center_lon + lon_offset

        # Normalize demands to UAV parcel limits (0.1 to 0.75 kg)
        d_norm = demands / (demands.max() + 1e-6) * 0.75
        d_norm[0] = 0.0  # Depot demand is 0

        # Inject micro-climate wind vector fields
        np.random.seed(42)
        wind_u = np.random.normal(2.5, 1.2, len(coords))
        wind_v = np.random.normal(-1.5, 1.0, len(coords))
        elevations = np.random.uniform(15.0, 35.0, len(coords))

        df_adapted = pd.DataFrame({
            "node_id": [f"NODE_{i:03d}" if i > 0 else "DEPOT" for i in range(len(coords))],
            "lat": np.round(gps_lats, 6),
            "lon": np.round(gps_lons, 6),
            "elevation_m": np.round(elevations, 1),
            "demand_kg": np.round(d_norm, 2),
            "wind_u_ms": np.round(wind_u, 2),
            "wind_v_ms": np.round(wind_v, 2)
        })

        out_path = os.path.join(self.output_dir, f"{instance_name}_uav_adapted.csv")
        df_adapted.to_csv(out_path, index=False)
        print(f"[+] Adapted {instance_name} ({len(coords)} nodes) -> {out_path}")
        return df_adapted

if __name__ == "__main__":
    # Synthetic standard CVRPLIB instance sample (Augerat A-n32-k5 format)
    sample_cvrplib_text = """
    NAME : A-n10-k2
    TYPE : CVRP
    DIMENSION : 10
    EDGE_WEIGHT_TYPE : EUC_2D
    CAPACITY : 100
    NODE_COORD_SECTION
    1 50 50
    2 38 42
    3 62 70
    4 25 80
    5 78 30
    6 90 65
    7 15 20
    8 45 90
    9 82 15
    10 30 60
    DEMAND_SECTION
    1 0
    2 19
    3 21
    4 16
    5 25
    6 12
    7 18
    8 22
    9 10
    10 14
    DEPOT_SECTION
    1
    -1
    EOF
    """
    adapter = CVRPLIBBenchmarkAdapter()
    parsed = adapter.parse_vrp_instance(sample_cvrplib_text)
    df_uav = adapter.adapt_to_uav_instance("Augerat_A-n10", parsed["coords"], parsed["demands"])
    print(df_uav.to_string(index=False))