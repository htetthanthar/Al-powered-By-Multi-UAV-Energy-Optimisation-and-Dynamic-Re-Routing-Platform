import math
import numpy as np
import json
from shapely.geometry import Point, LineString, Polygon

class AirspaceGeometryResolver:
    def __init__(self, geojson_path="data/singapore_airspace.geojson"):
        self.restricted_polygons = []
        self._load_airspace(geojson_path)

    def _load_airspace(self, path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for feature in data["features"]:
                if feature["properties"].get("action") == "DIVERT_TO_GROUND":
                    poly = Polygon(feature["geometry"]["coordinates"][0])
                    self.restricted_polygons.append({
                        "name": feature["properties"]["name"],
                        "polygon": poly,
                        "buffered_safety": poly.buffer(0.005) # 500m safety buffer
                    })
            print(f"[+] Loaded {len(self.restricted_polygons)} restricted airspace polygons.")
        except FileNotFoundError:
            # Fallback polygon
            p = Polygon([[103.88, 1.34], [103.93, 1.34], [103.93, 1.39], [103.88, 1.39]])
            self.restricted_polygons.append({"name": "Restricted Fallback NFZ", "polygon": p, "buffered_safety": p.buffer(0.005)})

    def check_trajectory_conflict(self, p1_lonlat, p2_lonlat):
        """
        Returns True if the direct line between p1 and p2 intersects any NFZ boundary.
        """
        flight_line = LineString([p1_lonlat, p2_lonlat])
        for zone in self.restricted_polygons:
            if flight_line.intersects(zone["buffered_safety"]):
                return True, zone["name"]
        return False, None

    def compute_tangent_detour(self, start_lonlat, end_lonlat):
        """
        Computes the shortest path around intersecting polygon vertices using tangent chords.
        """
        flight_line = LineString([start_lonlat, end_lonlat])
        detour_nodes = [start_lonlat]

        for zone in self.restricted_polygons:
            poly = zone["buffered_safety"]
            if flight_line.intersects(poly):
                # Extract exterior polygon vertices as candidate navigation waypoints
                exterior_coords = list(poly.exterior.coords)[:-1]
                
                # Find the vertex that minimizes the detour distance
                best_vertex = None
                min_total_dist = float('inf')
                
                for v in exterior_coords:
                    d = math.dist(start_lonlat, v) + math.dist(v, end_lonlat)
                    if d < min_total_dist:
                        # Verify vertex line of sight
                        l1 = LineString([start_lonlat, v])
                        l2 = LineString([v, end_lonlat])
                        if not l1.crosses(zone["polygon"]) and not l2.crosses(zone["polygon"]):
                            min_total_dist = d
                            best_vertex = v

                if best_vertex:
                    detour_nodes.append(best_vertex)

        detour_nodes.append(end_lonlat)
        return detour_nodes

if __name__ == "__main__":
    resolver = AirspaceGeometryResolver()
    
    start = (103.86, 1.32) # West of NFZ
    end = (103.95, 1.37)   # East of NFZ (Intersects Paya Lebar)
    
    intersects, zone_name = resolver.check_trajectory_conflict(start, end)
    print(f"Direct Path Conflict : {intersects} ({zone_name})")
    
    if intersects:
        safe_path = resolver.compute_tangent_detour(start, end)
        print("Generated Safe Waypoint Detour Chords:")
        for idx, pt in enumerate(safe_path):
            print(f"  Hop {idx}: Lon {pt[0]:.4f}, Lat {pt[1]:.4f}")