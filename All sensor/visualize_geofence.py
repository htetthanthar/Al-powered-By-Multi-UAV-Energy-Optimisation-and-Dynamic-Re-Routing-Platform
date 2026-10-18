# visualize_geofence.py
import os
import json
from pathlib import Path
import folium

def generate_geofence_map(output_html: str = "singapore_geofence_map.html"):
    # Center coordinates for Singapore airspace
    sg_center = [1.3521, 103.8198]
    m = folium.Map(location=sg_center, zoom_start=12, tiles="CartoDB dark_matter")

    geojson_path = Path("data/singapore_airspace.geojson")
    
    # Ensure data directory and mock GeoJSON exist if missing
    if not geojson_path.exists():
        geojson_path.parent.mkdir(parents=True, exist_ok=True)
        sample_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Paya Lebar Restricted Airspace (NFZ)", "type": "NFZ"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[103.88, 1.34], [103.93, 1.34], [103.93, 1.39], [103.88, 1.39], [103.88, 1.34]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Changi Control Zone (NFZ)", "type": "NFZ"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[103.96, 1.33], [104.01, 1.33], [104.01, 1.38], [103.96, 1.38], [103.96, 1.33]]]
                    }
                }
            ]
        }
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(sample_geojson, f, indent=2)

    # Load and style GeoJSON boundaries
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    folium.GeoJson(
        geojson_data,
        name="No-Fly Zones (NFZ)",
        style_function=lambda x: {
            "fillColor": "#ff0000",
            "color": "#ff3333",
            "weight": 2,
            "fillOpacity": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Zone:"])
    ).add_to(m)

    # Add Mission Waypoints and Hub Depot Marker
    depot = [1.3000, 103.8000]
    folium.Marker(
        depot,
        popup="<b>Central Logistics Hub (Depot)</b>",
        icon=folium.Icon(color="green", icon="home")
    ).add_to(m)

    orders = [
        {"id": "ORD-01", "coords": [1.3840, 103.7470], "name": "ORD-01 (0.65 kg)"},
        {"id": "ORD-02", "coords": [1.4304, 103.8354], "name": "ORD-02 (0.70 kg)"},
        {"id": "ORD-04", "coords": [1.3532, 103.9447], "name": "ORD-04 (0.50 kg)"}
    ]

    flight_path = [depot]
    for o in orders:
        flight_path.append(o["coords"])
        folium.Marker(
            o["coords"],
            popup=f"<b>{o['name']}</b>",
            icon=folium.Icon(color="blue", icon="plane")
        ).add_to(m)
    flight_path.append(depot)

    # Draw Dispatched Flight Corridor
    folium.PolyLine(
        flight_path,
        color="#00e5ff",
        weight=3.5,
        opacity=0.85,
        dash_array="6",
        tooltip="Optimized Flight Path (GAT-VRP)"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"[+] Successfully exported interactive geofence map -> {os.path.abspath(output_html)}")

if __name__ == "__main__":
    generate_geofence_map()