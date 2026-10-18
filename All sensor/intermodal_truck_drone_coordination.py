import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class IntermodalTruckDroneCoordinator:
    def __init__(
        self,
        truck_speed_kmh: float = 30.0,       # Urban road transit speed
        drone_cruise_speed_kmh: float = 65.0, # Aerial direct cruise speed
        drone_service_time_min: float = 2.0,
        truck_service_time_min: float = 3.0,
        max_drone_flight_time_min: float = 40.0
    ):
        self.v_truck_ms = truck_speed_kmh / 3.6
        self.v_drone_ms = drone_cruise_speed_kmh / 3.6
        self.drone_service_s = drone_service_time_min * 60.0
        self.truck_service_s = truck_service_time_min * 60.0
        self.max_flight_s = max_drone_flight_time_min * 60.0

    def compute_aerial_flight_time(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean flight leg duration in seconds."""
        dist_m = math.hypot((p2[0] - p1[0]) * 111000.0, (p2[1] - p1[1]) * 111000.0)
        return dist_m / self.v_drone_ms

    def compute_road_transit_time(self, p1: Tuple[float, float], p2: Tuple[float, float], manhattan_factor: float = 1.35) -> float:
        """Simulated street grid road travel duration in seconds."""
        dist_m = (abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])) * 111000.0 * (manhattan_factor / 1.414)
        return dist_m / self.v_truck_ms

    def evaluate_rendezvous_feasibility(
        self,
        launch_node: Dict,
        drone_drop_node: Dict,
        rendezvous_node: Dict,
        intermediate_truck_stops: List[Dict]
    ) -> Dict:
        """
        Validates timing sync for triplet (i, j, k):
          - Drone launches at node i, delivers to customer j, rejoins truck at node k.
          - Truck services intermediate road customers between i and k.
        """
        p_i = (launch_node["lat"], launch_node["lon"])
        p_j = (drone_drop_node["lat"], drone_drop_node["lon"])
        p_k = (rendezvous_node["lat"], rendezvous_node["lon"])

        # 1. Drone Sub-Tour Duration
        t_drone_leg1 = self.compute_aerial_flight_time(p_i, p_j)
        t_drone_leg2 = self.compute_aerial_flight_time(p_j, p_k)
        t_drone_total = t_drone_leg1 + self.drone_service_s + t_drone_leg2

        # 2. Truck Path Duration (from node i through intermediate stops to node k)
        truck_path = [launch_node] + intermediate_truck_stops + [rendezvous_node]
        t_truck_total = 0.0
        for idx in range(len(truck_path) - 1):
            p_a = (truck_path[idx]["lat"], truck_path[idx]["lon"])
            p_b = (truck_path[idx+1]["lat"], truck_path[idx+1]["lon"])
            t_truck_total += self.compute_road_transit_time(p_a, p_b)
            if idx < len(truck_path) - 2:
                t_truck_total += self.truck_service_s

        # Synchronization Slack Time: max(t_drone, t_truck)
        rendezvous_time_s = max(t_drone_total, t_truck_total)
        drone_wait_time_s = max(0.0, t_truck_total - t_drone_total)
        truck_wait_time_s = max(0.0, t_drone_total - t_truck_total)

        is_feasible = (
            (t_drone_total <= self.max_flight_s) and
            (drone_drop_node["weight_kg"] <= 2.0) and
            (drone_wait_time_s <= 300.0) # Drone will not hover wait > 5 min
        )

        return {
            "rendezvous_feasible": is_feasible,
            "drone_flight_duration_min": round(t_drone_total / 60.0, 2),
            "truck_route_duration_min": round(t_truck_total / 60.0, 2),
            "rendezvous_elapsed_time_min": round(rendezvous_time_s / 60.0, 2),
            "drone_hover_wait_min": round(drone_wait_time_s / 60.0, 2),
            "truck_idle_wait_min": round(truck_wait_time_s / 60.0, 2),
            "time_saved_vs_pure_truck_min": round((t_truck_total + self.compute_road_transit_time(p_i, p_j) * 2 - rendezvous_time_s) / 60.0, 2)
        }

if __name__ == "__main__":
    coordinator = IntermodalTruckDroneCoordinator()

    node_launch = {"id": "TRUCK-DEPOT", "lat": 1.3000, "lon": 103.8000}
    node_drone_drop = {"id": "DRONE-CUST-12", "lat": 1.3500, "lon": 103.8400, "weight_kg": 1.25}
    node_truck_stop = {"id": "TRUCK-CUST-01", "lat": 1.3150, "lon": 103.8100}
    node_rendezvous = {"id": "RENDEZVOUS-HUB", "lat": 1.3400, "lon": 103.8250}

    sync_evaluation = coordinator.evaluate_rendezvous_feasibility(
        launch_node=node_launch,
        drone_drop_node=node_drone_drop,
        rendezvous_node=node_rendezvous,
        intermediate_truck_stops=[node_truck_stop]
    )

    print("==========================================================================")
    print("      INTERMODAL TRUCK-DRONE (FSTSP) SYNCHRONIZATION AUDIT                ")
    print("==========================================================================")
    for k, v in sync_evaluation.items():
        print(f"  • {k:32s}: {v}")
    print("==========================================================================\n")