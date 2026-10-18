# fleet_scheduler.py
import math
import numpy as np
from typing import List, Dict, Any
from dynamic_planner import optimize_stop_sequence, process_order_batch

try:
    from models.energy_mlp import EnergySurrogateMLP
except ModuleNotFoundError:
    from dynamic_planner import EnergySurrogateMLP


class FleetScheduler:
    def __init__(self, depot: Dict[str, Any] = None):
        self.depot = depot or {"lat": 1.3000, "lon": 103.8000, "name": "Central Hub Depot"}
        self.energy_model = EnergySurrogateMLP()
        if hasattr(self.energy_model, "eval"):
            self.energy_model.eval()

    def generate_fleet_schedule(
        self,
        orders: List[Dict[str, Any]],
        wind_speed_ms: float = 4.0,
        wind_angle_rad: float = 0.35,
        max_drones: int = 4
    ) -> Dict[str, Any]:
        if not orders:
            return {"missions": [], "total_energy_wh": 0.0, "total_flight_time_min": 0.0}
        if max_drones < 1:
            raise ValueError("max_drones must be at least one")

        batches = process_order_batch(orders, max_batch_size=5)
        missions = []
        total_fleet_energy = 0.0
        drone_available_at = [0.0] * max_drones

        for batch in batches:
            opt_route, eta_min, energy_wh = optimize_stop_sequence(
                orders=batch,
                depot=self.depot,
                model=self.energy_model,
                wind_speed_ms=wind_speed_ms,
                wind_angle_rad=wind_angle_rad,
                cruise_speed_ms=18.05
            )

            # Safely extract IDs for order stops, skipping depot nodes
            stop_ids = [s["id"] for s in opt_route if "id" in s]
            sequence_names = [self.depot["name"]] + stop_ids + [self.depot["name"]]

            drone_idx = min(range(max_drones), key=drone_available_at.__getitem__)
            launch_offset_min = drone_available_at[drone_idx]
            completion_time_min = launch_offset_min + eta_min
            drone_available_at[drone_idx] = completion_time_min
            drone_callsign = f"UAV-{drone_idx+1:02d}"
            missions.append({
                "drone_id": drone_callsign,
                "assigned_orders": stop_ids,
                "sequence": sequence_names,
                "eta_minutes": round(eta_min, 2),
                "launch_offset_minutes": round(launch_offset_min, 2),
                "completion_time_minutes": round(completion_time_min, 2),
                "energy_consumption_wh": round(energy_wh, 2),
                "payload_kg": round(sum(o.get("weight_kg", 0.5) for o in batch), 2)
            })

            total_fleet_energy += energy_wh

        return {
            "missions": missions,
            "drone_count": min(max_drones, len(missions)),
            "total_fleet_energy_wh": round(total_fleet_energy, 2),
            "makespan_flight_time_min": round(max(drone_available_at), 2)
        }
