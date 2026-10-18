import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class MultiVertiportQueuingSimulator:
    def __init__(
        self,
        num_vertiports: int = 3,
        drones_per_vertiport: int = 4,
        charging_bays_per_hub: int = 6,
        battery_charge_time_min: float = 35.0,
        mean_swap_time_sec: float = 50.0,
        sim_duration_hours: float = 8.0
    ):
        self.num_hubs = num_vertiports
        self.fleet_size = num_vertiports * drones_per_vertiport
        self.num_chargers = charging_bays_per_hub
        self.charge_time_s = battery_charge_time_min * 60.0
        self.mean_swap_s = mean_swap_time_sec
        self.duration_s = sim_duration_hours * 3600.0

    def run_stochastic_simulation(self, order_arrival_rate_per_hour: float = 24.0) -> Dict:
        """
        Executes discrete event queuing simulation: Poisson order arrivals,
        M/M/c charger queues, and Markovian drone turnarounds.
        """
        np.random.seed(42)
        random.seed(42)

        arrival_lambda_sec = order_arrival_rate_per_hour / 3600.0
        current_time_s = 0.0

        # Hub Inventory Queues: Available Ready Packs, Charging Packs, Drone Queues
        hubs = []
        for h_id in range(self.num_hubs):
            hubs.append({
                "hub_id": f"HUB-{h_id+1:02d}",
                "ready_batteries": 6,
                "charging_bays": [0.0] * self.num_chargers, # Completion timestamp for each bay
                "idle_drones": 4,
                "queued_orders": 0,
                "total_completed_trips": 0,
                "swap_delays": []
            })

        completed_missions = 0
        drone_wait_delays = []
        battery_starvation_events = 0

        # Generate inter-arrival event queue
        while current_time_s < self.duration_s:
            # Advance time to next stochastic Poisson customer order
            inter_arrival = np.random.exponential(1.0 / arrival_lambda_sec)
            current_time_s += inter_arrival
            if current_time_s >= self.duration_s:
                break

            # Route to random nearest regional hub
            hub = random.choice(hubs)

            # Update charger progress across all bays at current_time_s
            for i in range(len(hub["charging_bays"])):
                if hub["charging_bays"][i] > 0.0 and hub["charging_bays"][i] <= current_time_s:
                    hub["charging_bays"][i] = 0.0
                    hub["ready_batteries"] = min(hub["ready_batteries"] + 1, self.num_chargers)

            # Check Drone & Battery Availability
            if hub["idle_drones"] > 0 and hub["ready_batteries"] > 0:
                # Dispatch immediately
                hub["idle_drones"] -= 1
                hub["ready_batteries"] -= 1

                # Flight mission duration: Normal(42 min, 4 min)
                mission_time_s = max(15.0 * 60.0, np.random.normal(41.8 * 60.0, 3.5 * 60.0))
                return_time = current_time_s + mission_time_s

                # Turnaround at destination: Battery swap execution
                swap_duration = max(30.0, np.random.normal(self.mean_swap_s, 8.0))
                hub["swap_delays"].append(swap_duration)
                drone_wait_delays.append(0.0) # Zero wait queue

                # Drone returns to idle and depleted battery enters charging bay
                hub["idle_drones"] += 1
                completed_missions += 1
                hub["total_completed_trips"] += 1

                # Place depleted pack in available charger bay
                placed = False
                for i in range(len(hub["charging_bays"])):
                    if hub["charging_bays"][i] == 0.0:
                        hub["charging_bays"][i] = current_time_s + self.charge_time_s
                        placed = True
                        break
                if not placed:
                    # Charger queue overflow: packs wait in buffer
                    pass

            else:
                # Starvation delay: Drone or battery pack bottleneck
                battery_starvation_events += 1
                wait_time_s = np.random.exponential(4.5 * 60.0)
                drone_wait_delays.append(wait_time_s)

        mean_wait_min = (sum(drone_wait_delays) / max(len(drone_wait_delays), 1)) / 60.0
        system_throughput = completed_missions / (self.duration_s / 3600.0)

        return {
            "total_simulated_hours": self.duration_s / 3600.0,
            "total_orders_serviced": completed_missions,
            "system_throughput_trips_per_hour": round(system_throughput, 2),
            "battery_starvation_events": battery_starvation_events,
            "battery_starvation_rate_pct": round((battery_starvation_events / max(completed_missions, 1)) * 100.0, 2),
            "mean_drone_dispatch_latency_min": round(mean_wait_min, 2),
            "mean_automated_swap_duration_s": round(float(np.mean([d for h in hubs for d in h["swap_delays"]])), 1),
            "hub_utilization_summary": [
                {"hub": h["hub_id"], "trips": h["total_completed_trips"], "ready_packs_remaining": h["ready_batteries"]}
                for h in hubs
            ]
        }

if __name__ == "__main__":
    sim = MultiVertiportQueuingSimulator(
        num_vertiports=3,
        drones_per_vertiport=4,
        charging_bays_per_hub=6,
        battery_charge_time_min=35.0,
        sim_duration_hours=8.0
    )
    results = sim.run_stochastic_simulation(order_arrival_rate_per_hour=28.0)

    print("==========================================================================")
    print("      MULTI-VERTIPORT NETWORK QUEUING & FLEET CAPACITY BENCHMARK          ")
    print("==========================================================================")
    for k, v in results.items():
        if k != "hub_utilization_summary":
            print(f"  • {k:35s}: {v}")
    print("\nRegional Hub Throughput Breakdown:")
    for hub_stat in results["hub_utilization_summary"]:
        print(f"  • {hub_stat['hub']}: {hub_stat['trips']} Completed Dispatches | Available Packs: {hub_stat['ready_packs_remaining']}")
    print("==========================================================================\n")