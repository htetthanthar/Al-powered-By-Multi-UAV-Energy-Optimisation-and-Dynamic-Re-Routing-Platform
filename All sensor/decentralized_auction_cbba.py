import copy
import math
import numpy as np
from typing import List, Dict, Tuple

class DroneAgentCBBA:
    def __init__(
        self,
        agent_id: str,
        current_lat: float,
        current_lon: float,
        current_payload_kg: float,
        remaining_battery_wh: float,
        max_payload_kg: float = 2.0
    ):
        self.id = agent_id
        self.lat = current_lat
        self.lon = current_lon
        self.current_payload = current_payload_kg
        self.battery_wh = remaining_battery_wh
        self.max_payload = max_payload_kg

        self.bundle: List[str] = []         # Ordered list of accepted task IDs
        self.path: List[Dict] = []          # Ordered waypoints
        self.winning_bids: Dict[str, float] = {}  # Highest known bids for tasks
        self.winning_agents: Dict[str, str] = {}  # Winning agent for each task

    def compute_marginal_score(self, task: Dict) -> float:
        """
        Calculates marginal score (reward minus energy cost) for inserting a task.
        Returns -inf if payload capacity is exceeded.
        """
        if self.current_payload + task["weight_kg"] > self.max_payload:
            return -float('inf')

        # Distance calculation
        dx = (task["lon"] - self.lon) * 111.0
        dy = (task["lat"] - self.lat) * 111.0
        dist_km = math.hypot(dx, dy)

        # Baseline reward = 100.0 - (energy penalty ~ 4.28 Wh/km)
        cost_wh = dist_km * 4.28
        if cost_wh > (self.battery_wh * 0.70):  # Reserve margin
            return -float('inf')

        score = max(0.0, 100.0 - cost_wh)
        return score

    def build_bundle(self, task_pool: List[Dict]):
        """Phase 1: Greedily add tasks with highest marginal score."""
        while len(self.bundle) < 4:
            best_task = None
            best_score = -float('inf')

            for task in task_pool:
                t_id = task["id"]
                if t_id in self.bundle:
                    continue

                marginal_score = self.compute_marginal_score(task)
                current_winner_bid = self.winning_bids.get(t_id, -float('inf'))

                if marginal_score > current_winner_bid and marginal_score > best_score:
                    best_score = marginal_score
                    best_task = task

            if best_task is not None:
                t_id = best_task["id"]
                self.bundle.append(t_id)
                self.path.append(best_task)
                self.current_payload += best_task["weight_kg"]
                self.winning_bids[t_id] = best_score
                self.winning_agents[t_id] = self.id
            else:
                break

    def resolve_conflicts(self, peer_winning_bids: Dict[str, float], peer_winning_agents: Dict[str, str]):
        """Phase 2: Consensus update based on received peer messages."""
        for t_id in list(self.bundle):
            peer_bid = peer_winning_bids.get(t_id, -float('inf'))
            peer_winner = peer_winning_agents.get(t_id, None)

            if peer_winner != self.id and peer_bid > self.winning_bids.get(t_id, -float('inf')):
                # Outbid: release task and all subsequent tasks in bundle
                outbid_idx = self.bundle.index(t_id)
                released_tasks = self.bundle[outbid_idx:]
                
                for r_id in released_tasks:
                    self.winning_bids[r_id] = peer_bid if r_id == t_id else -float('inf')
                    self.winning_agents[r_id] = peer_winner if r_id == t_id else None

                self.bundle = self.bundle[:outbid_idx]
                self.path = self.path[:outbid_idx]
                # Recalculate payload
                self.current_payload = sum(t["weight_kg"] for t in self.path)

def run_cbba_auction_simulation():
    # Peer drones available for re-allocation
    drone_a = DroneAgentCBBA("UAV-01", 1.3500, 103.7800, current_payload_kg=0.5, remaining_battery_wh=320.0)
    drone_b = DroneAgentCBBA("UAV-02", 1.3700, 103.8500, current_payload_kg=0.4, remaining_battery_wh=290.0)

    # Orphaned delivery tasks from an aborted drone
    orphaned_tasks = [
        {"id": "ORPH-01", "lat": 1.3600, "lon": 103.7900, "weight_kg": 0.65},
        {"id": "ORPH-02", "lat": 1.3850, "lon": 103.8600, "weight_kg": 0.70}
    ]

    fleet = [drone_a, drone_b]
    converged = False
    iterations = 0

    while not converged and iterations < 10:
        iterations += 1
        # 1. Bundle construction
        for drone in fleet:
            drone.build_bundle(orphaned_tasks)

        # 2. Consensus message exchange
        converged = True
        for i in range(len(fleet)):
            for j in range(len(fleet)):
                if i != j:
                    old_bundle = list(fleet[i].bundle)
                    fleet[i].resolve_conflicts(fleet[j].winning_bids, fleet[j].winning_agents)
                    if fleet[i].bundle != old_bundle:
                        converged = False

    print("==========================================================================")
    print("      DECENTRALIZED AUCTION (CBBA) DYNAMIC TASK RE-ALLOCATION RESULTS     ")
    print("==========================================================================")
    print(f"Consensus Reached in : {iterations} Asynchronous Iterations\n")
    for drone in fleet:
        print(f"Agent {drone.id}:")
        print(f"  • Assigned Bundle : {drone.bundle}")
        print(f"  • Final Payload   : {drone.current_payload:.2f} kg / {drone.max_payload:.2f} kg")
        print(f"  • Accepted Tasks  : {[t['id'] for t in drone.path]}")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_cbba_auction_simulation()