import math
import numpy as np
from typing import List, Dict, Tuple

class ArtificialPotentialFieldDAA:
    def __init__(
        self,
        k_attractive: float = 1.2,
        k_repulsive: float = 150.0,
        obstacle_influence_radius_m: float = 40.0,
        emergency_bubble_radius_m: float = 10.0,
        max_avoidance_accel_ms2: float = 4.5
    ):
        self.k_att = k_attractive
        self.k_rep = k_repulsive
        self.d_inf = obstacle_influence_radius_m
        self.d_bubble = emergency_bubble_radius_m
        self.max_accel = max_avoidance_accel_ms2

    def compute_attractive_force(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        """
        Attractive force gradient toward next trajectory waypoint:
        F_att = -k_att * (p - p_goal)
        """
        dist_to_goal = np.linalg.norm(goal_pos - current_pos)
        if dist_to_goal < 1e-4:
            return np.zeros(3)
        return self.k_att * (goal_pos - current_pos)

    def compute_repulsive_force(
        self,
        current_pos: np.ndarray,
        obstacles: List[Dict]
    ) -> np.ndarray:
        """
        Repulsive force gradient away from nearby obstacles:
        F_rep = k_rep * (1/d - 1/d_0) * (1/d^2) * grad(d)
        """
        total_f_rep = np.zeros(3)

        for obs in obstacles:
            obs_pos = np.array([obs["x"], obs["y"], obs["z"]])
            diff = current_pos - obs_pos
            dist = np.linalg.norm(diff)

            if dist < self.d_inf:
                dist_safe = max(dist, 0.5)
                # Repulsive force formulation
                scale = self.k_rep * (1.0 / dist_safe - 1.0 / self.d_inf) * (1.0 / (dist_safe ** 2))
                direction = diff / dist_safe
                
                # Emergency escalation if within critical bubble
                if dist < self.d_bubble:
                    scale *= 3.0

                total_f_rep += scale * direction

        return total_f_rep

    def compute_reactive_steering(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        goal_pos: np.ndarray,
        detected_obstacles: List[Dict]
    ) -> Dict:
        """
        Combines potential fields and damps velocity to calculate safe steering acceleration vector.
        """
        f_att = self.compute_attractive_force(current_pos, goal_pos)
        f_rep = self.compute_repulsive_force(current_pos, detected_obstacles)

        # Total combined force vector
        f_total = f_att + f_rep

        # Velocity damping along force direction to prevent oscillation
        damping_force = -0.35 * current_vel
        a_cmd = f_total + damping_force

        # Bound acceleration
        a_mag = np.linalg.norm(a_cmd)
        if a_mag > self.max_accel:
            a_cmd = (a_cmd / a_mag) * self.max_accel

        return {
            "acceleration_command_ms2": a_cmd.tolist(),
            "repulsive_active": np.linalg.norm(f_rep) > 0.1,
            "dist_to_closest_obstacle_m": min([np.linalg.norm(current_pos - np.array([o['x'], o['y'], o['z']])) for o in detected_obstacles]) if detected_obstacles else float('inf')
        }

if __name__ == "__main__":
    apf = ArtificialPotentialFieldDAA()

    drone_pos = np.array([100.0, 100.0, 25.0])
    drone_vel = np.array([15.0, 0.0, 0.0])
    waypoint_target = np.array([200.0, 100.0, 25.0])

    # Uncooperative dynamic intruder detected 25m directly in flight path
    radar_obstacles = [
        {"id": "INTRUDER_01", "x": 125.0, "y": 102.0, "z": 25.0}
    ]

    steering = apf.compute_reactive_steering(drone_pos, drone_vel, waypoint_target, radar_obstacles)

    print("==========================================================================")
    print("      ARTIFICIAL POTENTIAL FIELD (APF) DAA STEERING TEST                  ")
    print("==========================================================================")
    print(f"Closest Obstacle Range      : {steering['dist_to_closest_obstacle_m']:.2f} m")
    print(f"Repulsive Field Engaged     : {steering['repulsive_active']}")
    print(f"Steering Acceleration (m/s²): [Ax: {steering['acceleration_command_ms2'][0]:+.2f}, "
          f"Ay: {steering['acceleration_command_ms2'][1]:+.2f}, Az: {steering['acceleration_command_ms2'][2]:+.2f}]")
    print("==========================================================================\n")