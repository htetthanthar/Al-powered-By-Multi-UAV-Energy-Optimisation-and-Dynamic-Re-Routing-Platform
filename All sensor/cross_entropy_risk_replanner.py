import math
import numpy as np
from typing import Dict, List, Tuple

class CrossEntropyTrajectoryReplanner:
    def __init__(
        self,
        horizon_steps: int = 12,
        num_samples: int = 400,
        elite_fraction: float = 0.10,
        max_iterations: int = 20,
        smoothing_alpha: float = 0.75,
        dt_seconds: float = 2.0
    ):
        self.H = horizon_steps
        self.N = num_samples
        self.n_elite = int(num_samples * elite_fraction)
        self.max_iter = max_iterations
        self.alpha = smoothing_alpha
        self.dt = dt_seconds

    def evaluate_trajectory_cost(
        self,
        control_seq: np.ndarray,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        dynamic_obstacles: List[Dict]
    ) -> float:
        """
        Evaluates composite cost = Goal Distance + Total Flight Energy + Risk Field Penalty
        control_seq: (H, 3) velocity vectors [v_x, v_y, v_z]
        """
        pos = start_pos.copy()
        total_cost = 0.0
        p_base_watts = 280.0

        for t in range(self.H):
            v_cmd = control_seq[t]
            pos = pos + v_cmd * self.dt

            # 1. Energy consumption penalty: P ~ P0 + k*||v||^3
            v_mag = float(np.linalg.norm(v_cmd))
            p_inst = p_base_watts + 0.18 * (v_mag ** 3)
            total_cost += p_inst * self.dt

            # 2. Dynamic Obstacle / TFR Repulsive Potential Field
            for obs in dynamic_obstacles:
                obs_center = np.array([obs["x"], obs["y"], obs["z"]])
                dist = float(np.linalg.norm(pos - obs_center))
                r_safe = obs["radius_m"]
                if dist < r_safe:
                    # Non-linear barrier penalty
                    total_cost += 50000.0 * ((r_safe - dist) / r_safe) ** 2

        # 3. Terminal Goal Reaching Penalty
        dist_to_goal = float(np.linalg.norm(pos - goal_pos))
        total_cost += 150.0 * (dist_to_goal ** 2)

        return total_cost

    def optimize_trajectory(
        self,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        dynamic_obstacles: List[Dict]
    ) -> Dict:
        """
        Executes CEM distribution updates over velocity vector sequences.
        """
        p_start = np.array(start_pos)
        p_goal = np.array(goal_pos)

        # Initialize Gaussian distribution parameters: mu, sigma
        nominal_vel = (p_goal - p_start) / (self.H * self.dt)
        mu = np.tile(nominal_vel, (self.H, 1)) # Shape (H, 3)
        sigma = np.ones((self.H, 3)) * 4.0     # Initial exploration variance

        best_cost_history = []
        best_trajectory_controls = mu.copy()

        for iteration in range(self.max_iter):
            # 1. Sample N candidate trajectories from N(mu, sigma)
            samples = np.random.normal(
                loc=np.tile(mu, (self.N, 1, 1)),
                scale=np.tile(sigma, (self.N, 1, 1))
            )

            # 2. Evaluate candidate costs
            costs = np.zeros(self.N)
            for i in range(self.N):
                costs[i] = self.evaluate_trajectory_cost(samples[i], p_start, p_goal, dynamic_obstacles)

            # 3. Select elite candidates
            elite_indices = np.argsort(costs)[:self.n_elite]
            elite_samples = samples[elite_indices]
            best_cost = float(costs[elite_indices[0]])
            best_cost_history.append(best_cost)

            # 4. Update distribution statistics with smoothing
            mu_elite = np.mean(elite_samples, axis=0)
            sigma_elite = np.std(elite_samples, axis=0) + 1e-3

            mu = self.alpha * mu_elite + (1.0 - self.alpha) * mu
            sigma = self.alpha * sigma_elite + (1.0 - self.alpha) * sigma
            best_trajectory_controls = elite_samples[0]

            if sigma.max() < 0.20: # Convergence threshold
                break

        # Compute optimal 3D spatial points
        optimized_path = [p_start.tolist()]
        curr = p_start.copy()
        for t in range(self.H):
            curr = curr + best_trajectory_controls[t] * self.dt
            optimized_path.append([round(float(c), 2) for c in curr])

        terminal_dist = float(np.linalg.norm(np.array(optimized_path[-1]) - p_goal))

        return {
            "cem_iterations_executed": len(best_cost_history),
            "initial_sample_cost": round(best_cost_history[0], 2),
            "final_optimized_cost": round(best_cost_history[-1], 2),
            "terminal_error_to_goal_m": round(terminal_dist, 2),
            "replanning_converged": terminal_dist <= 2.5,
            "sampled_path_3d": optimized_path
        }

if __name__ == "__main__":
    replanner = CrossEntropyTrajectoryReplanner(horizon_steps=10, num_samples=300, dt_seconds=2.0)

    start = (0.0, 0.0, 25.0)
    goal = (300.0, 300.0, 25.0)

    # Pop-up obstacle directly blocking line of sight at (150, 150)
    obstacles = [{"id": "DYNAMIC_STORM_CELL", "x": 150.0, "y": 150.0, "z": 25.0, "radius_m": 45.0}]

    plan = replanner.optimize_trajectory(start, goal, obstacles)

    print("==========================================================================")
    print("      CROSS-ENTROPY METHOD (CEM) DYNAMIC RISK REPLANNING AUDIT            ")
    print("==========================================================================")
    for k, v in plan.items():
        if k != "sampled_path_3d":
            print(f"  • {k:35s}: {v}")
    print(f"\nGenerated Replanned Trajectory Nodes ({len(plan['sampled_path_3d'])} waypoints):")
    for idx, pt in enumerate(plan["sampled_path_3d"][:6]):
        print(f"  Node {idx:02d}: {pt}")
    print("  ...")
    print("==========================================================================\n")