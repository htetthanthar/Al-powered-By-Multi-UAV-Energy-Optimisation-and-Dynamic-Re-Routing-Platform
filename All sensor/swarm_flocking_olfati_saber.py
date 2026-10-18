import math
import numpy as np
from typing import Dict, Tuple, List

class OlfatiSaberSwarmFlockingEngine:
    def __init__(
        self,
        num_agents: int = 6,
        desired_inter_agent_distance_m: float = 12.0,
        interaction_radius_m: float = 22.0,
        eps_sigma_norm: float = 0.10,
        bump_parameter_h: float = 0.20,
        max_acceleration_ms2: float = 4.0
    ):
        self.N = num_agents
        self.d = desired_inter_agent_distance_m
        self.r = interaction_radius_m
        self.eps = eps_sigma_norm
        self.h = bump_parameter_h
        self.a_max = max_acceleration_ms2

        # Transformed sigma-norm parameters
        self.d_sigma = self._sigma_norm(self.d)
        self.r_sigma = self._sigma_norm(self.r)

        # Gains
        self.c1_nav = 0.65 # Navigational feedback to group goal
        self.c2_vel = 0.85 # Velocity tracking to group goal

    def _sigma_norm(self, z: float) -> float:
        """Smooth sigma-norm: ||z||_sigma = 1/eps * (sqrt(1 + eps*z^2) - 1)"""
        return (1.0 / self.eps) * (math.sqrt(1.0 + self.eps * (z ** 2)) - 1.0)

    def _sigma_grad(self, z_vec: np.ndarray) -> np.ndarray:
        """Gradient of sigma-norm: sigma_grad(z) = z / sqrt(1 + eps*||z||^2)"""
        norm_sq = np.sum(z_vec ** 2)
        return z_vec / math.sqrt(1.0 + self.eps * norm_sq)

    def _bump_function(self, z: float) -> float:
        """Smooth cutoff bump function rho_h(z) in [0, 1]."""
        if z < self.h:
            return 1.0
        elif z <= 1.0:
            return 0.5 * (1.0 + math.cos(math.pi * (z - self.h) / (1.0 - self.h)))
        else:
            return 0.0

    def _phi_action(self, z_sigma: float) -> float:
        """Pairwise potential derivative phi(z) = 0.5 * ((a+b)*sigma_1(z+c) + (a-b))."""
        # Attractive-Repulsive potential with zero crossing at d_sigma
        a_param = 5.0
        b_param = 5.0
        c_shift = abs(a_param - b_param) / math.sqrt(4.0 * a_param * b_param)
        
        diff = z_sigma - self.d_sigma
        sig1 = diff / math.sqrt(1.0 + diff ** 2)
        return float(sig1)

    def step_swarm_flocking_dynamics(
        self,
        dt_seconds: float,
        agent_positions: np.ndarray,   # Shape (N, 3)
        agent_velocities: np.ndarray,  # Shape (N, 3)
        goal_position: np.ndarray,     # Shape (3,)
        goal_velocity: np.ndarray      # Shape (3,)
    ) -> Dict:
        """
        Computes decentralized spatial potential gradients and velocity alignment.
        """
        N = self.N
        accel_cmds = np.zeros((N, 3))
        active_edges_count = 0
        min_inter_agent_dist = float('inf')

        # Compute pair-wise flocking interactions
        for i in range(N):
            qi = agent_positions[i]
            pi = agent_velocities[i]
            u_interaction = np.zeros(3)

            for j in range(N):
                if i == j:
                    continue
                qj = agent_positions[j]
                pj = agent_velocities[j]

                dist_ij = float(np.linalg.norm(qj - qi))
                min_inter_agent_dist = min(min_inter_agent_dist, dist_ij)

                if dist_ij < self.r:
                    active_edges_count += 1
                    # 1. Spatial Gradient Potential Force
                    dist_sigma = self._sigma_norm(dist_ij)
                    n_ij = self._sigma_grad(qj - qi)
                    rho_val = self._bump_function(dist_sigma / self.r_sigma)
                    phi_val = self._phi_action(dist_sigma)
                    f_gradient = rho_val * phi_val * n_ij

                    # 2. Velocity Alignment Damping Force
                    a_ij = rho_val
                    f_velocity = a_ij * (pj - pi)

                    u_interaction += (f_gradient + f_velocity)

            # 3. Navigational Feedback to Swarm Goal Target
            f_nav = -self.c1_nav * (qi - goal_position) - self.c2_vel * (pi - goal_velocity)

            # Total Commanded Acceleration
            u_total = u_interaction + f_nav
            accel_mag = np.linalg.norm(u_total)
            if accel_mag > self.a_max:
                u_total = (u_total / accel_mag) * self.a_max

            accel_cmds[i] = u_total

        # Integrate kinematics forward (Euler Step)
        new_velocities = agent_velocities + accel_cmds * dt_seconds
        new_positions = agent_positions + new_velocities * dt_seconds

        # Group Cohesion Metrics: Swarm centroid and velocity variance
        centroid_pos = np.mean(new_positions, axis=0)
        centroid_vel = np.mean(new_velocities, axis=0)
        velocity_dispersion = float(np.mean(np.linalg.norm(new_velocities - centroid_vel, axis=1)))

        return {
            "dt_seconds": dt_seconds,
            "swarm_agents_count": N,
            "connected_spatial_edges": active_edges_count // 2,
            "min_inter_agent_distance_m": round(min_inter_agent_dist, 2),
            "swarm_centroid_pos_m": (round(float(centroid_pos[0]), 2),
                                     round(float(centroid_pos[1]), 2),
                                     round(float(centroid_pos[2]), 2)),
            "velocity_consensus_dispersion_ms": round(velocity_dispersion, 3),
            "collision_free_lattice_maintained": min_inter_agent_dist >= (self.d * 0.55),
            "flocking_status": "STABLE_QUASI_LATTICE_CONVERGED" if velocity_dispersion < 0.25 else "DYNAMIC_FORMATION_ALIGNING"
        }

if __name__ == "__main__":
    flock_engine = OlfatiSaberSwarmFlockingEngine(
        num_agents=5, desired_inter_agent_distance_m=12.0, interaction_radius_m=22.0
    )

    print("==========================================================================")
    print("      DECENTRALIZED SWARM FLOCKING & OLFATI-SABER CONSENSUS AUDIT         ")
    print("==========================================================================")

    # Initialize 5 drones randomly dispersed around origin with velocity perturbation
    np.random.seed(42)
    pos_init = np.random.uniform(-15.0, 15.0, (5, 3))
    pos_init[:, 2] = -25.0 # Cruising at 25m MSL
    vel_init = np.random.uniform(-1.0, 1.0, (5, 3))
    vel_init[:, 0] += 12.0 # General forward cruise

    p_goal = np.array([200.0, 0.0, -25.0])
    v_goal = np.array([15.0, 0.0, 0.0])

    p_curr = pos_init.copy()
    v_curr = vel_init.copy()

    # Simulate 8 seconds of decentralized flocking (dt = 0.5s)
    for step in range(1, 9):
        audit = flock_engine.step_swarm_flocking_dynamics(
            dt_seconds=0.5,
            agent_positions=p_curr,
            agent_velocities=v_curr,
            goal_position=p_goal,
            goal_velocity=v_goal
        )
        print(f"T+{step*0.5:04.1f}s | Min Dist: {audit['min_inter_agent_distance_m']:5.2f}m | "
              f"Centroid: {audit['swarm_centroid_pos_m']} | Vel Dispersion: {audit['velocity_consensus_dispersion_ms']:5.3f} m/s | "
              f"Edges: {audit['connected_spatial_edges']:2d} | Status: {audit['flocking_status']}")
        
        # Advance state
        p_curr = p_curr + v_curr * 0.5
    print("==========================================================================\n")