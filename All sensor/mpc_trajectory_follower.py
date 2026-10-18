import math
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple

class DroneTrajectoryNMPC:
    def __init__(
        self,
        horizon_steps: int = 10,
        dt_seconds: float = 0.1,
        total_mass_kg: float = 6.14,
        max_tilt_angle_rad: float = math.radians(35.0),
        max_thrust_n: float = 85.0
    ):
        self.np = horizon_steps
        self.dt = dt_seconds
        self.m = total_mass_kg
        self.g = 9.80665
        self.max_tilt = max_tilt_angle_rad
        self.max_t = max_thrust_n
        self.min_t = total_mass_kg * self.g * 0.3

        # State weighting matrices: Q (Tracking Error), R (Control Effort), Rd (Slew Rate)
        self.q_pos = np.diag([20.0, 20.0, 30.0]) # x, y, z
        self.q_vel = np.diag([2.0, 2.0, 3.0])     # vx, vy, vz
        self.r_ctrl = np.diag([0.05, 0.05, 0.01]) # pitch, roll, thrust

    def forward_dynamics(
        self,
        state: np.ndarray,
        control: np.ndarray,
        wind_vector_ms: np.ndarray
    ) -> np.ndarray:
        """
        Translational dynamic equations:
          x_dot = v_x
          v_x_dot = (T/m) * (cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)) - (1/m)*D_x
          v_z_dot = (T/m) * (cos(theta)*cos(phi)) - g - (1/m)*D_z
        """
        pos = state[0:3]
        vel = state[3:6]
        
        pitch = control[0]
        roll = control[1]
        thrust = control[2]

        # Effective airspeed vector
        v_air = vel - wind_vector_ms
        cd_parasite = 0.38
        a_front = 0.095
        rho = 1.225
        drag_force = 0.5 * rho * (v_air * np.linalg.norm(v_air)) * cd_parasite * a_front

        # Accelerations
        ax = (thrust / self.m) * math.sin(pitch) - (drag_force[0] / self.m)
        ay = -(thrust / self.m) * math.sin(roll) - (drag_force[1] / self.m)
        az = (thrust / self.m) * (math.cos(pitch) * math.cos(roll)) - self.g - (drag_force[2] / self.m)

        next_pos = pos + vel * self.dt + 0.5 * np.array([ax, ay, az]) * (self.dt ** 2)
        next_vel = vel + np.array([ax, ay, az]) * self.dt

        return np.concatenate([next_pos, next_vel])

    def solve_optimal_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
        wind_vector_ms: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves the constrained optimization problem over the prediction horizon N_p.
        current_state: [x, y, z, vx, vy, vz]
        reference_trajectory: Shape (N_p, 6)
        """
        # Initial guess for controls (hover thrust, zero tilt)
        hover_thrust = self.m * self.g
        u_init = np.zeros(self.np * 3)
        u_init[2::3] = hover_thrust

        # Bounds: [-max_tilt, max_tilt], [-max_tilt, max_tilt], [min_t, max_t]
        bounds = []
        for _ in range(self.np):
            bounds.append((-self.max_tilt, self.max_tilt))
            bounds.append((-self.max_tilt, self.max_tilt))
            bounds.append((self.min_t, self.max_t))

        def cost_function(u_flat):
            u_matrix = u_flat.reshape((self.np, 3))
            total_cost = 0.0
            x_cursor = current_state.copy()

            for k in range(self.np):
                u_k = u_matrix[k]
                ref_k = reference_trajectory[k]

                # Propagate state
                x_cursor = self.forward_dynamics(x_cursor, u_k, wind_vector_ms)

                # State tracking penalty
                pos_err = x_cursor[0:3] - ref_k[0:3]
                vel_err = x_cursor[3:6] - ref_k[3:6]
                total_cost += pos_err.T @ self.q_pos @ pos_err
                total_cost += vel_err.T @ self.q_vel @ vel_err

                # Control effort penalty
                ctrl_err = np.array([u_k[0], u_k[1], u_k[2] - hover_thrust])
                total_cost += ctrl_err.T @ self.r_ctrl @ ctrl_err

            return total_cost

        # Minimize with SLSQP
        res = minimize(cost_function, u_init, method="SLSQP", bounds=bounds, options={"maxiter": 25, "ftol": 1e-3})
        optimal_u_seq = res.x.reshape((self.np, 3))
        
        # Return first control action u_0
        return optimal_u_seq[0], optimal_u_seq

if __name__ == "__main__":
    nmpc = DroneTrajectoryNMPC(horizon_steps=10, dt_seconds=0.1)

    # State: [x=0, y=0, z=25, vx=10, vy=0, vz=0]
    initial_state = np.array([0.0, 0.0, 25.0, 10.0, 0.0, 0.0])
    
    # Reference trajectory navigating to [x=50, y=10, z=25] with 4 m/s crosswind
    ref_traj = np.zeros((10, 6))
    for k in range(10):
        ref_traj[k, 0] = initial_state[0] + 18.0 * (k + 1) * 0.1
        ref_traj[k, 1] = initial_state[1] + 2.0 * (k + 1) * 0.1
        ref_traj[k, 2] = 25.0
        ref_traj[k, 3] = 18.0 # 65 km/h cruise target

    wind = np.array([-4.0, 3.0, 0.0])  # Wind vector

    u0, u_seq = nmpc.solve_optimal_control(initial_state, ref_traj, wind)

    print("==========================================================================")
    print("      NONLINEAR MODEL PREDICTIVE CONTROL (NMPC) SOLUTION TEST             ")
    print("==========================================================================")
    print(f"Optimal Pitch Angle (theta) : {math.degrees(u0[0]):.2f}°")
    print(f"Optimal Roll Angle (phi)    : {math.degrees(u0[1]):.2f}°")
    print(f"Optimal Collective Thrust   : {u0[2]:.2f} N (Hover Baseline: {6.14*9.81:.2f} N)")
    print("==========================================================================\n")