import math
import numpy as np
from typing import Dict, Tuple, List

class IterativeLQREmergencyAutoLandPlanner:
    def __init__(
        self,
        horizon_steps: int = 30,
        dt_seconds: float = 0.10,
        max_ilqr_iterations: int = 20,
        uav_mass_kg: float = 6.14,
        max_thrust_n: float = 85.0
    ):
        self.N = horizon_steps
        self.dt = dt_seconds
        self.max_iter = max_ilqr_iterations
        self.m = uav_mass_kg
        self.max_t = max_thrust_n
        self.g = 9.80665

        # State: x = [px, py, pz, vx, vy, vz]^T (6x1)
        # Control: u = [pitch, roll, thrust]^T (3x1)
        self.nx = 6
        self.nu = 3

        # Cost Weighting Matrices
        self.Q_stage = np.diag([10.0, 10.0, 25.0, 1.0, 1.0, 5.0])
        self.R_stage = np.diag([0.5, 0.5, 0.005])
        self.Q_final = np.diag([150.0, 150.0, 500.0, 50.0, 50.0, 200.0])

    def _dynamics_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Nonlinear translational quadrotor dynamics."""
        pos = x[0:3]
        vel = x[3:6]
        pitch, roll, thrust = u[0], u[1], u[2]

        ax = (thrust / self.m) * math.sin(pitch)
        ay = -(thrust / self.m) * math.sin(roll)
        az = (thrust / self.m) * (math.cos(pitch) * math.cos(roll)) - self.g

        next_pos = pos + vel * self.dt + 0.5 * np.array([ax, ay, az]) * (self.dt ** 2)
        next_vel = vel + np.array([ax, ay, az]) * self.dt
        return np.concatenate([next_pos, next_vel])

    def _get_dynamics_jacobians(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analytic continuous-to-discrete Jacobians F_x (6x6) and F_u (6x3)."""
        pitch, roll, thrust = u[0], u[1], u[2]
        
        # State Jacobian F_x
        Fx = np.eye(6)
        Fx[0:3, 3:6] = np.eye(3) * self.dt

        # Control Jacobian F_u
        Fu = np.zeros((6, 3))
        # d(ax)/d(u)
        d_ax_dpitch = (thrust / self.m) * math.cos(pitch)
        d_ax_dthrust = math.sin(pitch) / self.m

        # d(ay)/d(u)
        d_ay_droll = -(thrust / self.m) * math.cos(roll)
        d_ay_dthrust = -math.sin(roll) / self.m

        # d(az)/d(u)
        d_az_dpitch = -(thrust / self.m) * math.sin(pitch) * math.cos(roll)
        d_az_droll = -(thrust / self.m) * math.cos(pitch) * math.sin(roll)
        d_az_dthrust = (math.cos(pitch) * math.cos(roll)) / self.m

        B_accel = np.array([
            [d_ax_dpitch, 0.0, d_ax_dthrust],
            [0.0, d_ay_droll, d_ay_dthrust],
            [d_az_dpitch, d_az_droll, d_az_dthrust]
        ])

        Fu[0:3, :] = 0.5 * (self.dt ** 2) * B_accel
        Fu[3:6, :] = self.dt * B_accel
        return Fx, Fu

    def compute_emergency_landing_trajectory(
        self,
        x_initial: np.ndarray,
        landing_site_target_pos_ned: np.ndarray # [x_land, y_land, z_land]
    ) -> Dict:
        """
        Executes iLQR Riccati backward-forward iteration to synthesize descent trajectory.
        """
        target_final_state = np.concatenate([landing_site_target_pos_ned, np.array([0.0, 0.0, 0.5])]) # 0.5 m/s touchdown

        # Initialize trajectory rollout with nominal hover thrust
        U = np.zeros((self.N, self.nu))
        U[:, 2] = self.m * self.g

        X = np.zeros((self.N + 1, self.nx))
        X[0] = x_initial
        for k in range(self.N):
            X[k + 1] = self._dynamics_step(X[k], U[k])

        # Compute initial baseline cost
        total_cost = 0.0
        for k in range(self.N):
            total_cost += float(0.5 * (X[k] - target_final_state).T @ self.Q_stage @ (X[k] - target_final_state) +
                                0.5 * U[k].T @ self.R_stage @ U[k])
        total_cost += float(0.5 * (X[-1] - target_final_state).T @ self.Q_final @ (X[-1] - target_final_state))

        # iLQR Optimization Loop
        for iteration in range(self.max_iter):
            # 1. Backward Riccati Pass
            V_x = self.Q_final @ (X[-1] - target_final_state)
            V_xx = self.Q_final.copy()

            d_seq = np.zeros((self.N, self.nu))
            K_seq = np.zeros((self.N, self.nu, self.nx))

            for k in range(self.N - 1, -1, -1):
                Fx, Fu = self._get_dynamics_jacobians(X[k], U[k])

                q_x = self.Q_stage @ (X[k] - target_final_state) + Fx.T @ V_x
                q_u = self.R_stage @ U[k] + Fu.T @ V_x

                Q_xx = self.Q_stage + Fx.T @ V_xx @ Fx
                Q_uu = self.R_stage + Fu.T @ V_xx @ Fu + 1e-4 * np.eye(self.nu)
                Q_ux = Fu.T @ V_xx @ Fx

                # Invert Q_uu
                Q_uu_inv = np.linalg.inv(Q_uu)
                d_k = -Q_uu_inv @ q_u
                K_k = -Q_uu_inv @ Q_ux

                d_seq[k] = d_k
                K_seq[k] = K_k

                # Update Value Function approximations
                V_x = q_x + K_k.T @ Q_uu @ d_k + K_k.T @ q_u + Q_ux.T @ d_k
                V_xx = Q_xx + K_k.T @ Q_uu @ K_k + K_k.T @ Q_ux + Q_ux.T @ K_k

            # 2. Forward Pass with Armijo Line Search
            alpha = 1.0
            improved = False

            for _ in range(6):
                X_new = np.zeros_like(X)
                U_new = np.zeros_like(U)
                X_new[0] = x_initial

                for k in range(self.N):
                    u_cmd = U[k] + alpha * d_seq[k] + K_seq[k] @ (X_new[k] - X[k])
                    # Enforce actuator constraints
                    u_cmd[0] = np.clip(u_cmd[0], -math.radians(35.0), math.radians(35.0))
                    u_cmd[1] = np.clip(u_cmd[1], -math.radians(35.0), math.radians(35.0))
                    u_cmd[2] = np.clip(u_cmd[2], 0.0, self.max_t)
                    
                    U_new[k] = u_cmd
                    X_new[k + 1] = self._dynamics_step(X_new[k], u_cmd)

                # Evaluate new cost
                cost_new = 0.0
                for k in range(self.N):
                    cost_new += float(0.5 * (X_new[k] - target_final_state).T @ self.Q_stage @ (X_new[k] - target_final_state) +
                                      0.5 * U_new[k].T @ self.R_stage @ U_new[k])
                cost_new += float(0.5 * (X_new[-1] - target_final_state).T @ self.Q_final @ (X_new[-1] - target_final_state))

                if cost_new < total_cost:
                    total_cost = cost_new
                    X = X_new
                    U = U_new
                    improved = True
                    break
                alpha *= 0.5

            if not improved or np.max(np.abs(d_seq)) < 1e-3:
                break

        # Check Touchdown Residual Bounds
        touchdown_pos_err = float(np.linalg.norm(X[-1, 0:3] - landing_site_target_pos_ned))
        touchdown_vz = float(X[-1, 5])

        return {
            "ilqr_iterations_executed": iteration + 1,
            "optimized_trajectory_duration_s": round(self.N * self.dt, 2),
            "final_touchdown_position_err_m": round(touchdown_pos_err, 3),
            "final_touchdown_vz_ms": round(touchdown_vz, 3),
            "total_trajectory_energy_cost": round(total_cost, 2),
            "autoland_trajectory_feasible": touchdown_pos_err <= 0.35 and touchdown_vz <= 0.90,
            "optimized_initial_action": {
                "pitch_cmd_deg": round(math.degrees(U[0, 0]), 2),
                "roll_cmd_deg": round(math.degrees(U[0, 1]), 2),
                "thrust_cmd_n": round(float(U[0, 2]), 2)
            }
        }

if __name__ == "__main__":
    planner = IterativeLQREmergencyAutoLandPlanner(horizon_steps=30, dt_seconds=0.10)

    print("==========================================================================")
    print("      iLQR / DDP EMERGENCY FORCED AUTO-LANDING TRAJECTORY AUDIT           ")
    print("==========================================================================")

    # Initial state: Drone at (x=45m, y=-20m, alt=30m MSL, vx=14.0 m/s forward speed)
    init_state = np.array([45.0, -20.0, -30.0, 14.0, 0.0, 0.0])
    # Target Vertipad Ground Center at (x=100m, y=0m, alt=0m MSL)
    pad_target = np.array([100.0, 0.0, 0.0])

    plan = planner.compute_emergency_landing_trajectory(init_state, pad_target)

    for k, v in plan.items():
        if k != "optimized_initial_action":
            print(f"  • {k:35s}: {v}")
    print("\nInitial Immediate Optimal Action:")
    for k, v in plan["optimized_initial_action"].items():
        print(f"    - {k:25s}: {v}")
    print("==========================================================================\n")