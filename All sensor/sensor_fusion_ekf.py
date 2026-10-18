import math
import numpy as np
from typing import Tuple, Dict

class UAVStateEstimatorEKF:
    def __init__(self, dt: float = 0.05):
        """
        State Vector x (9x1):
          x[0]: Position North (p_x, m)
          x[1]: Position East  (p_y, m)
          x[2]: Position Down  (p_z, m)
          x[3]: Velocity North (v_x, m/s)
          x[4]: Velocity East  (v_y, m/s)
          x[5]: Velocity Down  (v_z, m/s)
          x[6]: Wind Vector East-West (w_x, m/s)
          x[7]: Wind Vector North-South (w_y, m/s)
          x[8]: Barometer Bias (b_z, m)
        """
        self.dt = dt
        self.x = np.zeros((9, 1))
        
        # State Covariance Matrix (P)
        self.P = np.eye(9) * 1.0
        self.P[6, 6] = 5.0  # Higher initial uncertainty for wind
        self.P[7, 7] = 5.0
        self.P[8, 8] = 2.0

        # Process Noise Covariance (Q)
        self.Q = np.diag([
            0.01, 0.01, 0.01,   # Position noise
            0.08, 0.08, 0.08,   # Velocity noise (accelerometer integration)
            0.02, 0.02,         # Wind variation rate
            0.001               # Barometer bias drift
        ])

        # Measurement Noise Covariances (R)
        self.R_gps_pos = np.diag([1.5**2, 1.5**2])        # GPS Horizontal Position: 1.5m SD
        self.R_gps_vel = np.diag([0.2**2, 0.2**2])        # GPS Horizontal Velocity: 0.2 m/s SD
        self.R_baro = np.array([[0.8**2]])                # Barometer Altitude: 0.8m SD
        self.R_airspeed = np.array([[0.5**2]])            # Pitot Airspeed: 0.5 m/s SD

    def predict(self, accel_body_ned: np.ndarray):
        """
        Time update (prediction) using IMU linear acceleration inputs in NED frame.
        accel_body_ned: np.ndarray shape (3,) [a_x, a_y, a_z] (gravity compensated)
        """
        F = np.eye(9)
        # Position propagation from velocity
        F[0, 3] = self.dt
        F[1, 4] = self.dt
        F[2, 5] = self.dt

        # State transition: x_{k|k-1} = F * x_{k-1} + B * u
        self.x[0:3] += self.x[3:6] * self.dt + 0.5 * accel_body_ned.reshape(3, 1) * (self.dt ** 2)
        self.x[3:6] += accel_body_ned.reshape(3, 1) * self.dt

        # Covariance propagation: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, gps_pos_xy: np.ndarray, gps_vel_xy: np.ndarray):
        """
        Measurement update using GPS horizontal position and velocity.
        """
        z = np.vstack([gps_pos_xy.reshape(2, 1), gps_vel_xy.reshape(2, 1)])
        
        H = np.zeros((4, 9))
        H[0, 0] = 1.0  # p_x
        H[1, 1] = 1.0  # p_y
        H[2, 3] = 1.0  # v_x
        H[3, 4] = 1.0  # v_y

        R = np.block([
            [self.R_gps_pos, np.zeros((2, 2))],
            [np.zeros((2, 2)), self.R_gps_vel]
        ])

        y = z - H @ self.x  # Innovation residual
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P

    def update_barometer(self, baro_alt_m: float):
        """
        Measurement update using Barometric Altitude (z_down = -altitude).
        """
        z = np.array([[-baro_alt_m]])
        H = np.zeros((1, 9))
        H[0, 2] = 1.0  # p_z
        H[0, 8] = 1.0  # Baro bias

        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_baro
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P

    def update_airspeed(self, pitot_airspeed_ms: float):
        """
        Non-linear measurement update: V_air = sqrt((v_x - w_x)^2 + (v_y - w_y)^2 + v_z^2)
        """
        vx, vy, vz = self.x[3, 0], self.x[4, 0], self.x[5, 0]
        wx, wy = self.x[6, 0], self.x[7, 0]

        v_rel_x = vx - wx
        v_rel_y = vy - wy
        v_air_pred = math.sqrt(max(v_rel_x**2 + v_rel_y**2 + vz**2, 1e-4))

        # Jacobian H of non-linear airspeed measurement function h(x)
        H = np.zeros((1, 9))
        H[0, 3] = v_rel_x / v_air_pred   # d(V_air)/d(v_x)
        H[0, 4] = v_rel_y / v_air_pred   # d(V_air)/d(v_y)
        H[0, 5] = vz / v_air_pred        # d(V_air)/d(v_z)
        H[0, 6] = -v_rel_x / v_air_pred  # d(V_air)/d(w_x)
        H[0, 7] = -v_rel_y / v_air_pred  # d(V_air)/d(w_y)

        y = np.array([[pitot_airspeed_ms - v_air_pred]])
        S = H @ self.P @ H.T + self.R_airspeed
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P

    def get_estimated_state(self) -> Dict[str, float]:
        ground_speed = math.sqrt(self.x[3, 0]**2 + self.x[4, 0]**2)
        wind_speed = math.sqrt(self.x[6, 0]**2 + self.x[7, 0]**2)
        wind_heading_deg = math.degrees(math.atan2(self.x[7, 0], self.x[6, 0])) % 360.0

        return {
            "pos_north_m": round(float(self.x[0, 0]), 2),
            "pos_east_m": round(float(self.x[1, 0]), 2),
            "altitude_m": round(-float(self.x[2, 0]), 2),
            "ground_speed_ms": round(ground_speed, 2),
            "ground_speed_kmh": round(ground_speed * 3.6, 2),
            "estimated_wind_u_ms": round(float(self.x[6, 0]), 2),
            "estimated_wind_v_ms": round(float(self.x[7, 0]), 2),
            "estimated_wind_magnitude_kmh": round(wind_speed * 3.6, 2),
            "estimated_wind_direction_deg": round(wind_heading_deg, 1)
        }

if __name__ == "__main__":
    ekf = UAVStateEstimatorEKF(dt=0.05)
    
    # Simulate a quadcopter flying North at 18 m/s against a 4 m/s East-West wind
    print("==========================================================================")
    print("      EXTENDED KALMAN FILTER: SENSOR FUSION & WIND ESTIMATION TEST        ")
    print("==========================================================================")
    
    true_wind_u, true_wind_v = 4.0, -2.0  # Actual wind ground truth

    for step in range(100):
        # 1. IMU forward acceleration
        accel_imu = np.array([0.0, 0.0, 0.0]) # Constant velocity cruise
        ekf.predict(accel_imu)

        # 2. GPS measurement (every 0.2s / 5Hz)
        if step % 4 == 0:
            gps_pos = np.array([18.0 * (step * 0.05) + np.random.normal(0, 0.5), 0.0 + np.random.normal(0, 0.5)])
            gps_vel = np.array([18.0 + np.random.normal(0, 0.1), 0.0 + np.random.normal(0, 0.1)])
            ekf.update_gps(gps_pos, gps_vel)

        # 3. Barometer altitude (25 meters)
        if step % 2 == 0:
            ekf.update_barometer(25.0 + np.random.normal(0, 0.2))

        # 4. Pitot tube airspeed: V_air = sqrt((18 - 4)^2 + (0 - (-2))^2) = sqrt(14^2 + 2^2) = 14.14 m/s
        true_airspeed = math.sqrt((18.0 - true_wind_u)**2 + (0.0 - true_wind_v)**2)
        ekf.update_airspeed(true_airspeed + np.random.normal(0, 0.3))

    est = ekf.get_estimated_state()
    for k, v in est.items():
        print(f"  • {k:32s}: {v}")
    print("==========================================================================\n")