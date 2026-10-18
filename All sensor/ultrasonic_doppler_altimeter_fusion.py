import math
import numpy as np
from typing import Dict, Tuple

class UltrasonicDopplerAltimeterFusion:
    def __init__(
        self,
        transducer_freq_hz: float = 40000.0, # 40 kHz ultrasonic carrier
        beam_elevation_angle_deg: float = 45.0, # 4-beam Janus Doppler configuration
        time_of_flight_std_s: float = 1.2e-5,
        doppler_freq_std_hz: float = 15.0,
        atmospheric_pressure_hpa: float = 1013.25
    ):
        self.f0 = transducer_freq_hz
        self.theta_beam = math.radians(beam_elevation_angle_deg)
        self.sigma_tof = time_of_flight_std_s
        self.sigma_fd = doppler_freq_std_hz
        self.p_atm = atmospheric_pressure_hpa

        # Filter State: x = [alt_agl_m, vz_ms, vx_ms, vy_ms]^T
        self.x = np.array([1.5, 0.0, 0.0, 0.0])
        self.P = np.diag([0.02**2, 0.05**2, 0.10**2, 0.10**2])
        self.Q = np.diag([0.001**2, 0.01**2, 0.02**2, 0.02**2])

    def compute_speed_of_sound(self, temp_celsius: float, relative_humidity_pct: float) -> float:
        """
        Calculates exact acoustic propagation velocity corrected for humidity and temperature.
        """
        t = temp_celsius
        h = relative_humidity_pct

        # Buck equation for saturation vapor pressure (hPa)
        e_s = 6.1121 * math.exp((18.678 - t / 234.5) * (t / (257.14 + t)))
        e_v = (h / 100.0) * e_s

        c_dry = 331.3 * math.sqrt(1.0 + t / 273.15)
        c_moist = c_dry * (1.0 + 0.16 * (e_v / self.p_atm))
        return float(c_moist)

    def predict_state(self, dt_seconds: float, imu_accel_ned_ms2: np.ndarray):
        """Propagates vertical and horizontal kinematic state using strapdown IMU."""
        f_matrix = np.eye(4)
        f_matrix[0, 1] = dt_seconds

        g_matrix = np.zeros((4, 3))
        g_matrix[0, 2] = 0.5 * (dt_seconds ** 2)
        g_matrix[1, 2] = dt_seconds
        g_matrix[2, 0] = dt_seconds
        g_matrix[3, 1] = dt_seconds

        self.x = f_matrix @ self.x + g_matrix @ imu_accel_ned_ms2
        self.P = f_matrix @ self.P @ f_matrix.T + self.Q

    def update_acoustic_measurements(
        self,
        tof_seconds: float,
        doppler_shift_x_hz: float,
        doppler_shift_y_hz: float,
        temp_celsius: float = 30.0,
        relative_humidity_pct: float = 80.0
    ) -> Dict:
        """
        Processes acoustic round-trip ToF echo and Janus Doppler shifts.
        """
        c_sound = self.compute_speed_of_sound(temp_celsius, relative_humidity_pct)

        # Measurement model:
        # z = [ ToF, Delta_f_x, Delta_f_y ]^T
        # h(x) = [ 2*alt / c, (2*f0*vx*cos(theta))/c, (2*f0*vy*cos(theta))/c ]^T
        z_meas = np.array([tof_seconds, doppler_shift_x_hz, doppler_shift_y_hz])

        h_pred = np.array([
            (2.0 * self.x[0]) / c_sound,
            (2.0 * self.f0 * self.x[2] * math.cos(self.theta_beam)) / c_sound,
            (2.0 * self.f0 * self.x[3] * math.cos(self.theta_beam)) / c_sound
        ])

        H = np.zeros((3, 4))
        H[0, 0] = 2.0 / c_sound
        H[1, 2] = (2.0 * self.f0 * math.cos(self.theta_beam)) / c_sound
        H[2, 3] = (2.0 * self.f0 * math.cos(self.theta_beam)) / c_sound

        R = np.diag([self.sigma_tof**2, self.sigma_fd**2, self.sigma_fd**2])

        y_residual = z_meas - h_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_residual
        self.P = (np.eye(4) - K @ H) @ self.P

        return {
            "ambient_speed_of_sound_ms": round(c_sound, 2),
            "estimated_alt_agl_m": round(float(self.x[0]), 3),
            "estimated_vz_ms": round(float(self.x[1]), 3),
            "estimated_vx_ground_ms": round(float(self.x[2]), 3),
            "estimated_vy_ground_ms": round(float(self.x[3]), 3),
            "altitude_variance_mm2": round(float(self.P[0, 0]) * 1e6, 2),
            "ground_lock_status": "HIGH_CONFIDENCE_ACOUSTIC_LOCK" if self.x[0] <= 4.5 else "SIGNAL_ATTENUATED"
        }

if __name__ == "__main__":
    alt_fusion = UltrasonicDopplerAltimeterFusion()

    print("==========================================================================")
    print("      ULTRASONIC DOPPLER MICRO-ALTIMETER & SPEED-OF-SOUND FUSION AUDIT   ")
    print("==========================================================================")
    
    # 1. Sound speed sensitivity to tropical humidity (32°C, 85% RH)
    c_tropical = alt_fusion.compute_speed_of_sound(temp_celsius=32.0, relative_humidity_pct=85.0)
    c_standard = alt_fusion.compute_speed_of_sound(temp_celsius=15.0, relative_humidity_pct=0.0)
    print(f"Standard Atmosphere Sound Speed (15°C, 0% RH) : {c_standard:.2f} m/s")
    print(f"Tropical Urban Sound Speed (32°C, 85% RH)     : {c_tropical:.2f} m/s (+{c_tropical - c_standard:.2f} m/s delta)\n")

    # 2. Simulate 5 fusion steps descending to touchdown from 0.80m AGL at 0.15 m/s
    for step in range(1, 6):
        alt_fusion.predict_state(dt_seconds=0.05, imu_accel_ned_ms2=np.array([0.0, 0.0, 0.0]))
        true_h = 0.80 - (step * 0.05 * 0.15)
        sim_tof = (2.0 * true_h) / c_tropical + np.random.normal(0, 1e-5)
        sim_fd_x = (2.0 * 40000.0 * 0.45 * math.cos(math.radians(45))) / c_tropical + np.random.normal(0, 5.0)
        
        report = alt_fusion.update_acoustic_measurements(
            tof_seconds=sim_tof, doppler_shift_x_hz=sim_fd_x, doppler_shift_y_hz=0.0,
            temp_celsius=32.0, relative_humidity_pct=85.0
        )
        print(f"Step {step:02d} | Est AGL: {report['estimated_alt_agl_m']:5.3f} m | Vz: {report['estimated_vz_ms']:+5.3f} m/s | "
              f"Vx: {report['estimated_vx_ground_ms']:+5.2f} m/s | Var: {report['altitude_variance_mm2']:5.2f} mm² | Status: {report['ground_lock_status']}")
    print("==========================================================================\n")