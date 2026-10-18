import math
import numpy as np
from typing import Dict, Tuple

class PropulsionLumpedThermalNetwork:
    def __init__(
        self,
        phase_resistance_r0_ohms: float = 0.065, # Base phase resistance at 20°C
        copper_temp_coeff: float = 0.00393,      # alpha_Cu per °C
        mosfet_rds_on_ohms: float = 0.0025,
        switching_frequency_hz: float = 24000.0, # 24 kHz FOC switching
        demagnetization_temp_limit_c: float = 120.0,
        mosfet_max_junction_temp_c: float = 150.0
    ):
        self.r0 = phase_resistance_r0_ohms
        self.alpha_cu = copper_temp_coeff
        self.rds = mosfet_rds_on_ohms
        self.f_sw = switching_frequency_hz
        self.t_mag_limit = demagnetization_temp_limit_c
        self.t_mos_limit = mosfet_max_junction_temp_c

        # State Vector: T = [T_winding, T_stator, T_magnet, T_mosfet]^T in °C
        self.T = np.array([28.0, 28.0, 28.0, 32.0])

        # Thermal Capacitance Matrix C_th (J/K)
        self.C = np.array([45.0, 120.0, 60.0, 8.5])

        # Thermal Conductance Conductances G_th (W/K)
        self.g_ws = 2.8   # Winding to Stator Core
        self.g_sa = 1.4   # Stator Core to Ambient Airflow
        self.g_sm = 0.65  # Stator to Rotor Magnets across Air Gap
        self.g_ma = 1.1   # Rotor Magnets to Ambient Airflow
        self.g_ja = 0.85  # Inverter MOSFET Junction to Heatsink/Air

    def compute_instantaneous_losses(
        self,
        phase_current_rms_a: float,
        rotor_speed_rad_s: float,
        battery_voltage_v: float
    ) -> np.ndarray:
        """
        Calculates heat generation across subsystems: [Q_winding, Q_stator, Q_magnet, Q_mosfet]
        """
        t_w = self.T[0]
        t_j = self.T[3]

        # 1. Temperature-dependent copper winding resistive loss
        r_phase = self.r0 * (1.0 + self.alpha_cu * (t_w - 20.0))
        q_copper = 3.0 * (phase_current_rms_a ** 2) * r_phase

        # 2. Stator core iron eddy current and hysteresis losses: Q_iron ~ k_h * omega + k_e * omega^2
        q_iron = (0.012 * rotor_speed_rad_s) + (0.000085 * (rotor_speed_rad_s ** 2))

        # 3. Magnet rotor surface eddy losses
        q_magnet = 0.05 * q_iron

        # 4. Inverter ESC conduction and switching losses
        p_cond = (phase_current_rms_a ** 2) * self.rds * (1.0 + 0.005 * (t_j - 25.0))
        p_sw = self.f_sw * 1.8e-6 * battery_voltage_v * phase_current_rms_a
        q_mosfet = p_cond + p_sw

        return np.array([q_copper, q_iron, q_magnet, q_mosfet])

    def step_thermal_integration(
        self,
        dt_seconds: float,
        phase_current_rms_a: float,
        rotor_speed_rad_s: float,
        ambient_temp_c: float = 32.0,
        battery_voltage_v: float = 22.8
    ) -> Dict:
        """
        Integrates thermal state differential equations via 4th-order Runge-Kutta (RK4).
        """
        def derivatives(t_vec: np.ndarray) -> np.ndarray:
            tw, ts, tm, tj = t_vec[0], t_vec[1], t_vec[2], t_vec[3]
            q_losses = self.compute_instantaneous_losses(phase_current_rms_a, rotor_speed_rad_s, battery_voltage_v)

            dtw = (q_losses[0] - self.g_ws * (tw - ts)) / self.C[0]
            dts = (q_losses[1] + self.g_ws * (tw - ts) - self.g_sa * (ts - ambient_temp_c) - self.g_sm * (ts - tm)) / self.C[1]
            dtm = (q_losses[2] + self.g_sm * (ts - tm) - self.g_ma * (tm - ambient_temp_c)) / self.C[2]
            dtj = (q_losses[3] - self.g_ja * (tj - ambient_temp_c)) / self.C[3]

            return np.array([dtw, dts, dtm, dtj])

        # RK4 Integration
        k1 = derivatives(self.T)
        k2 = derivatives(self.T + 0.5 * dt_seconds * k1)
        k3 = derivatives(self.T + 0.5 * dt_seconds * k2)
        k4 = derivatives(self.T + dt_seconds * k3)

        self.T += (dt_seconds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        mag_overtemp = self.T[2] >= self.t_mag_limit
        mos_overtemp = self.T[3] >= self.t_mos_limit

        status = "THERMALLY_NOMINAL"
        if mag_overtemp:
            status = "CRITICAL_ROTOR_MAGNET_DEMAGNETIZATION_RISK"
        elif mos_overtemp:
            status = "CRITICAL_MOSFET_JUNCTION_OVERTEMPERATURE"
        elif self.T[0] >= 105.0:
            status = "WARNING_STATOR_WINDING_THERMAL_SATURATION"

        return {
            "winding_temperature_c": round(float(self.T[0]), 2),
            "stator_core_temperature_c": round(float(self.T[1]), 2),
            "rotor_magnet_temperature_c": round(float(self.T[2]), 2),
            "inverter_mosfet_temperature_c": round(float(self.T[3]), 2),
            "rotor_magnet_margin_c": round(float(self.t_mag_limit - self.T[2]), 2),
            "inverter_thermal_margin_c": round(float(self.t_mos_limit - self.T[3]), 2),
            "thermal_health_status": status,
            "throttle_reduction_required": mag_overtemp or mos_overtemp
        }

if __name__ == "__main__":
    thermal_net = PropulsionLumpedThermalNetwork()

    print("==========================================================================")
    print("      ELECTRIC MOTOR & INVERTER LUMPED PARAMETER THERMAL TRANSIENT AUDIT  ")
    print("==========================================================================")

    # Simulate 60 seconds of heavy climb (22.0A RMS current per motor, 620 rad/s rotor speed)
    for t_sec in range(1, 11):
        # 6-second bulk steps
        state = thermal_net.step_thermal_integration(
            dt_seconds=6.0, phase_current_rms_a=22.0, rotor_speed_rad_s=620.0, ambient_temp_c=32.0
        )
        print(f"T+{t_sec*6:02d}s | Winding: {state['winding_temperature_c']:5.1f}°C | Stator: {state['stator_core_temperature_c']:5.1f}°C | "
              f"Magnets: {state['rotor_magnet_temperature_c']:5.1f}°C | Inverter: {state['inverter_mosfet_temperature_c']:5.1f}°C | Status: {state['thermal_health_status']}")
    print("==========================================================================\n")