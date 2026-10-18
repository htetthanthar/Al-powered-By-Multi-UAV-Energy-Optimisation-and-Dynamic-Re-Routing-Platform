import os
import time
from enum import Enum
from typing import Dict, List, Optional

class JetsonPowerMode(Enum):
    MAX_PERFORMANCE_15W = "MODE_15W"
    BALANCED_10W = "MODE_10W"
    ENERGY_SAVING_7W = "MODE_7W"
    EMERGENCY_THROTTLE_5W = "MODE_5W"

class CompanionThermalGovernor:
    def __init__(
        self,
        soc_temp_warn_c: float = 75.0,
        soc_temp_critical_c: float = 85.0,
        poll_interval_s: float = 2.0
    ):
        self.warn_temp = soc_temp_warn_c
        self.crit_temp = soc_temp_critical_c
        self.poll_interval = poll_interval_s
        self.current_mode = JetsonPowerMode.MAX_PERFORMANCE_15W
        self.active_precision = "FP16"
        self.fan_pwm_target = 128  # 0 to 255 (50% speed)

    def read_hardware_thermal_zones(self) -> Dict[str, float]:
        """
        Reads system thermal sensors. In hardware, reads:
        /sys/devices/virtual/thermal/thermal_zone*/temp
        """
        # Simulated thermal zone telemetry under ambient 32°C conditions
        # SoC core temp fluctuates with active inference load
        sim_cpu_temp = 68.5
        sim_gpu_temp = 72.0
        sim_aux_temp = 48.0

        return {
            "cpu_temperature_c": sim_cpu_temp,
            "gpu_temperature_c": sim_gpu_temp,
            "aux_ambient_c": sim_aux_temp,
            "max_core_temperature_c": max(sim_cpu_temp, sim_gpu_temp)
        }

    def evaluate_thermal_policy(self, max_temp_c: float) -> Dict:
        """
        Applies closed-loop thermal throttling rules:
          - T < 70°C: Mode 15W, FP16 Precision, 50% Fan PWM
          - 70°C <= T < 78°C: Mode 10W, FP16 Precision, 80% Fan PWM
          - 78°C <= T < 85°C: Mode 7W, Quantize to INT8, 100% Fan PWM
          - T >= 85°C: Emergency Throttle 5W, Suspend non-critical Vision
        """
        if max_temp_c < 70.0:
            self.current_mode = JetsonPowerMode.MAX_PERFORMANCE_15W
            self.active_precision = "FP16"
            self.fan_pwm_target = 128
            action = "MAINTAIN_NOMINAL_PERFORMANCE"

        elif 70.0 <= max_temp_c < 78.0:
            self.current_mode = JetsonPowerMode.BALANCED_10W
            self.active_precision = "FP16"
            self.fan_pwm_target = 200
            action = "SCALE_FAN_UP_BALANCED_POWER"

        elif 78.0 <= max_temp_c < 85.0:
            self.current_mode = JetsonPowerMode.ENERGY_SAVING_7W
            self.active_precision = "INT8"
            self.fan_pwm_target = 255
            action = "ENGAGE_INT8_QUANTIZATION_THERMAL_BACKOFF"

        else:
            self.current_mode = JetsonPowerMode.EMERGENCY_THROTTLE_5W
            self.active_precision = "INT8"
            self.fan_pwm_target = 255
            action = "CRITICAL_THERMAL_MITIGATION_SUSPEND_HEAVY_INFERENCE"

        return {
            "assigned_power_mode": self.current_mode.value,
            "tensor_inference_precision": self.active_precision,
            "cooling_fan_pwm": self.fan_pwm_target,
            "cooling_fan_speed_pct": round((self.fan_pwm_target / 255.0) * 100.0, 1),
            "governor_action": action
        }

if __name__ == "__main__":
    governor = CompanionThermalGovernor()
    
    print("==========================================================================")
    print("      COMPANION COMPUTER THERMAL & DVFS POWER GOVERNOR AUDIT              ")
    print("==========================================================================")
    
    test_temperatures = [64.0, 73.5, 81.0, 87.5]
    for temp in test_temperatures:
        policy = governor.evaluate_thermal_policy(temp)
        print(f"SoC Core Temp: {temp:4.1f}°C -> Power: {policy['assigned_power_mode']:8s} | "
              f"Precision: {policy['tensor_inference_precision']:4s} | Fan: {policy['cooling_fan_speed_pct']:5.1f}% | Action: {policy['governor_action']}")
    print("==========================================================================\n")