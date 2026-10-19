import time
import math
import numpy as np
from enum import Enum
from typing import Dict, Optional

class WinchState(Enum):
    RETRACTED = "RETRACTED"
    LOWERING = "LOWERING"
    TOUCHDOWN_DETECTED = "TOUCHDOWN_DETECTED"
    RELEASE_PACKAGE = "RELEASE_PACKAGE"
    RETRACTING = "RETRACTING"
    EMERGENCY_SEVER = "EMERGENCY_SEVER"
    FAULT = "FAULT"

class TetherWinchController:
    def __init__(
        self,
        max_cable_length_m: float = 15.0,
        nominal_deploy_speed_ms: float = 0.6,
        stall_current_threshold_a: float = 4.2,
        tension_touchdown_threshold_n: float = 1.5,
        nominal_package_weight_kg: float = 1.8
    ):
        self.max_len = max_cable_length_m
        self.deploy_speed = nominal_deploy_speed_ms
        self.stall_current_a = stall_current_threshold_a
        self.touchdown_tension_n = tension_touchdown_threshold_n
        self.pkg_weight_kg = nominal_package_weight_kg
        
        # State tracking
        self.state = WinchState.RETRACTED
        self.deployed_length_m = 0.0
        self.current_tension_n = nominal_package_weight_kg * 9.81
        self.motor_current_a = 1.2
        self.package_released = False

    def process_winch_step(
        self,
        dt_seconds: float,
        measured_tension_n: float,
        measured_current_a: float,
        drone_alt_agl_m: float
    ) -> Dict:
        """
        Executes one control cycle of the automated tether delivery sequence.
        """
        self.current_tension_n = measured_tension_n
        self.motor_current_a = measured_current_a

        # 1. Stall / Entanglement Guard
        if self.motor_current_a > self.stall_current_a and self.state in [WinchState.LOWERING, WinchState.RETRACTING]:
            print("[!] CRITICAL ALERT: Tether motor stall / cable entanglement detected.")
            self.state = WinchState.EMERGENCY_SEVER

        # 2. State Machine Execution
        if self.state == WinchState.RETRACTED:
            # Standby until lowering initiated
            pass

        elif self.state == WinchState.LOWERING:
            self.deployed_length_m += self.deploy_speed * dt_seconds
            
            # Check Touchdown: Tension drops below threshold while package hits ground
            if self.deployed_length_m >= (drone_alt_agl_m - 0.5) and self.current_tension_n < self.touchdown_tension_n:
                print(f"[*] Package touchdown detected (Tension: {self.current_tension_n:.2f} N). Halting spool.")
                self.state = WinchState.TOUCHDOWN_DETECTED

            elif self.deployed_length_m >= self.max_len:
                print("[!] Warning: Max cable length reached without touchdown. Initiating retract.")
                self.state = WinchState.RETRACTING

        elif self.state == WinchState.TOUCHDOWN_DETECTED:
            # Actuate release solenoid
            self.package_released = True
            self.pkg_weight_kg = 0.0
            print("[+] Cargo release solenoid fired. Package detached.")
            self.state = WinchState.RELEASE_PACKAGE

        elif self.state == WinchState.RELEASE_PACKAGE:
            # Transition to retraction
            self.state = WinchState.RETRACTING

        elif self.state == WinchState.RETRACTING:
            self.deployed_length_m = max(0.0, self.deployed_length_m - self.deploy_speed * dt_seconds)
            if self.deployed_length_m <= 0.05:
                self.deployed_length_m = 0.0
                self.state = WinchState.RETRACTED
                print("[+] Tether fully retracted and locked into docking bay.")

        elif self.state == WinchState.EMERGENCY_SEVER:
            # Fire pyrotechnic or solenoid line cutter
            print("[!] EMERGENCY SEVER TRIGGERED: Cable severed to protect aircraft.")
            self.deployed_length_m = 0.0
            self.package_released = True
            self.state = WinchState.FAULT

        return {
            "winch_state": self.state.value,
            "deployed_length_m": round(self.deployed_length_m, 2),
            "line_tension_n": round(self.current_tension_n, 2),
            "motor_current_a": round(self.motor_current_a, 2),
            "package_released": self.package_released
        }

    def initiate_delivery(self):
        """Starts tether lowering sequence."""
        if self.state == WinchState.RETRACTED:
            self.state = WinchState.LOWERING
            print("[*] Initiating tether lowering sequence...")

if __name__ == "__main__":
    controller = TetherWinchController(nominal_package_weight_kg=1.8)
    controller.initiate_delivery()

    print("==========================================================================")
    print("      TETHER WINCH DEPLOYMENT & TOUCHDOWN SEQUENCE SIMULATION             ")
    print("==========================================================================")
    
    drone_altitude = 8.0  # Drone hovering at 8 meters AGL
    
    for step in range(20):
        # Simulate tension drop at step 13 (~8 meters deployed)
        if controller.deployed_length_m < 7.5:
            sim_tension = 1.8 * 9.81 + np.random.normal(0, 0.2) if 'np' in globals() else 17.6
            sim_current = 1.4
        elif controller.deployed_length_m >= 7.5 and not controller.package_released:
            sim_tension = 0.8  # Ground support relieves weight
            sim_current = 0.9
        else:
            sim_tension = 0.2  # Bare hook weight
            sim_current = 1.1

        telemetry = controller.process_winch_step(
            dt_seconds=0.5,
            measured_tension_n=sim_tension,
            measured_current_a=sim_current,
            drone_alt_agl_m=drone_altitude
        )
        print(f"Step {step:02d} (T+{(step*0.5):4.1f}s): State={telemetry['winch_state']:20s} | "
              f"Cable={telemetry['deployed_length_m']:4.2f}m | Tension={telemetry['line_tension_n']:5.2f}N | "
              f"Released={telemetry['package_released']}")

        if telemetry["winch_state"] == WinchState.RETRACTED.value and step > 5:
            break
    print("==========================================================================\n")