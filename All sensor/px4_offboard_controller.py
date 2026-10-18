import time
import math
import numpy as np
from pymavlink import mavutil
from typing import List, Dict, Tuple

class PX4OffboardFlightController:
    def __init__(self, connection_str: str = "udpin:0.0.0.0:14540", target_sys: int = 1, target_comp: int = 1):
        """
        Initializes MAVLink connection to PX4 Autopilot (SITL or Companion Serial Interface).
        """
        self.master = mavutil.mavlink_connection(connection_str, baud=921600)
        self.target_sys = target_sys
        self.target_comp = target_comp
        
        # Local NED Odometry State
        self.local_position = {"x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw": 0.0}
        self.armed = False
        self.nav_state = "MANUAL"
        
        self._wait_for_connection()

    def _wait_for_connection(self):
        print(f"[*] Awaiting MAVLink Heartbeat from PX4 on {self.master.address}...")
        self.master.wait_heartbeat()
        print(f"[+] Connected to Autopilot (SysID: {self.master.target_system}, CompID: {self.master.target_component})")

    def update_state(self):
        """Processes incoming MAVLink messages to update NED state."""
        while True:
            msg = self.master.recv_match(blocking=False)
            if not msg:
                break
            msg_type = msg.get_type()

            if msg_type == 'LOCAL_POSITION_NED':
                self.local_position["x"] = msg.x
                self.local_position["y"] = msg.y
                self.local_position["z"] = msg.z  # Negative in NED (up is -z)
                self.local_position["vx"] = msg.vx
                self.local_position["vy"] = msg.vy
                self.local_position["vz"] = msg.vz

            elif msg_type == 'HEARTBEAT':
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def arm_and_set_offboard(self):
        """Arms drone and transitions PX4 flight mode to OFFBOARD."""
        print("[*] Streaming initial setpoint stream before offboard switch...")
        # Stream 100 setpoints to satisfy PX4 offboard watchdog (> 2 Hz required)
        for _ in range(50):
            self.send_position_target_ned(0.0, 0.0, -10.0, yaw=0.0)
            time.sleep(0.02)

        # Set Mode: OFFBOARD (Custom Mode 6 for PX4)
        print("[*] Engaging OFFBOARD Mode...")
        self.master.mav.command_long_send(
            self.target_sys, self.target_comp,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            6, 0, 0, 0, 0, 0
        )

        # Arm Motors
        print("[*] Arming Propulsion System...")
        self.master.mav.command_long_send(
            self.target_sys, self.target_comp,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

    def send_position_target_ned(self, x: float, y: float, z: float, yaw: float = 0.0):
        """
        Transmits SET_POSITION_TARGET_LOCAL_NED setpoints at coordinate targets.
        Bitmask 0b0000111111111000 -> Position + Yaw enabled only.
        """
        type_mask = 0b0000101111111000  # Position + Yaw target
        self.master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (ignored)
            self.target_sys, self.target_comp,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            x, y, z,        # Position X, Y, Z in meters (NED)
            0, 0, 0,        # Velocity vx, vy, vz
            0, 0, 0,        # Acceleration ax, ay, az
            yaw, 0.0        # Yaw, Yaw Rate
        )

    def execute_trajectory(self, ned_waypoints: List[Dict[str, float]], acceptance_radius_m: float = 2.0):
        """
        Iterates through optimized NED waypoints, controlling position setpoints until convergence.
        """
        print(f"[*] Commencing Autonomous Trajectory Tracking ({len(ned_waypoints)} Nodes)...")
        
        for idx, wp in enumerate(ned_waypoints):
            tx, ty, tz = wp["x"], wp["y"], wp["z"]
            print(f"  -> Navigating to Waypoint {idx+1}/{len(ned_waypoints)}: [X: {tx:.1f}m, Y: {ty:.1f}m, Alt: {-tz:.1f}m]")

            while True:
                self.update_state()
                self.send_position_target_ned(tx, ty, tz)

                dx = tx - self.local_position["x"]
                dy = ty - self.local_position["y"]
                dz = tz - self.local_position["z"]
                tracking_error = math.sqrt(dx**2 + dy**2 + dz**2)

                if tracking_error <= acceptance_radius_m:
                    print(f"  [+] Waypoint {idx+1} Reached (Tracking Error: {tracking_error:.2f}m).")
                    if wp.get("hold_time_s", 0) > 0:
                        print(f"      Holding for {wp['hold_time_s']} seconds (Package Delivery Sequence)...")
                        hold_end = time.time() + wp["hold_time_s"]
                        while time.time() < hold_end:
                            self.send_position_target_ned(tx, ty, tz)
                            time.sleep(0.05)
                    break

                time.sleep(0.05)  # 20 Hz update loop

        print("[+] Mission Complete. Executing Automated Landing Sequence...")
        self.master.mav.command_long_send(
            self.target_sys, self.target_comp,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )

if __name__ == "__main__":
    controller = PX4OffboardFlightController()
    
    # 3D Local NED Trajectory (Z is negative upwards)
    trajectory = [
        {"x": 0.0, "y": 0.0, "z": -15.0, "hold_time_s": 2.0},    # Climb to 15m
        {"x": 50.0, "y": 100.0, "z": -25.0, "hold_time_s": 5.0}, # Delivery Point Alpha
        {"x": 120.0, "y": -40.0, "z": -25.0, "hold_time_s": 5.0},# Delivery Point Beta
        {"x": 0.0, "y": 0.0, "z": -15.0, "hold_time_s": 2.0}     # Return Over Depot
    ]

    controller.arm_and_set_offboard()
    controller.execute_trajectory(trajectory)