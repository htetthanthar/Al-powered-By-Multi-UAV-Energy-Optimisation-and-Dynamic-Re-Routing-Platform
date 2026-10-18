import time
import math
from pymavlink import mavutil

class DroneMAVLinkBridge:
    def __init__(self, connection_string="udpin:localhost:14540", baudrate=57600):
        """
        Connects to a PX4/ArduPilot SITL simulator or physical drone telemetry radio.
        Default port 14540 is standard for PX4 UDP SITL.
        """
        print(f"[*] Connecting to UAV Autopilot via {connection_string}...")
        self.master = mavutil.mavlink_connection(connection_string, baud=baudrate)
        self.master.wait_heartbeat()
        print(f"[+] Heartbeat received from System (ID: {self.master.target_system}, Component: {self.master.target_component})")
        
        self.current_state = {
            "lat": 0.0,
            "lon": 0.0,
            "alt_rel_m": 0.0,
            "ground_speed_ms": 0.0,
            "battery_soc_pct": 100.0,
            "voltage_v": 24.0,
            "current_a": 0.0,
            "armed": False,
            "flight_mode": "UNKNOWN"
        }

    def update_telemetry(self):
        """
        Polls non-blocking MAVLink message queues to update internal drone state.
        """
        while True:
            msg = self.master.recv_match(blocking=False)
            if not msg:
                break
            
            msg_type = msg.get_type()
            
            # 1. GPS and Position Telemetry
            if msg_type == 'GLOBAL_POSITION_INT':
                self.current_state["lat"] = msg.lat / 1e7
                self.current_state["lon"] = msg.lon / 1e7
                self.current_state["alt_rel_m"] = msg.relative_alt / 1e3
                vx = msg.vx / 100.0
                vy = msg.vy / 100.0
                self.current_state["ground_speed_ms"] = math.sqrt(vx**2 + vy**2)

            # 2. Battery & Electrical State
            elif msg_type == 'BATTERY_STATUS':
                self.current_state["battery_soc_pct"] = msg.battery_remaining
                self.current_state["voltage_v"] = msg.voltages[0] / 1000.0 if msg.voltages[0] != 65535 else 0.0
                self.current_state["current_a"] = msg.current_battery / 100.0 if msg.current_battery != -1 else 0.0

            # 3. System Status & Heartbeat
            elif msg_type == 'HEARTBEAT':
                self.current_state["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.current_state["flight_mode"] = mavutil.mode_string_v10(msg)

        return self.current_state

    def upload_waypoint_mission(self, waypoints):
        """
        Uploads a sequenced waypoint list to the autopilot mission storage.
        waypoints: List of dicts [{'lat': float, 'lon': float, 'alt': float}]
        """
        print(f"[*] Uploading {len(waypoints)} waypoints to drone flight controller...")
        self.master.mav.mission_clear_all_send(self.master.target_system, self.master.target_component)
        
        # Announce mission count
        self.master.mav.mission_count_send(
            self.master.target_system,
            self.master.target_component,
            len(waypoints)
        )

        for seq, wp in enumerate(waypoints):
            # Wait for autopilot to request item
            self.master.recv_match(type=['MISSION_REQUEST'], blocking=True)
            self.master.mav.mission_item_send(
                self.master.target_system,
                self.master.target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 1, 0, 0, 0, 0,
                wp["lat"], wp["lon"], wp.get("alt", 25.0)
            )
        
        # Acknowledge completion
        ack = self.master.recv_match(type=['MISSION_ACK'], blocking=True)
        print(f"[+] Mission upload acknowledged by UAV (Result: {ack.type})")

    def trigger_emergency_return_to_base(self):
        """
        Commands the flight autopilot to abort and Return-to-Launch (RTL).
        """
        print("[!] CRITICAL SAFETY ALERT: Commanding Immediate Return-to-Base (RTL)...")
        # Mode 6 = RTL in PX4 Custom Modes
        self.master.set_mode_rtl()

if __name__ == "__main__":
    # Test stub against mock/loopback connection
    bridge = DroneMAVLinkBridge(connection_string="udpout:localhost:14550")
    print("Bridge initialized. Ready for telemetry polling.")