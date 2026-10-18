import time
from enum import Enum

class UAVState(Enum):
    IDLE = "IDLE"
    PRE_FLIGHT_CHECK = "PRE_FLIGHT_CHECK"
    EN_ROUTE = "EN_ROUTE"
    SERVICING_DROP = "SERVICING_DROP"
    HOVER_HOLD = "HOVER_HOLD"
    DYNAMIC_REROUTING = "DYNAMIC_REROUTING"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    EMERGENCY_LAND = "EMERGENCY_LAND"

class MissionExecutionFSM:
    def __init__(self, mission_id, total_stops):
        self.mission_id = mission_id
        self.total_stops = total_stops
        self.current_stop_idx = 0
        self.current_state = UAVState.IDLE
        self.hover_start_time = None
        self.max_hover_duration_s = 180.0 # 3 minute maximum hold timeout

    def process_telemetry_frame(self, telemetry):
        """
        Evaluates incoming telemetry and transitions state based on safety limits.
        telemetry: dict containing 'wind_kmh', 'rain_flag', 'battery_soc', 'alt_m'
        """
        # Global Emergency Guard: Critical Battery Exhaustion
        if telemetry["battery_soc"] < 15.0 and self.current_state != UAVState.EMERGENCY_LAND:
            self._transition_to(UAVState.EMERGENCY_LAND, "Battery SOC < 15% Emergency Margin")
            return self.current_state

        # State Machine Transitions
        if self.current_state == UAVState.IDLE:
            self._transition_to(UAVState.PRE_FLIGHT_CHECK, "Mission initialized")

        elif self.current_state == UAVState.PRE_FLIGHT_CHECK:
            if telemetry["wind_kmh"] <= 30.0 and not telemetry["rain_flag"]:
                self._transition_to(UAVState.EN_ROUTE, "Safety pre-checks cleared")
            else:
                self._transition_to(UAVState.IDLE, "Weather safety threshold exceeded before takeoff")

        elif self.current_state == UAVState.EN_ROUTE:
            # Check Environmental Disruptions
            if telemetry["rain_flag"] or telemetry["wind_kmh"] > 30.0:
                self.hover_start_time = time.time()
                self._transition_to(UAVState.HOVER_HOLD, f"Environmental shock: Wind {telemetry['wind_kmh']} km/h, Rain={telemetry['rain_flag']}")
            elif telemetry["battery_soc"] < 25.0:
                self._transition_to(UAVState.RETURN_TO_BASE, "Low battery reserve warning")

        elif self.current_state == UAVState.HOVER_HOLD:
            # Check recovery vs timeout
            elapsed_hover = time.time() - self.hover_start_time
            if not telemetry["rain_flag"] and telemetry["wind_kmh"] <= 28.0:
                self._transition_to(UAVState.EN_ROUTE, "Weather cleared within hold window")
            elif elapsed_hover > self.max_hover_duration_s:
                self._transition_to(UAVState.RETURN_TO_BASE, "Hold duration timed out (180s); aborting mission")

        elif self.current_state == UAVState.DYNAMIC_REROUTING:
            # Instant transition after RL policy recalculates path
            self._transition_to(UAVState.EN_ROUTE, "New feasible trajectory loaded")

        return self.current_state

    def _transition_to(self, new_state, reason):
        print(f"[*] FSM STATE TRANSITION: [{self.current_state.value}] -> [{new_state.value}] | Reason: {reason}")
        self.current_state = new_state

if __name__ == "__main__":
    fsm = MissionExecutionFSM("MSN-TEST-001", total_stops=3)
    
    # 1. Normal Startup
    fsm.process_telemetry_frame({"wind_kmh": 12.0, "rain_flag": False, "battery_soc": 98.0, "alt_m": 25.0})
    fsm.process_telemetry_frame({"wind_kmh": 12.0, "rain_flag": False, "battery_soc": 95.0, "alt_m": 25.0})
    
    # 2. Weather Disruption Occurs
    fsm.process_telemetry_frame({"wind_kmh": 34.5, "rain_flag": True, "battery_soc": 88.0, "alt_m": 25.0})