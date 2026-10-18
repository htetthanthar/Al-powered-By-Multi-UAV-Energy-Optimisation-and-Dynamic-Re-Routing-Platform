import time
from enum import Enum
from typing import Dict, List, Optional

class DockingBayState(Enum):
    AVAILABLE = "AVAILABLE"
    DRONE_LANDED = "DRONE_LANDED"
    SWAPPING_BATTERY = "SWAPPING_BATTERY"
    PREFLIGHT_DIAGNOSTICS = "PREFLIGHT_DIAGNOSTICS"
    READY_FOR_DEPARTURE = "READY_FOR_DEPARTURE"
    FAULT = "FAULT"

class BatterySlotState(Enum):
    EMPTY = "EMPTY"
    CHARGING = "CHARGING"
    BALANCING = "BALANCING"
    FULLY_CHARGED_READY = "FULLY_CHARGED_READY"
    COOLDOWN = "COOLDOWN"

class AutomatedGroundVertiport:
    def __init__(self, vertiport_id: str = "VERTIPORT-CENTRAL-01", num_charging_slots: int = 6):
        self.vertiport_id = vertiport_id
        self.bay_state = DockingBayState.AVAILABLE
        self.current_drone_id: Optional[str] = None
        
        # 6-Bay Charging Carousel Initial State
        self.battery_inventory = [
            {"slot_id": 1, "pack_id": "BATT-6S-01", "soc_pct": 100.0, "soh_pct": 98.2, "temp_c": 26.5, "state": BatterySlotState.FULLY_CHARGED_READY},
            {"slot_id": 2, "pack_id": "BATT-6S-02", "soc_pct": 100.0, "soh_pct": 96.5, "temp_c": 27.0, "state": BatterySlotState.FULLY_CHARGED_READY},
            {"slot_id": 3, "pack_id": "BATT-6S-03", "soc_pct": 74.0,  "soh_pct": 94.0, "temp_c": 34.2, "state": BatterySlotState.CHARGING},
            {"slot_id": 4, "pack_id": "BATT-6S-04", "soc_pct": 45.0,  "soh_pct": 92.5, "temp_c": 36.8, "state": BatterySlotState.CHARGING},
            {"slot_id": 5, "pack_id": "BATT-6S-05", "soc_pct": 15.0,  "soh_pct": 91.0, "temp_c": 42.1, "state": BatterySlotState.COOLDOWN},
            {"slot_id": 6, "pack_id": None,          "soc_pct": 0.0,   "soh_pct": 0.0,  "temp_c": 25.0, "state": BatterySlotState.EMPTY}
        ]

    def select_best_ready_battery(self) -> Optional[Dict]:
        """Selects the battery with the highest SOH among fully charged and cooled packs."""
        ready_packs = [
            b for b in self.battery_inventory 
            if b["state"] == BatterySlotState.FULLY_CHARGED_READY and b["temp_c"] <= 30.0
        ]
        if not ready_packs:
            return None
        # Prioritize highest health pack
        ready_packs.sort(key=lambda x: x["soh_pct"], reverse=True)
        return ready_packs[0]

    def execute_automated_swap_sequence(self, drone_id: str, depleted_soc_pct: float, depleted_soh_pct: float) -> Dict:
        """
        Executes kinematic swap routine: Centering clamps -> Eject depleted -> Insert fresh -> Lock.
        Turnaround time target: <= 90 seconds.
        """
        t0 = time.time()
        self.current_drone_id = drone_id
        self.bay_state = DockingBayState.DRONE_LANDED
        print(f"[*] {drone_id} landed on {self.vertiport_id}. Engaging kinematic alignment clamps...")

        # 1. Locate empty receiver slot for depleted pack
        empty_slot = next((b for b in self.battery_inventory if b["state"] == BatterySlotState.EMPTY), None)
        if not empty_slot:
            self.bay_state = DockingBayState.FAULT
            return {"status": "FAILED", "reason": "NO_EMPTY_CHARGING_BAY_AVAILABLE"}

        # 2. Select best replacement pack
        fresh_pack = self.select_best_ready_battery()
        if not fresh_pack:
            self.bay_state = DockingBayState.FAULT
            return {"status": "FAILED", "reason": "NO_COOLED_FULLY_CHARGED_PACK_READY"}

        # 3. Swap Kinematics Execution
        self.bay_state = DockingBayState.SWAPPING_BATTERY
        depleted_pack_id = f"BATT-RET-{int(time.time())%1000:03d}"
        print(f"  -> Actuating 2-axis manipulator: Extracting depleted pack ({depleted_soc_pct:.1f}% SOC)...")
        print(f"  -> Depositing into Carousel Slot {empty_slot['slot_id']} (State: COOLDOWN).")
        
        empty_slot["pack_id"] = depleted_pack_id
        empty_slot["soc_pct"] = depleted_soc_pct
        empty_slot["soh_pct"] = depleted_soh_pct
        empty_slot["temp_c"] = 38.5  # Post-flight warm pack
        empty_slot["state"] = BatterySlotState.COOLDOWN

        print(f"  -> Gripping fresh pack {fresh_pack['pack_id']} from Slot {fresh_pack['slot_id']} ({fresh_pack['soc_pct']}% SOC)...")
        print(f"  -> Inserting and locking into {drone_id} battery bay...")
        
        installed_pack_id = fresh_pack["pack_id"]
        fresh_pack["pack_id"] = None
        fresh_pack["soc_pct"] = 0.0
        fresh_pack["soh_pct"] = 0.0
        fresh_pack["state"] = BatterySlotState.EMPTY

        # 4. Automated Pre-Flight Diagnostics
        self.bay_state = DockingBayState.PREFLIGHT_DIAGNOSTICS
        voltage_check = True
        bus_comm_check = True

        self.bay_state = DockingBayState.READY_FOR_DEPARTURE
        swap_duration_s = 48.5  # Simulated kinematic execution duration

        return {
            "status": "SUCCESS",
            "drone_id": drone_id,
            "installed_pack_id": installed_pack_id,
            "received_pack_id": depleted_pack_id,
            "turnaround_duration_s": swap_duration_s,
            "bus_voltage_v": 25.18,
            "ready_for_takeoff": voltage_check and bus_comm_check
        }

if __name__ == "__main__":
    vertiport = AutomatedGroundVertiport()
    swap_result = vertiport.execute_automated_swap_sequence(
        drone_id="UAV-01", depleted_soc_pct=18.5, depleted_soh_pct=95.0
    )

    print("==========================================================================")
    print("      AUTOMATED VERTIPORT BATTERY SWAP & TURNAROUND BENCHMARK             ")
    print("==========================================================================")
    for k, v in swap_result.items():
        print(f"  • {k:24s}: {v}")
    print("==========================================================================\n")