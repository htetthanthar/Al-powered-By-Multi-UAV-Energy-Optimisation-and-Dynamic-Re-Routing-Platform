import struct
import time
import math
from typing import Dict, List, Optional, Tuple

class DroneCANBatteryMonitor:
    def __init__(
        self,
        can_interface: str = "can0",
        node_id: int = 50,
        max_cell_imbalance_v: float = 0.040, # 40 mV max delta threshold
        critical_cell_low_v: float = 3.30,   # 3.30V per cell cutoff under load
        max_pack_temp_c: float = 55.0
    ):
        self.interface = can_interface
        self.node_id = node_id
        self.max_imbalance = max_cell_imbalance_v
        self.crit_cell_low = critical_cell_low_v
        self.max_temp = max_pack_temp_c

        # Pack Hardware State
        self.pack_voltage_v = 0.0
        self.pack_current_a = 0.0
        self.pack_temperature_c = 25.0
        self.state_of_charge_pct = 100.0
        self.state_of_health_pct = 100.0
        self.cell_voltages_v: List[float] = [4.20] * 6 # 6S Pack
        self.cumulative_mah_discharged = 0.0
        self.last_update_epoch_s = 0.0

    def encode_dronecan_battery_info(
        self,
        battery_id: int,
        voltage: float,
        current: float,
        temperature_k: float,
        soc_pct: float,
        soh_pct: float
    ) -> bytes:
        """
        Encodes UAVCAN/DroneCAN `uavcan.equipment.power.BatteryInfo` (Data Type ID 1092 / 0x0444).
        Byte Layout:
          - uint8: battery_id
          - float16: voltage (V)
          - float16: current (A)
          - float16: temperature (C)
          - uint8: state_of_charge_pct (0..100)
          - uint8: state_of_health_pct (0..100)
        """
        # IEEE 754 half-precision float encoding
        def float_to_f16(val: float) -> int:
            return struct.unpack('>H', struct.pack('>e', val))[0]

        v_f16 = float_to_f16(voltage)
        i_f16 = float_to_f16(current)
        t_f16 = float_to_f16(temperature_k)
        soc_u8 = int(min(100, max(0, soc_pct)))
        soh_u8 = int(min(100, max(0, soh_pct)))

        return struct.pack("<BHHHBB", battery_id, v_f16, i_f16, t_f16, soc_u8, soh_u8)

    def decode_dronecan_frame(self, can_id: int, payload_bytes: bytes) -> Dict:
        """
        Decodes incoming standard DroneCAN frame payloads and cell voltage arrays.
        """
        self.last_update_epoch_s = time.time()

        if len(payload_bytes) >= 9:
            battery_id, v_raw, i_raw, t_raw, soc, soh = struct.unpack("<BHHHBB", payload_bytes[:9])
            
            # Unpack float16 fields
            self.pack_voltage_v = struct.unpack('>e', struct.pack('>H', v_raw))[0]
            self.pack_current_a = struct.unpack('>e', struct.pack('>H', i_raw))[0]
            self.pack_temperature_c = struct.unpack('>e', struct.pack('>H', t_raw))[0] - 273.15
            self.state_of_charge_pct = float(soc)
            self.state_of_health_pct = float(soh)

        # Simulated 6S cell distribution
        avg_v = self.pack_voltage_v / 6.0
        self.cell_voltages_v = [
            round(avg_v + 0.005 * math.sin(i), 3) for i in range(6)
        ]

        # Calculate cell imbalance
        v_min = min(self.cell_voltages_v)
        v_max = max(self.cell_voltages_v)
        v_delta = v_max - v_min

        # Diagnostic Flags
        imbalance_fault = v_delta > self.max_imbalance
        undervoltage_fault = v_min < self.crit_cell_low
        overtemp_fault = self.pack_temperature_c > self.max_temp

        status = "NOMINAL"
        if overtemp_fault:
            status = "CRITICAL_OVERTEMPERATURE_PROTECTION"
        elif undervoltage_fault:
            status = "CRITICAL_CELL_UNDERVOLTAGE_CUTOFF"
        elif imbalance_fault:
            status = "WARNING_CELL_IMBALANCE_DEGRADATION"

        return {
            "pack_voltage_v": round(self.pack_voltage_v, 2),
            "pack_current_a": round(self.pack_current_a, 2),
            "power_draw_watts": round(self.pack_voltage_v * self.pack_current_a, 1),
            "temperature_c": round(self.pack_temperature_c, 1),
            "soc_pct": self.state_of_charge_pct,
            "soh_pct": self.state_of_health_pct,
            "cell_voltages_v": self.cell_voltages_v,
            "cell_imbalance_mv": round(v_delta * 1000.0, 1),
            "bms_hardware_status": status,
            "discharge_permitted": not (overtemp_fault or undervoltage_fault)
        }

if __name__ == "__main__":
    bms = DroneCANBatteryMonitor()

    # Encode nominal DroneCAN battery payload
    sample_payload = bms.encode_dronecan_battery_info(
        battery_id=1,
        voltage=22.80,         # 22.8V pack (3.80V/cell nominal)
        current=14.20,         # 14.2A discharge (~323W cruise power)
        temperature_k=306.65,  # 33.5°C
        soc_pct=78.0,
        soh_pct=95.0
    )

    telemetry = bms.decode_dronecan_frame(can_id=0x10044432, payload_bytes=sample_payload)

    print("==========================================================================")
    print("      DRONECAN / UAVCAN v1.0 SMART BMS TELEMETRY DECODER AUDIT            ")
    print("==========================================================================")
    for k, v in telemetry.items():
        print(f"  • {k:28s}: {v}")
    print("==========================================================================\n")