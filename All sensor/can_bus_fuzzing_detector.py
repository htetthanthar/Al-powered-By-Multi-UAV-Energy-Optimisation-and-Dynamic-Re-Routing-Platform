import math
import time
import numpy as np
from collections import deque, Counter
from typing import Dict, List, Tuple, Optional

class DroneCANSecurityIntrusionDetector:
    def __init__(
        self,
        entropy_window_size: int = 64,
        max_acceptable_entropy_deviation: float = 1.45,
        nominal_mean_entropy: float = 3.82,
        max_interarrival_jitter_ms: float = 15.0
    ):
        self.w_size = entropy_window_size
        self.entropy_dev_limit = max_acceptable_entropy_deviation
        self.baseline_h = nominal_mean_entropy
        self.max_jitter = max_interarrival_jitter_ms

        self.packet_payload_window: deque = deque(maxlen=entropy_window_size)
        self.last_msg_timestamps: Dict[int, float] = {}
        self.expected_message_intervals_ms = {
            1092: 100.0,  # BatteryInfo: 10 Hz (100 ms)
            1030: 20.0,   # ESC Actuator Command: 50 Hz (20 ms)
            1060: 50.0    # GNSS Fix: 20 Hz (50 ms)
        }

    def compute_shannon_entropy(self, byte_sequence: bytes) -> float:
        """
        Computes empirical Shannon Entropy H(X) in bits per byte across the payload stream:
          H(X) = - sum_i p(x_i) * log2(p(x_i))
        """
        if not byte_sequence:
            return 0.0

        counts = Counter(byte_sequence)
        total_bytes = len(byte_sequence)
        entropy = 0.0

        for count in counts.values():
            p_x = count / total_bytes
            entropy -= p_x * math.log2(p_x)

        return entropy

    def inspect_can_frame(
        self,
        can_id: int,
        data_type_id: int,
        payload_bytes: bytes,
        receive_time_s: float
    ) -> Dict:
        """
        Evaluates frame entropy, inter-arrival timing jitter, and unauthorized injection anomalies.
        """
        # 1. Update Sliding Byte Entropy Window
        self.packet_payload_window.extend(payload_bytes)
        current_entropy = self.compute_shannon_entropy(bytes(self.packet_payload_window))

        # 2. Inter-arrival Timing Jitter Check
        jitter_ms = 0.0
        timing_anomaly = False
        
        if data_type_id in self.last_msg_timestamps:
            delta_t_ms = (receive_time_s - self.last_msg_timestamps[data_type_id]) * 1000.0
            expected_ms = self.expected_message_intervals_ms.get(data_type_id, 50.0)
            jitter_ms = abs(delta_t_ms - expected_ms)
            if jitter_ms > self.max_jitter:
                timing_anomaly = True

        self.last_msg_timestamps[data_type_id] = receive_time_s

        # 3. Entropy Anomaly Detection (Fuzzing attack pushes entropy to near 8.0, or zero-flood to 0.0)
        entropy_diff = abs(current_entropy - self.baseline_h)
        entropy_anomaly = len(self.packet_payload_window) >= self.w_size and (entropy_diff > self.entropy_dev_limit)

        # 4. Threat Classification
        status = "CAN_BUS_NOMINAL"
        threat_detected = False
        action = "ALLOW_FRAME"

        if entropy_anomaly and timing_anomaly:
            status = "CRITICAL_FUZZING_AND_BUS_FLOOD_ATTACK"
            threat_detected = True
            action = "ISOLATE_CAN_NODE_ENGAGE_HARDWARE_FIREWALL"
        elif timing_anomaly:
            status = "WARNING_CAN_TIMING_JITTER_ANOMALY"
        elif entropy_anomaly:
            status = "WARNING_HIGH_ENTROPY_PAYLOAD_TAMPERING"

        return {
            "can_id_hex": hex(can_id).upper(),
            "data_type_id": data_type_id,
            "window_shannon_entropy": round(current_entropy, 3),
            "entropy_deviation": round(entropy_diff, 3),
            "interarrival_jitter_ms": round(jitter_ms, 2),
            "threat_detected": threat_detected,
            "bus_security_status": status,
            "firewall_action": action
        }

if __name__ == "__main__":
    ids = DroneCANSecurityIntrusionDetector(entropy_window_size=64)

    print("==========================================================================")
    print("      DRONECAN BUS SHANNON ENTROPY FUZZING & INTRUSION DETECTION AUDIT    ")
    print("==========================================================================")

    now = time.time()

    # 1. Normal Flight Stream (Nominal 10Hz BatteryInfo packets)
    for i in range(20):
        dummy_payload = b"\x01\x18\x05\x32\x00\x55\x60"
        t_recv = now + (i * 0.100) # 100 ms intervals
        report = ids.inspect_can_frame(can_id=0x10044432, data_type_id=1092, payload_bytes=dummy_payload, receive_time_s=t_recv)

    print(f"Nominal State -> Entropy: {report['window_shannon_entropy']} bits/byte | Status: {report['bus_security_status']}")

    # 2. Inject Random High-Entropy Cryptographic Fuzzing Attack (Bus Flooding)
    np.random.seed(42)
    fuzz_payload = bytes(np.random.randint(0, 256, 32, dtype=np.uint8).tolist())
    # Attack packets arrive in rapid burst (1 ms apart)
    t_fuzz = now + 2.001
    attack_report = ids.inspect_can_frame(can_id=0x10040632, data_type_id=1030, payload_bytes=fuzz_payload, receive_time_s=t_fuzz)

    print("\nSimulated CAN Bus Fuzzing Injection Assessment:")
    for k, v in attack_report.items():
        print(f"  • {k:26s}: {v}")
    print("==========================================================================\n")