import os
import secrets
import time
import hmac
import hashlib
import json
import math
import numpy as np
from typing import Dict, Tuple, Optional

class C2SecurityAndSpoofingGuard:
    def __init__(
        self,
        shared_secret_key: Optional[str] = None,
        max_timestamp_skew_s: float = 5.0,
        gps_imu_drift_threshold_ms2: float = 2.8
    ):
        shared_secret_key = shared_secret_key or os.getenv("UAV_C2_SHARED_SECRET")
        if not shared_secret_key:
            raise ValueError("UAV_C2_SHARED_SECRET must be configured before enabling C2 commands")
        self.key = shared_secret_key.encode("utf-8")
        self.max_skew = max_timestamp_skew_s
        self.drift_thresh = gps_imu_drift_threshold_ms2
        self.seen_nonces: Dict[str, float] = {}

    def generate_signed_c2_command(self, action: str, target_coords: Dict[str, float]) -> Dict:
        """
        Packages and signs an offboard command message with timestamp, nonce, and HMAC-SHA256 signature.
        """
        payload = {
            "action": action,
            "target": target_coords,
            "timestamp": time.time(),
            "nonce": secrets.token_urlsafe(32)
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        signature = hmac.new(self.key, serialized, hashlib.sha256).hexdigest()

        return {
            "payload": payload,
            "signature": signature
        }

    def verify_c2_command(self, signed_message: Dict) -> Tuple[bool, str]:
        """
        Validates cryptographic authenticity, timestamp freshness, and replay-attack nonce uniqueness.
        """
        payload = signed_message.get("payload", {})
        received_sig = signed_message.get("signature", "")

        # 1. Timestamp Freshness Check
        try:
            msg_time = float(payload["timestamp"])
            nonce = str(payload["nonce"])
        except (KeyError, TypeError, ValueError):
            return False, "MALFORMED_COMMAND"
        if abs(time.time() - msg_time) > self.max_skew:
            return False, "TIMESTAMP_SKEW_EXCEEDED"

        # 2. Authenticate the message before any mutable replay-state update.
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        expected_sig = hmac.new(self.key, serialized, hashlib.sha256).hexdigest()

        if not hmac.compare_digest(received_sig, expected_sig):
            return False, "INVALID_SIGNATURE_TAMPERING"

        # 3. Retain valid nonces only for the replay window.
        now = time.time()
        self.seen_nonces = {
            known_nonce: seen_at
            for known_nonce, seen_at in self.seen_nonces.items()
            if now - seen_at <= self.max_skew
        }
        if nonce in self.seen_nonces:
            return False, "REPLAY_ATTACK_DETECTED"
        self.seen_nonces[nonce] = now

        return True, "AUTHENTICATION_SUCCESS"

    def detect_gps_spoofing(
        self,
        gps_velocity_ned_ms: np.ndarray,
        prev_gps_velocity_ned_ms: np.ndarray,
        imu_accel_ned_ms2: np.ndarray,
        dt_seconds: float
    ) -> Dict:
        """
        Cross-checks GPS derived acceleration against physical IMU strapdown acceleration:
        a_gps = (v_gps[k] - v_gps[k-1]) / dt
        residual = || a_gps - a_imu ||
        """
        a_gps = (gps_velocity_ned_ms - prev_gps_velocity_ned_ms) / max(dt_seconds, 1e-4)
        residual_norm = float(np.linalg.norm(a_gps - imu_accel_ned_ms2))

        is_spoofed = residual_norm > self.drift_thresh

        return {
            "gps_accel_norm": round(float(np.linalg.norm(a_gps)), 2),
            "imu_accel_norm": round(float(np.linalg.norm(imu_accel_ned_ms2)), 2),
            "kinematic_residual_ms2": round(residual_norm, 2),
            "gps_integrity_status": "SPOOFING_ALERT_ENGAGE_DEAD_RECKONING" if is_spoofed else "AUTHENTIC_GPS_LOCKED"
        }

if __name__ == "__main__":
    security_guard = C2SecurityAndSpoofingGuard(shared_secret_key="development-only-demo-key")

    # 1. Test Command Signing & Verification
    cmd = security_guard.generate_signed_c2_command(
        action="UPDATE_WAYPOINTS",
        target_coords={"lat": 1.3521, "lon": 103.8198, "alt": 25.0}
    )
    is_valid, reason = security_guard.verify_c2_command(cmd)

    print("==========================================================================")
    print("      C2 COMMAND CRYPTOGRAPHIC AUTHENTICATION & SPOOFING AUDIT            ")
    print("==========================================================================")
    print(f"Command Signature Valid : {is_valid} ({reason})")

    # Tampering test
    tampered_cmd = dict(cmd)
    tampered_cmd["payload"] = dict(cmd["payload"])
    tampered_cmd["payload"]["action"] = "UNAUTHORIZED_LAND"
    t_valid, t_reason = security_guard.verify_c2_command(tampered_cmd)
    print(f"Tampered Packet Rejected: {not t_valid} ({t_reason})")

    # 2. Test GPS Spoofing Inconsistency
    # Real physical flight: gentle 0.2 m/s^2 forward acceleration measured on IMU
    imu_acc = np.array([0.2, 0.0, 0.0])
    v_gps_prev = np.array([18.0, 0.0, 0.0])
    # Attacker injects sudden artificial 8.0 m/s step jump into GPS stream
    v_gps_spoofed = np.array([26.0, 0.0, 0.0])

    spoof_report = security_guard.detect_gps_spoofing(v_gps_spoofed, v_gps_prev, imu_acc, dt_seconds=0.1)
    print("\nGPS Signal Integrity Assessment:")
    for k, v in spoof_report.items():
        print(f"  • {k:26s}: {v}")
    print("==========================================================================\n")
