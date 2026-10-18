import math
import numpy as np
from typing import Dict, Tuple, List, Optional

class VLADeliveryGestureInterpreter:
    def __init__(
        self,
        min_safe_clearance_distance_m: float = 2.8,
        wave_frequency_abort_threshold_hz: float = 1.25,
        gesture_confidence_threshold: float = 0.82
    ):
        self.d_safe = min_safe_clearance_distance_m
        self.f_abort = wave_frequency_abort_threshold_hz
        self.conf_thresh = gesture_confidence_threshold
        self.wrist_history_x: List[float] = []

    def extract_skeletal_arm_geometry(
        self,
        keypoints_dict: Dict[str, Tuple[float, float, float]] # (x, y, confidence)
    ) -> Dict[str, float]:
        """
        Computes joint angles for left and right arms from 2D keypoint projections.
        Keypoints expected: 'nose', 'l_shoulder', 'l_elbow', 'l_wrist', 'r_shoulder', 'r_elbow', 'r_wrist'.
        """
        def angle_between_vectors(p1, p2, p3):
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            norm_prod = (np.linalg.norm(v1) * np.linalg.norm(v2))
            if norm_prod < 1e-4:
                return 0.0
            cos_a = np.clip(np.dot(v1, v2) / norm_prod, -1.0, 1.0)
            return math.degrees(math.acos(cos_a))

        ls = keypoints_dict.get("l_shoulder", (0, 0, 0))
        le = keypoints_dict.get("l_elbow", (0, 0, 0))
        lw = keypoints_dict.get("l_wrist", (0, 0, 0))
        rs = keypoints_dict.get("r_shoulder", (0, 0, 0))
        re = keypoints_dict.get("r_elbow", (0, 0, 0))
        rw = keypoints_dict.get("r_wrist", (0, 0, 0))

        angle_l_arm = angle_between_vectors(ls, le, lw)
        angle_r_arm = angle_between_vectors(rs, re, rw)

        # Height of wrists relative to shoulders (negative is above shoulder in image frame)
        lw_above = (ls[1] - lw[1]) > 0.05
        rw_above = (rs[1] - rw[1]) > 0.05

        return {
            "left_arm_extension_deg": angle_l_arm,
            "right_arm_extension_deg": angle_r_arm,
            "both_arms_raised_above_shoulders": lw_above and rw_above,
            "mean_confidence": float((ls[2] + le[2] + lw[2] + rs[2] + re[2] + rw[2]) / 6.0)
        }

    def evaluate_ground_marshalling_intent(
        self,
        human_pixel_coords: Tuple[int, int],
        keypoints: Dict[str, Tuple[float, float, float]],
        ground_distance_m: float,
        dt_seconds: float = 0.1
    ) -> Dict:
        """
        Classifies recipient marshalling gestures and executes winch safety logic.
        """
        geom = self.extract_skeletal_arm_geometry(keypoints)
        rw = keypoints.get("r_wrist", (0, 0, 0))

        # Update lateral wrist oscillation buffer
        self.wrist_history_x.append(rw[0])
        if len(self.wrist_history_x) > 20:
            self.wrist_history_x.pop(0)

        # Compute wrist lateral oscillation frequency
        zero_crossings = 0
        if len(self.wrist_history_x) >= 6:
            detrended = np.array(self.wrist_history_x) - np.mean(self.wrist_history_x)
            zero_crossings = int(np.sum(np.diff(np.sign(detrended) != 0)))
        
        oscillation_freq_hz = (zero_crossings / 2.0) / (len(self.wrist_history_x) * dt_seconds)

        # Gesture Classification Tree:
        # 1. Abort / Wave-Off: Both arms raised waving overhead rapidly (freq > 1.25 Hz)
        # 2. Clear to Lower: Arms extended horizontally / downward thumbs-up
        # 3. Hazard Encroachment: Person inside 2.8m drop radius
        is_wave_off = geom["both_arms_raised_above_shoulders"] and (oscillation_freq_hz >= self.f_abort)
        is_safe_distance = ground_distance_m >= self.d_safe

        detected_gesture = "NEUTRAL_STANDING"
        confidence = geom["mean_confidence"]

        if is_wave_off:
            detected_gesture = "EMERGENCY_ABORT_WAVEOFF_GESTURE"
            confidence = max(confidence, 0.95)
        elif is_safe_distance and not geom["both_arms_raised_above_shoulders"]:
            detected_gesture = "RECIPIENT_CLEAR_STAND_BACK"
            confidence = max(confidence, 0.88)
        elif not is_safe_distance:
            detected_gesture = "WARNING_RECIPIENT_TOO_CLOSE_TO_DROPZONE"
            confidence = 0.92

        # Action Decision
        action = "MAINTAIN_HOVER_HOLD"
        if is_wave_off:
            action = "ABORT_WINCH_RETRACT_TETHER_CLIMB_15M"
        elif is_safe_distance and detected_gesture == "RECIPIENT_CLEAR_STAND_BACK":
            action = "PROCEED_WITH_TETHER_PARCEL_LOWERING"
        elif not is_safe_distance:
            action = "PAUSE_WINCH_AUDIBLE_PROXIMITY_ALARM"

        return {
            "recipient_ground_distance_m": round(ground_distance_m, 2),
            "safe_keepout_radius_m": self.d_safe,
            "detected_gesture": detected_gesture,
            "gesture_confidence": round(confidence, 2),
            "overhead_wave_frequency_hz": round(oscillation_freq_hz, 2),
            "winch_action_policy": action,
            "delivery_clearance_granted": action == "PROCEED_WITH_TETHER_PARCEL_LOWERING"
        }

if __name__ == "__main__":
    vla = VLADeliveryGestureInterpreter()

    print("==========================================================================")
    print("      VLA RECIPIENT GROUND MARSHALLING GESTURE INTERPRETER AUDIT          ")
    print("==========================================================================")

    # Scenario 1: Overhead rapid waving (Emergency abort from ground observer)
    synth_keypoints_abort = {
        "l_shoulder": (0.45, 0.40, 0.95), "l_elbow": (0.42, 0.25, 0.92), "l_wrist": (0.40, 0.15, 0.90),
        "r_shoulder": (0.55, 0.40, 0.95), "r_elbow": (0.58, 0.25, 0.92), "r_wrist": (0.62, 0.15, 0.90)
    }
    # Simulate alternating wrist positions (1.8 Hz oscillation)
    for x_pos in [0.60, 0.68, 0.58, 0.70, 0.55, 0.72, 0.54, 0.71]:
        synth_keypoints_abort["r_wrist"] = (x_pos, 0.15, 0.90)
        report_abort = vla.evaluate_ground_marshalling_intent(
            (320, 240), synth_keypoints_abort, ground_distance_m=4.2, dt_seconds=0.1
        )

    print("Scenario 1 (Rapid Overhead Wave-Off):")
    for k, v in report_abort.items():
        print(f"  • {k:32s}: {v}")

    # Scenario 2: Recipient standing safely at 3.8m distance
    synth_keypoints_safe = {
        "l_shoulder": (0.45, 0.40, 0.95), "l_elbow": (0.43, 0.55, 0.92), "l_wrist": (0.42, 0.68, 0.90),
        "r_shoulder": (0.55, 0.40, 0.95), "r_elbow": (0.57, 0.55, 0.92), "r_wrist": (0.58, 0.68, 0.90)
    }
    vla.wrist_history_x.clear()
    report_safe = vla.evaluate_ground_marshalling_intent(
        (320, 240), synth_keypoints_safe, ground_distance_m=3.8, dt_seconds=0.1
    )
    print("\nScenario 2 (Recipient Stood Clear at 3.8m):")
    for k, v in report_safe.items():
        print(f"  • {k:32s}: {v}")
    print("==========================================================================\n")