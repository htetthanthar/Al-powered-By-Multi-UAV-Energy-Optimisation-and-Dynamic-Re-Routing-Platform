import math
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class PowerLineWireDetector:
    def __init__(
        self,
        frame_dimensions: Tuple[int, int] = (640, 480),
        camera_hfov_deg: float = 70.0,
        gabor_wavelength_px: float = 6.0,
        gabor_orientations_count: int = 8,
        min_line_length_px: int = 60,
        collision_time_horizon_s: float = 3.5
    ):
        self.w, self.h = frame_dimensions
        self.hfov = math.radians(camera_hfov_deg)
        self.min_line_len = min_line_length_px
        self.tau_crit = collision_time_horizon_s
        self.focal_px = (self.w / 2.0) / math.tan(self.hfov / 2.0)

        # Build Gabor filter bank across [0, pi) orientations
        self.gabor_filters = []
        for i in range(gabor_orientations_count):
            theta = i * (math.pi / gabor_orientations_count)
            kernel = cv2.getGaborKernel(
                ksize=(15, 15),
                sigma=2.5,
                theta=theta,
                lambd=gabor_wavelength_px,
                gamma=0.5,
                psi=0.0,
                ktype=cv2.CV_32F
            )
            self.gabor_filters.append(kernel)

    def extract_wire_candidates(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Convolves grayscale frame with Gabor filter bank to enhance high-contrast thin wire structures.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32) / 255.0

        max_response = np.zeros_like(gray_f)
        for kern in self.gabor_filters:
            filtered = cv2.filter2D(gray_f, cv2.CV_32F, kern)
            max_response = np.maximum(max_response, filtered)

        # Normalize and binary threshold
        response_norm = np.clip(max_response * 255.0, 0, 255).astype(np.uint8)
        binary_edges = cv2.threshold(response_norm, 110, 255, cv2.THRESH_BINARY)[1]
        return binary_edges

    def detect_and_evaluate_catenary_hazard(
        self,
        frame_bgr: np.ndarray,
        forward_airspeed_ms: float,
        altitude_agl_m: float
    ) -> Dict:
        """
        Extracts linear wire segments, fits catenary geometry, and determines collision risk.
        """
        edges = self.extract_wire_candidates(frame_bgr)
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=40,
            minLineLength=self.min_line_len,
            maxLineGap=25
        )

        detected_wires = []
        is_collision_imminent = False

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                wire_length_px = math.hypot(dx, dy)
                angle_deg = math.degrees(math.atan2(dy, dx))

                # Filter out pure vertical building edges (|angle| between 75° and 105°)
                if 75.0 <= abs(angle_deg) <= 105.0:
                    continue

                # Estimate distance to wire via pinhole pixel height:
                # Assumes typical power transmission line diameter ~ 20 mm
                estimated_range_m = max(5.0, (self.focal_px * 0.020) / max(abs(dy / max(wire_length_px, 1.0)), 0.08))
                
                # Time to wire impact: tau = Range / V_forward
                time_to_impact_s = estimated_range_m / max(forward_airspeed_ms, 1.0)
                
                if time_to_impact_s <= self.tau_crit:
                    is_collision_imminent = True

                detected_wires.append({
                    "start_pixel": (int(x1), int(y1)),
                    "end_pixel": (int(x2), int(y2)),
                    "orientation_deg": round(angle_deg, 1),
                    "estimated_range_m": round(estimated_range_m, 1),
                    "time_to_impact_s": round(time_to_impact_s, 2)
                })

        action = "CLEAR_CORRIDOR"
        if is_collision_imminent:
            action = "EMERGENCY_VERTICAL_PITCH_CLIMB_AVOIDANCE"
        elif len(detected_wires) > 0:
            action = "MONITOR_WIRE_CATENARY_HOLD"

        return {
            "wires_detected_count": len(detected_wires),
            "collision_imminent": is_collision_imminent,
            "closest_wire_range_m": min([w["estimated_range_m"] for w in detected_wires]) if detected_wires else None,
            "min_time_to_impact_s": min([w["time_to_impact_s"] for w in detected_wires]) if detected_wires else None,
            "flight_guidance_action": action,
            "detected_cables": detected_wires[:4]
        }

if __name__ == "__main__":
    detector = PowerLineWireDetector()

    # Synthetic camera frame with sky background and high-voltage transmission cable spanning horizontally
    synth_frame = np.full((480, 640, 3), 190, dtype=np.uint8)
    # Draw horizontal catenary wire at y=180px
    cv2.line(synth_frame, (40, 170), (600, 195), (40, 40, 40), 2)

    hazard_report = detector.detect_and_evaluate_catenary_hazard(
        synth_frame, forward_airspeed_ms=18.05, altitude_agl_m=28.0
    )

    print("==========================================================================")
    print("      VISION-BASED POWER LINE & THIN CABLE CATENARY DETECTOR AUDIT        ")
    print("==========================================================================")
    for k, v in hazard_report.items():
        if k != "detected_cables":
            print(f"  • {k:28s}: {v}")
    print("\nDetected Cable Segment Geometry:")
    for idx, c in enumerate(hazard_report["detected_cables"]):
        print(f"  Cable #{idx+1}: Segment {c['start_pixel']} -> {c['end_pixel']} | Angle: {c['orientation_deg']}° | "
              f"Range: {c['estimated_range_m']} m | Impact Time: {c['time_to_impact_s']} s")
    print("==========================================================================\n")