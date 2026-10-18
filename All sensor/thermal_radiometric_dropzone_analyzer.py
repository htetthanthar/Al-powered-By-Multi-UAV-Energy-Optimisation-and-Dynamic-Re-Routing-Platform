import math
import numpy as np
import cv2
from typing import Dict, List, Tuple

class ThermalRadiometricDropzoneAnalyzer:
    def __init__(
        self,
        sensor_resolution: Tuple[int, int] = (160, 120), # Standard Lepton/Boson LWIR core
        radiometric_scale_factor: float = 0.01,         # 0.01 K per ADU count
        kelvin_to_celsius_offset: float = 273.15,
        human_body_temp_range_c: Tuple[float, float] = (32.0, 39.0),
        vehicle_hotspot_threshold_c: float = 48.0
    ):
        self.width, self.height = sensor_resolution
        self.scale = radiometric_scale_factor
        self.k_offset = kelvin_to_celsius_offset
        self.human_min, self.human_max = human_body_temp_range_c
        self.vehicle_thresh = vehicle_hotspot_threshold_c

    def raw_adu_to_celsius(self, raw_16bit_frame: np.ndarray) -> np.ndarray:
        """
        Converts 16-bit raw digital radiometric counts to physical Celsius temperature:
          T_K = ADU * Scale  =>  T_C = T_K - 273.15
        """
        temp_k = raw_16bit_frame.astype(np.float32) * self.scale
        temp_c = temp_k - self.k_offset
        return temp_c

    def segment_thermal_hazards(
        self,
        celsius_frame: np.ndarray,
        ground_sampling_distance_m_per_px: float = 0.08
    ) -> Dict:
        """
        Identifies living human heat signatures, warm pets, and operating vehicle engines.
        """
        # 1. Human Body Thermal Mask
        human_mask = cv2.inRange(
            celsius_frame,
            np.array([self.human_min], dtype=np.float32),
            np.array([self.human_max], dtype=np.float32)
        )

        # 2. High-Temperature Machinery Mask (Vehicles/Generators)
        machinery_mask = cv2.threshold(
            celsius_frame,
            self.vehicle_thresh,
            255,
            cv2.THRESH_BINARY
        )[1].astype(np.uint8)

        # 3. Morphological Cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        human_clean = cv2.morphologyEx(human_mask, cv2.MORPH_OPEN, kernel)
        machinery_clean = cv2.morphologyEx(machinery_mask, cv2.MORPH_OPEN, kernel)

        # 4. Contour Extraction and Sizing
        contours_human, _ = cv2.findContours(human_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_machinery, _ = cv2.findContours(machinery_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_entities = []
        center_x, center_y = self.width // 2, self.height // 2

        # Extract Biological Obstacles
        for cnt in contours_human:
            area_px = cv2.contourArea(cnt)
            area_m2 = area_px * (ground_sampling_distance_m_per_px ** 2)
            if 0.10 <= area_m2 <= 2.5: # Human physical surface projection size
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    dist_to_center_m = math.hypot(cx - center_x, cy - center_y) * ground_sampling_distance_m_per_px
                    detected_entities.append({
                        "type": "HUMAN_BIOLOGICAL_HEAT_SIGNATURE",
                        "center_pixel": (cx, cy),
                        "offset_from_target_m": round(dist_to_center_m, 2),
                        "surface_area_m2": round(area_m2, 3),
                        "peak_temp_c": round(float(np.max(celsius_frame[cnt[:, 0, 1], cnt[:, 0, 0]])), 1)
                    })

        # Extract Machinery Obstacles
        for cnt in contours_machinery:
            area_px = cv2.contourArea(cnt)
            area_m2 = area_px * (ground_sampling_distance_m_per_px ** 2)
            if area_m2 >= 0.50:
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    dist_to_center_m = math.hypot(cx - center_x, cy - center_y) * ground_sampling_distance_m_per_px
                    detected_entities.append({
                        "type": "HIGH_HEAT_VEHICLE_SURFACE",
                        "center_pixel": (cx, cy),
                        "offset_from_target_m": round(dist_to_center_m, 2),
                        "surface_area_m2": round(area_m2, 3),
                        "peak_temp_c": round(float(np.max(celsius_frame[cnt[:, 0, 1], cnt[:, 0, 0]])), 1)
                    })

        # Safe drop zone determination: No obstacles within 3.5m radius of target center
        is_safe = all(e["offset_from_target_m"] > 3.5 for e in detected_entities)

        return {
            "mean_ambient_ground_temp_c": round(float(np.mean(celsius_frame)), 1),
            "max_surface_temp_c": round(float(np.max(celsius_frame)), 1),
            "detected_thermal_hazards_count": len(detected_entities),
            "thermal_hazards": detected_entities,
            "dropzone_thermally_clear": is_safe,
            "action": "PERMIT_NIGHT_TOUCHDOWN_RELEASE" if is_safe else "HOLD_ABORT_THERMAL_OBSTACLE_INTERCEPT"
        }

if __name__ == "__main__":
    analyzer = ThermalRadiometricDropzoneAnalyzer()

    # Synthetic 16-bit raw LWIR frame: Ambient grass at 26°C (299.15 K -> ~29915 ADU)
    raw_frame = np.full((120, 160), 29915, dtype=np.uint16)

    # Insert warm human body at (x=80, y=70) -> 36.5°C (309.65 K -> 30965 ADU)
    cv2.circle(raw_frame, (80, 70), 8, 30965, -1)

    # Insert idling vehicle exhaust at (x=130, y=30) -> 58.0°C (331.15 K -> 33115 ADU)
    cv2.circle(raw_frame, (130, 30), 12, 33115, -1)

    celsius_grid = analyzer.raw_adu_to_celsius(raw_frame)
    audit = analyzer.segment_thermal_hazards(celsius_grid, ground_sampling_distance_m_per_px=0.08)

    print("==========================================================================")
    print("      RADIOMETRIC LWIR THERMAL NIGHT-VISION DROPZONE AUDIT                ")
    print("==========================================================================")
    print(f"Mean Ambient Ground Temperature : {audit['mean_ambient_ground_temp_c']} °C")
    print(f"Peak Scene Temperature          : {audit['max_surface_temp_c']} °C")
    print(f"Detected Thermal Obstacles      : {audit['detected_thermal_hazards_count']}")
    print(f"Dropzone Thermal Clearance      : {'PASSED (SAFE)' if audit['dropzone_thermally_clear'] else 'REJECTED (ACTIVE HAZARD)'}\n")
    for idx, ent in enumerate(audit["thermal_hazards"]):
        print(f"  [{idx+1}] Type: {ent['type']} | Peak Temp: {ent['peak_temp_c']}°C | "
              f"Area: {ent['surface_area_m2']} m² | Offset from Target: {ent['offset_from_target_m']} m")
    print("==========================================================================\n")