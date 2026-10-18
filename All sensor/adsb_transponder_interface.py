import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

class ADSBMsgDecoderAndTrafficAvoider:
    def __init__(
        self,
        uav_icao_hex: str = "76A412",
        horizontal_alert_radius_m: float = 600.0,
        vertical_alert_separation_m: float = 60.0,
        tcpa_warning_threshold_s: float = 35.0
    ):
        self.own_icao = uav_icao_hex.upper()
        self.d_alert_h = horizontal_alert_radius_m
        self.d_alert_v = vertical_alert_separation_m
        self.tau_tcpa = tcpa_warning_threshold_s
        self.traffic_table: Dict[str, Dict] = {}

    def decode_df17_airborne_position(self, raw_hex: str) -> Dict:
        """
        Decodes Mode S DF17 Airborne Position Message (Type Code 9 to 18).
        Extracts 24-bit ICAO address, barometric altitude (11-bit Q-bit encoding), and CPR lat/lon.
        """
        msg_bin = bin(int(raw_hex, 16))[2:].zfill(112)
        
        df = int(msg_bin[0:5], 2)
        if df != 17:
            return {"valid": False, "error": "NOT_DF17"}

        icao = hex(int(msg_bin[8:32], 2))[2:].upper().zfill(6)
        type_code = int(msg_bin[32:37], 2)
        
        # Altitude decoding (Bits 40..51)
        alt_bits = msg_bin[40:52]
        q_bit = int(alt_bits[7])  # Bit 48 is Q-bit (1 = 25ft resolution)
        
        if q_bit == 1:
            raw_alt = int(alt_bits[:7] + alt_bits[8:], 2)
            alt_ft = raw_alt * 25 - 1000
            alt_msl_m = alt_ft * 0.3048
        else:
            alt_msl_m = 0.0

        # Compact Position Reporting (CPR) Bits
        cpr_format = int(msg_bin[53]) # 0 = Even, 1 = Odd
        cpr_lat = int(msg_bin[54:71], 2) / 131072.0
        cpr_lon = int(msg_bin[71:88], 2) / 131072.0

        return {
            "valid": True,
            "icao": icao,
            "type_code": type_code,
            "altitude_msl_m": round(alt_msl_m, 1),
            "cpr_lat": cpr_lat,
            "cpr_lon": cpr_lon,
            "cpr_format": "ODD" if cpr_format else "EVEN"
        }

    def decode_df17_airborne_velocity(self, raw_hex: str) -> Dict:
        """
        Decodes Mode S DF17 Airborne Velocity Message (Type Code 19, Subtype 1).
        Extracts East-West / North-South velocity components and vertical rate.
        """
        msg_bin = bin(int(raw_hex, 16))[2:].zfill(112)
        icao = hex(int(msg_bin[8:32], 2))[2:].upper().zfill(6)

        v_ew_sign = int(msg_bin[45])
        v_ew_val = int(msg_bin[46:56], 2) - 1
        v_ew_kts = -v_ew_val if v_ew_sign else v_ew_val

        v_ns_sign = int(msg_bin[56])
        v_ns_val = int(msg_bin[57:67], 2) - 1
        v_ns_kts = -v_ns_val if v_ns_sign else v_ns_val

        vx_ms = (v_ns_kts * 1852.0) / 3600.0  # North
        vy_ms = (v_ew_kts * 1852.0) / 3600.0  # East

        return {
            "valid": True,
            "icao": icao,
            "vel_north_ms": round(vx_ms, 2),
            "vel_east_ms": round(vy_ms, 2),
            "ground_speed_ms": round(math.hypot(vx_ms, vy_ms), 2)
        }

    def compute_conflict_tcpa(
        self,
        own_pos_ned: Tuple[float, float, float],
        own_vel_ned: Tuple[float, float, float],
        intruder_pos_ned: Tuple[float, float, float],
        intruder_vel_ned: Tuple[float, float, float]
    ) -> Dict:
        """
        Computes the Closest Point of Approach (CPA) and Time to CPA (TCPA):
          r(t) = r_rel + v_rel * t
          t_cpa = - (r_rel . v_rel) / ||v_rel||^2
        """
        r_rel = np.array([intruder_pos_ned[0] - own_pos_ned[0],
                          intruder_pos_ned[1] - own_pos_ned[1]])
        v_rel = np.array([intruder_vel_ned[0] - own_vel_ned[0],
                          intruder_vel_ned[1] - own_vel_ned[1]])

        dist_now_h = float(np.linalg.norm(r_rel))
        diff_alt = abs(intruder_pos_ned[2] - own_pos_ned[2])
        v_rel_sq = float(np.dot(v_rel, v_rel))

        if v_rel_sq < 1e-4:
            t_cpa = 0.0
            d_cpa = dist_now_h
        else:
            t_cpa = -float(np.dot(r_rel, v_rel)) / v_rel_sq
            if t_cpa < 0.0:
                t_cpa = 0.0
                d_cpa = dist_now_h
            else:
                pos_at_cpa = r_rel + v_rel * t_cpa
                d_cpa = float(np.linalg.norm(pos_at_cpa))

        # Threat Assessment
        is_traffic_advisory = (t_cpa <= self.tau_tcpa) and (d_cpa <= self.d_alert_h) and (diff_alt <= self.d_alert_v)
        is_resolution_advisory = (t_cpa <= 15.0) and (d_cpa <= self.d_alert_h * 0.5) and (diff_alt <= self.d_alert_v)

        evasive_action = "MAINTAIN_COURSE"
        if is_resolution_advisory:
            evasive_action = "EMERGENCY_DIVE_DESCENT_TO_TERRAIN_FLOOR"
        elif is_traffic_advisory:
            evasive_action = "LATERAL_BREAKOUT_90_DEG_RIGHT"

        return {
            "current_horizontal_range_m": round(dist_now_h, 1),
            "vertical_separation_m": round(diff_alt, 1),
            "time_to_cpa_s": round(t_cpa, 1),
            "projected_min_separation_cpa_m": round(d_cpa, 1),
            "threat_level": "RESOLUTION_ADVISORY" if is_resolution_advisory else ("TRAFFIC_ADVISORY" if is_traffic_advisory else "CLEAR"),
            "recommended_evasion": evasive_action
        }

if __name__ == "__main__":
    tcas = ADSBMsgDecoderAndTrafficAvoider()

    # Synthetic ADS-B DF17 Airborne Position frame
    sample_raw_pos = "8D4062D258C382D690C8AC2863A7"
    decoded_pos = tcas.decode_df17_airborne_position(sample_raw_pos)

    print("==========================================================================")
    print("      ADS-B 1090MHz MODE S ES INGESTION & TCAS THREAT AUDIT               ")
    print("==========================================================================")
    print("Decoded ADS-B Position Telemetry:")
    for k, v in decoded_pos.items():
        print(f"  • {k:25s}: {v}")

    # Simulated Intruder Closing In (General Aviation aircraft on intersecting path)
    own_p = (0.0, 0.0, -25.0)       # 25m MSL, hovering/cruising
    own_v = (18.05, 0.0, 0.0)       # 18.05 m/s North
    intruder_p = (1200.0, 400.0, -45.0) # 45m MSL (20m vertical delta)
    intruder_v = (-45.0, -15.0, 0.0)    # 47.4 m/s closing south-west

    threat = tcas.compute_conflict_tcpa(own_p, own_v, intruder_p, intruder_v)
    print("\nTCAS Conflict Prediction Analysis:")
    for k, v in threat.items():
        print(f"  • {k:32s}: {v}")
    print("==========================================================================\n")