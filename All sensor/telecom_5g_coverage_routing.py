import math
import numpy as np
from typing import Dict, List, Tuple

class CellularCoverage5GRouter:
    def __init__(
        self,
        carrier_freq_ghz: float = 3.5, # 5G NR n78 band
        min_allowable_rsrp_dbm: float = -105.0,
        min_allowable_sinr_db: float = 3.0,
        thermal_noise_density_dbm_hz: float = -174.0,
        channel_bandwidth_mhz: float = 40.0
    ):
        self.fc = carrier_freq_ghz
        self.min_rsrp = min_allowable_rsrp_dbm
        self.min_sinr = min_allowable_sinr_db
        
        # Thermal Noise Power (dBm) = -174 + 10*log10(BW_Hz) + Noise_Figure (7 dB)
        bw_hz = channel_bandwidth_mhz * 1e6
        self.noise_floor_dbm = thermal_noise_density_dbm_hz + 10.0 * math.log10(bw_hz) + 7.0

        # Base Station (gNodeB) Default Parameters
        self.gnb_tx_power_dbm = 46.0 # 40 Watts
        self.gnb_height_m = 30.0
        self.gnb_downtilt_deg = 6.0
        self.gnb_antenna_gain_dbi = 18.0

        # Cellular Towers in Urban Operations Area
        self.base_stations = [
            {"id": "GNB-CENTRAL", "x": 0.0, "y": 0.0, "z": 30.0},
            {"id": "GNB-NORTH",   "x": 2500.0, "y": 3000.0, "z": 32.0},
            {"id": "GNB-EAST",    "x": 3500.0, "y": -1500.0, "z": 28.0},
            {"id": "GNB-WEST",    "x": -2800.0, "y": 1200.0, "z": 35.0}
        ]

    def compute_3gpp_path_loss_db(self, d_3d_m: float, d_2d_m: float, uav_alt_m: float) -> Tuple[float, bool]:
        """
        3GPP TR 38.901 Urban Micro (UMi) Path Loss for Aerial Vehicles:
          PL_LOS  = 28.0 + 22.0*log10(d_3D) + 20.0*log10(fc)
          PL_NLOS = max(PL_LOS, 13.54 + 39.08*log10(d_3D) + 20.0*log10(fc) - 0.6*(h_UT - 1.5))
        """
        d_3d = max(d_3d_m, 10.0)
        
        # Line-of-Sight Probability for low-altitude drones
        p_los = 1.0 if d_2d_m <= 18.0 else (18.0 / d_2d_m + math.exp(-d_2d_m / 63.0) * (1.0 - 18.0 / d_2d_m))
        is_los = np.random.rand() < p_los

        pl_los = 28.0 + 22.0 * math.log10(d_3d) + 20.0 * math.log10(self.fc)
        
        if is_los:
            return pl_los, True
        else:
            pl_nlos = 13.54 + 39.08 * math.log10(d_3d) + 20.0 * math.log10(self.fc) - 0.6 * (uav_alt_m - 1.5)
            return max(pl_los, pl_nlos), False

    def compute_antenna_radiation_gain(self, elevation_angle_deg: float) -> float:
        """
        Computes 3GPP vertical antenna radiation pattern attenuation based on downtilt angle.
        """
        theta_3db = 10.0 # 3dB vertical beamwidth
        sla_v = 30.0     # Side-lobe attenuation
        angle_diff = elevation_angle_deg - (-self.gnb_downtilt_deg)
        attenuation_db = -min(12.0 * ((angle_diff / theta_3db) ** 2), sla_v)
        return self.gnb_antenna_gain_dbi + attenuation_db

    def evaluate_c2_link_at_waypoint(self, uav_pos_ned: Tuple[float, float, float]) -> Dict:
        """
        Calculates received signal metrics (RSRP, SINR) from serving and interfering gNodeBs.
        uav_pos_ned: (x_north, y_east, z_down) where altitude = -z_down.
        """
        uav_x, uav_y, uav_alt = uav_pos_ned[0], uav_pos_ned[1], -uav_pos_ned[2]
        
        received_powers_mw = []

        for gnb in self.base_stations:
            dx = uav_x - gnb["x"]
            dy = uav_y - gnb["y"]
            dz = uav_alt - gnb["z"]
            d_2d = math.hypot(dx, dy)
            d_3d = math.sqrt(dx**2 + dy**2 + dz**2)

            elev_angle_deg = math.degrees(math.atan2(dz, max(d_2d, 1.0)))
            ant_gain_dbi = self.compute_antenna_radiation_gain(elev_angle_deg)
            path_loss_db, is_los = self.compute_3gpp_path_loss_db(d_3d, d_2d, uav_alt)

            # Link budget: RSRP = Tx_Power + Tx_Gain + Rx_Gain(0 dBi) - Path_Loss
            rsrp_dbm = self.gnb_tx_power_dbm + ant_gain_dbi - path_loss_db
            rsrp_mw = 10.0 ** (rsrp_dbm / 10.0)

            received_powers_mw.append({
                "gnb_id": gnb["id"],
                "rsrp_dbm": rsrp_dbm,
                "rsrp_mw": rsrp_mw,
                "is_los": is_los,
                "distance_3d_m": d_3d
            })

        # Sort to identify serving cell (strongest RSRP) and interferers
        received_powers_mw.sort(key=lambda x: x["rsrp_mw"], reverse=True)
        serving_cell = received_powers_mw[0]
        interfering_cells = received_powers_mw[1:]

        # SINR Calculation: SINR = P_serving / (Sum(P_interferers) + Thermal_Noise)
        interference_mw = sum(c["rsrp_mw"] for c in interfering_cells)
        noise_mw = 10.0 ** (self.noise_floor_dbm / 10.0)
        
        sinr_linear = serving_cell["rsrp_mw"] / (interference_mw + noise_mw)
        sinr_db = 10.0 * math.log10(max(sinr_linear, 1e-6))

        is_c2_viable = (serving_cell["rsrp_dbm"] >= self.min_rsrp) and (sinr_db >= self.min_sinr)

        return {
            "serving_gnb_id": serving_cell["gnb_id"],
            "serving_rsrp_dbm": round(serving_cell["rsrp_dbm"], 2),
            "sinr_db": round(sinr_db, 2),
            "los_status": "LOS" if serving_cell["is_los"] else "NLOS",
            "distance_to_gnb_m": round(serving_cell["distance_3d_m"], 1),
            "c2_link_reliable": is_c2_viable,
            "action": "CORRIDOR_PERMITTED" if is_c2_viable else "REROUTE_RADIO_HOLE_DETECTED"
        }

if __name__ == "__main__":
    c2_router = CellularCoverage5GRouter()

    print("==========================================================================")
    print("      5G NR / BVLOS CELLULAR C2 LINK BUDGET & RLOS COVERAGE AUDIT         ")
    print("==========================================================================")

    # Evaluate coverage along ascending altitude profile at (x=1200m, y=1500m)
    for alt in [15.0, 25.0, 45.0, 60.0]:
        pos = (1200.0, 1500.0, -alt)
        link = c2_router.evaluate_c2_link_at_waypoint(pos)
        print(f"Altitude {alt:4.1f}m AGL | Serving: {link['serving_gnb_id']:11s} | "
              f"RSRP: {link['serving_rsrp_dbm']:6.1f} dBm | SINR: {link['sinr_db']:5.1f} dB | "
              f"Status: {link['los_status']:4s} | C2 Link: {'PASS' if link['c2_link_reliable'] else 'FAIL'}")
    print("==========================================================================\n")