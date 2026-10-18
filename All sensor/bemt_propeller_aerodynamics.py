# bemt_propeller_aerodynamics.py
import math
import numpy as np
from typing import Dict, Tuple

class BladeElementMomentumSolver:
    def __init__(
        self,
        propeller_radius_m: float = 0.1905, # 15-inch propeller
        hub_radius_m: float = 0.025,
        num_blades: int = 2,
        num_radial_stations: int = 25,
        air_density_kg_m3: float = 1.225
    ):
        self.R = propeller_radius_m
        self.r_hub = hub_radius_m
        self.B = num_blades
        self.N_r = num_radial_stations
        self.rho = air_density_kg_m3

        # Radial discretization
        self.r_stations = np.linspace(self.r_hub, self.R, self.N_r)
        self.dr = (self.R - self.r_hub) / max(self.N_r - 1, 1)

        # Baseline carbon-fiber propeller geometry (linear twist and chord distribution)
        r_norm = (self.r_stations - self.r_hub) / (self.R - self.r_hub)
        self.theta_dist = np.radians(22.0 - 12.0 * r_norm) # 22° at hub to 10° at tip
        self.chord_dist = 0.038 - 0.018 * r_norm          # 38 mm at hub to 20 mm at tip

    def _airfoil_polars(self, alpha_rad: float, reynolds: float) -> Tuple[float, float]:
        """Calculates 2D airfoil lift and parasite drag coefficients."""
        alpha_deg = math.degrees(alpha_rad)
        
        if abs(alpha_deg) < 14.0:
            c_l = 2.0 * math.pi * alpha_rad + 0.25
            c_d = 0.012 + 0.045 * (alpha_rad ** 2)
        else:
            c_l = 1.15 * math.sin(2.0 * alpha_rad)
            c_d = 0.08 + 1.25 * (math.sin(alpha_rad) ** 2)

        re_correction = max(0.85, min(1.3, (max(reynolds, 1000.0) / 100000.0) ** 0.12))
        return float(c_l * re_correction), float(c_d / re_correction)

    def compute_bemt_performance(
        self,
        omega_rpm: float,
        freestream_axial_velocity_ms: float = 0.0,
        **kwargs
    ) -> Dict:
        """Solves local induced velocity v_i(r) iteratively at each radial blade station."""
        omega_rad_s = (omega_rpm / 60.0) * 2.0 * math.pi
        v_inf = max(0.0, float(freestream_axial_velocity_ms))

        total_thrust_n = 0.0
        total_torque_nm = 0.0

        for idx, r in enumerate(self.r_stations):
            theta = self.theta_dist[idx]
            chord = self.chord_dist[idx]
            v_theta = max(omega_rad_s * r, 1e-3)

            solidity = (self.B * chord) / (2.0 * math.pi * r)
            phi = math.atan2(v_inf + 0.5, v_theta)
            v_induced = 1.0

            for _ in range(35):
                v_eff = math.sqrt(max(0.0, (v_inf + v_induced)**2 + v_theta**2))
                alpha = theta - phi
                reynolds = (self.rho * v_eff * chord) / 1.789e-5

                c_l, c_d = self._airfoil_polars(alpha, reynolds)

                # Prandtl Tip & Hub Loss Factors
                sin_phi = max(abs(math.sin(phi)), 1e-4)
                f_tip_exp = - (self.B * (self.R - r)) / max(2.0 * r * sin_phi, 1e-5)
                f_hub_exp = - (self.B * (r - self.r_hub)) / max(2.0 * self.r_hub * sin_phi, 1e-5)
                
                f_tip_val = math.exp(max(f_tip_exp, -20.0))
                f_hub_val = math.exp(max(f_hub_exp, -20.0))
                
                f_tip = (2.0 / math.pi) * math.acos(min(1.0, max(-1.0, f_tip_val)))
                f_hub = (2.0 / math.pi) * math.acos(min(1.0, max(-1.0, f_hub_val)))
                f_total = max(f_tip * f_hub, 1e-4)

                c_z = c_l * math.cos(phi) - c_d * math.sin(phi)
                kappa = (solidity * c_z) / max(8.0 * f_total * sin_phi, 1e-5)

                # High-thrust momentum correction with protected square root
                if kappa < 0.25 and v_inf > 0.1:
                    radicand = max(0.0, 1.0 + 4.0 * kappa * (v_eff / v_inf))
                    v_induced_new = (v_inf / 2.0) * (-1.0 + math.sqrt(radicand))
                else:
                    v_induced_new = max(0.0, v_eff * kappa)

                v_induced = 0.65 * v_induced + 0.35 * v_induced_new
                phi_new = math.atan2(v_inf + v_induced, v_theta)
                
                if abs(phi_new - phi) < 1e-4:
                    break
                phi = phi_new

            c_z = c_l * math.cos(phi) - c_d * math.sin(phi)
            c_x = c_l * math.sin(phi) + c_d * math.cos(phi)

            d_thrust = 0.5 * self.rho * (v_eff ** 2) * chord * c_z * self.B * self.dr
            d_torque = 0.5 * self.rho * (v_eff ** 2) * chord * c_x * self.B * r * self.dr

            total_thrust_n += max(0.0, d_thrust)
            total_torque_nm += max(0.0, d_torque)

        aero_power_watts = total_torque_nm * omega_rad_s
        n_rev_s = omega_rpm / 60.0
        diameter = 2.0 * self.R
        
        c_t = total_thrust_n / max(1e-5, self.rho * (n_rev_s ** 2) * (diameter ** 4))
        c_p = aero_power_watts / max(1e-5, self.rho * (n_rev_s ** 3) * (diameter ** 5))

        disk_area = math.pi * (self.R ** 2)
        ideal_power = (max(0.0, total_thrust_n) ** 1.5) / math.sqrt(2.0 * self.rho * disk_area)
        figure_of_merit = ideal_power / max(aero_power_watts, 1e-3)

        return {
            "rotational_speed_rpm": omega_rpm,
            "axial_flight_speed_ms": freestream_axial_velocity_ms,
            "thrust_output_n": round(total_thrust_n, 3),
            "torque_load_nm": round(total_torque_nm, 4),
            "mechanical_power_watts": round(aero_power_watts, 2),
            "thrust_coefficient_ct": round(c_t, 5),
            "power_coefficient_cp": round(c_p, 5),
            "figure_of_merit_fm": round(float(min(1.0, figure_of_merit)), 3),
            "thrust_to_weight_4x_payload_6kg": round((4.0 * total_thrust_n) / (6.14 * 9.80665), 2)
        }

    # Backward compatibility alias
    def solve_inflow_and_thrust(self, rpm: float, v_inf_ms: float = 0.0, **kwargs) -> Dict:
        res = self.compute_bemt_performance(omega_rpm=rpm, freestream_axial_velocity_ms=v_inf_ms)
        return {
            "rpm": rpm,
            "thrust_n": res["thrust_output_n"],
            "advance_ratio_j": round(v_inf_ms / max(1e-3, (rpm/60.0) * 2.0 * self.R), 3)
        }