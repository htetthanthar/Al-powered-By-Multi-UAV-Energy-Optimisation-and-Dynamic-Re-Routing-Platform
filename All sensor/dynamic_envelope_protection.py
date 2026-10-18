import math
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, List

class DynamicFlightEnvelopeProtector:
    def __init__(
        self,
        uav_mass_kg: float = 6.14,
        max_load_factor_g: float = 2.2,        # Structural limit n_z_max
        min_load_factor_g: float = 0.2,        # Zero-g / negative-g floor
        max_bank_angle_deg: float = 38.0,
        max_pitch_angle_deg: float = 25.0,
        stall_aoa_clean_deg: float = 14.5,
        cbf_gamma_gain: float = 5.0
    ):
        self.m = uav_mass_kg
        self.nz_max = max_load_factor_g
        self.nz_min = min_load_factor_g
        self.phi_max = math.radians(max_bank_angle_deg)
        self.theta_max = math.radians(max_pitch_angle_deg)
        self.alpha_stall_clean = math.radians(stall_aoa_clean_deg)
        self.gamma = cbf_gamma_gain
        self.g = 9.80665

    def compute_degraded_envelope_limits(
        self,
        icing_mass_kg: float,
        actuator_health_fraction: float
    ) -> Dict[str, float]:
        """
        Dynamically derates aerodynamic envelope limits based on ice mass and actuator degradation.
        """
        # Icing reduces effective stall margin: ~1.5 deg loss per 100g ice
        delta_alpha_ice = math.radians(15.0 * (icing_mass_kg / 1.0))
        alpha_stall_eff = max(math.radians(6.0), self.alpha_stall_clean - delta_alpha_ice)

        # Structural G-limit derating based on actuator capability
        nz_max_eff = max(1.2, self.nz_max * actuator_health_fraction)
        
        # Derate maximum permissible bank angle: cos(phi_max) = 1 / n_z_max
        phi_max_eff = math.acos(np.clip(1.0 / nz_max_eff, 0.2, 1.0))
        phi_max_eff = min(self.phi_max, phi_max_eff)

        return {
            "effective_stall_aoa_deg": round(math.degrees(alpha_stall_eff), 2),
            "effective_nz_max_g": round(nz_max_eff, 2),
            "effective_max_bank_angle_deg": round(math.degrees(phi_max_eff), 2)
        }

    def filter_control_commands_cbf(
        self,
        current_attitude_rad: Tuple[float, float], # (roll_phi, pitch_theta)
        current_rates_rad_s: Tuple[float, float],  # (p_roll_rate, q_pitch_rate)
        commanded_rates_rad_s: Tuple[float, float], # (p_cmd, q_cmd)
        current_airspeed_ms: float,
        icing_mass_kg: float = 0.0,
        actuator_health_fraction: float = 1.0
    ) -> Dict:
        """
        Solves QP filter on angular rate inputs (p_cmd, q_cmd) to maintain attitude inside safe envelope.
        Barrier Functions:
          h1(phi)   = phi_max^2 - phi^2 >= 0
          h2(theta) = theta_max^2 - theta^2 >= 0
        """
        phi, theta = current_attitude_rad
        p_curr, q_curr = current_rates_rad_s
        u_nom = np.array(commanded_rates_rad_s)

        limits = self.compute_degraded_envelope_limits(icing_mass_kg, actuator_health_fraction)
        phi_limit = math.radians(limits["effective_max_bank_angle_deg"])
        theta_limit = self.theta_max

        # Barrier 1: Roll Envelope Protection
        h1 = (phi_limit ** 2) - (phi ** 2)
        # L_f h1 + L_g h1 * p >= -gamma * h1  =>  -2*phi*p >= -gamma * (phi_limit^2 - phi^2)
        # => 2*phi * p <= gamma * h1
        a_cbf_1 = 2.0 * phi
        b_cbf_1 = self.gamma * h1

        # Barrier 2: Pitch Envelope Protection
        h2 = (theta_limit ** 2) - (theta ** 2)
        a_cbf_2 = 2.0 * theta
        b_cbf_2 = self.gamma * h2

        # Quadratic Program: min 0.5 * ||u - u_nom||^2  s.t.  A_cbf * u <= b_cbf
        def qp_objective(u):
            return 0.5 * np.sum((u - u_nom) ** 2)

        constraints = [
            {"type": "ineq", "fun": lambda u: b_cbf_1 - a_cbf_1 * u[0]},
            {"type": "ineq", "fun": lambda u: b_cbf_2 - a_cbf_2 * u[1]}
        ]

        bounds = [(-math.radians(120.0), math.radians(120.0)),
                  (-math.radians(90.0),  math.radians(90.0))]

        res = minimize(qp_objective, u_nom, method="SLSQP", bounds=bounds, constraints=constraints)
        u_safe = res.x if res.success else np.clip(u_nom, -0.5, 0.5)

        # Calculate current flight load factor: n_z = 1 / cos(phi)
        nz_current = 1.0 / max(math.cos(phi), 0.1)

        is_intervening = float(np.linalg.norm(u_safe - u_nom)) > 1e-3

        return {
            "commanded_rates_deg_s": (round(math.degrees(u_nom[0]), 2), round(math.degrees(u_nom[1]), 2)),
            "safe_filtered_rates_deg_s": (round(math.degrees(u_safe[0]), 2), round(math.degrees(u_safe[1]), 2)),
            "cbf_active_intervention": is_intervening,
            "current_load_factor_nz_g": round(nz_current, 2),
            "effective_limits": limits,
            "barrier_margins": (round(h1, 4), round(h2, 4)),
            "protection_status": "ENVELOPE_PROTECTION_INTERVENING" if is_intervening else "WITHIN_NOMINAL_ENVELOPE"
        }

if __name__ == "__main__":
    protector = DynamicFlightEnvelopeProtector()

    print("==========================================================================")
    print("      DYNAMIC FLIGHT ENVELOPE PROTECTION & CONTROL BARRIER AUDIT          ")
    print("==========================================================================")
    
    # Simulate aggressive roll command (p_cmd = +85 deg/s) while already banked at 34° (near 38° limit)
    # with 120g of rotor ice accretion
    audit = protector.filter_control_commands_cbf(
        current_attitude_rad=(math.radians(34.0), math.radians(4.0)),
        current_rates_rad_s=(math.radians(10.0), math.radians(0.0)),
        commanded_rates_rad_s=(math.radians(85.0), math.radians(0.0)),
        current_airspeed_ms=18.05,
        icing_mass_kg=0.120,
        actuator_health_fraction=0.90
    )

    for k, v in audit.items():
        if k != "effective_limits":
            print(f"  • {k:35s}: {v}")
    print("\nDerated Physical Envelope Limits:")
    for k, v in audit["effective_limits"].items():
        print(f"    - {k:32s}: {v}")
    print("==========================================================================\n")