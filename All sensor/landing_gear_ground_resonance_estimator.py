import math
import numpy as np
from scipy.linalg import eig
from typing import Dict, Tuple, List

class LandingGearGroundResonanceAnalyzer:
    def __init__(
        self,
        uav_mass_kg: float = 6.80,
        rotor_speed_rad_s: float = 471.24, # 4500 RPM
        blade_lead_lag_freq_hz: float = 28.5,
        blade_mass_kg: float = 0.085,
        rotor_hinge_offset_m: float = 0.035,
        piston_area_m2: float = 0.00035,   # ~21mm diameter strut
        gas_volume_v0_m3: float = 1.2e-5,  # 12 cm^3 precharge volume
        precharge_pressure_p0_pa: float = 4.5e5, # 4.5 bar
        hydraulic_discharge_cd: float = 0.62,
        orifice_area_m2: float = 3.2e-6,
        polytropic_exponent_gamma: float = 1.35
    ):
        self.m = uav_mass_kg
        self.omega_rot = rotor_speed_rad_s
        self.omega_zeta = 2.0 * math.pi * blade_lead_lag_freq_hz
        self.m_blade = blade_mass_kg
        self.e_hinge = rotor_hinge_offset_m
        self.a_piston = piston_area_m2
        self.v0 = gas_volume_v0_m3
        self.p0 = precharge_pressure_p0_pa
        self.cd = hydraulic_discharge_cd
        self.a_orifice = orifice_area_m2
        self.gamma = polytropic_exponent_gamma
        self.rho_hyd = 870.0 # Hydraulic fluid density kg/m^3

    def compute_strut_force_and_damping(
        self,
        stroke_compression_m: float,
        stroke_velocity_ms: float
    ) -> Tuple[float, float, float]:
        """
        Calculates oleo-pneumatic spring-damper force F_strut, tangent stiffness K, and damping C.
        """
        x = np.clip(stroke_compression_m, 0.0, (self.v0 / self.a_piston) * 0.85)
        v = stroke_velocity_ms

        # 1. Polytropic Gas Spring Force: F_gas = P0 * A * (V0 / (V0 - A*x))^gamma
        vol_comp = max(self.v0 - self.a_piston * x, 1e-6)
        f_pneumatic = self.p0 * self.a_piston * ((self.v0 / vol_comp) ** self.gamma)

        # Tangent Pneumatic Stiffness K_gas = dF/dx
        k_stiffness = (self.gamma * self.p0 * (self.a_piston ** 2) / self.v0) * ((self.v0 / vol_comp) ** (self.gamma + 1.0))

        # 2. Hydraulic Orifice Damping Force: F_hyd = 0.5 * rho * (A^3 / (Cd*Ao)^2) * v^2 * sign(v)
        c_hyd_coeff = 0.5 * self.rho_hyd * (self.a_piston ** 3) / ((self.cd * self.a_orifice) ** 2)
        f_hydraulic = c_hyd_coeff * (abs(v) * v)
        c_damping = 2.0 * c_hyd_coeff * max(abs(v), 0.01)

        total_strut_force = f_pneumatic + f_hydraulic
        return float(total_strut_force), float(k_stiffness), float(c_damping)

    def evaluate_ground_resonance_stability(
        self,
        strut_compression_m: float,
        strut_velocity_ms: float
    ) -> Dict:
        """
        Builds Coleman 4-DOF coupled airframe roll-pitch and rotor regressing lead-lag matrix.
        Solves generalized eigenvalue problem: (lambda^2 * M + lambda * C + K) * q = 0.
        """
        _, k_strut, c_strut = self.compute_strut_force_and_damping(strut_compression_m, strut_velocity_ms)

        # Coupled 4-DOF Coordinates: q = [x_fuselage, y_fuselage, zeta_cos, zeta_sin]^T
        # Fuselage + Landing Gear Effective Parameters
        m_x = self.m
        m_y = self.m
        k_x = 2.0 * k_strut
        k_y = 2.0 * k_strut
        c_x = 2.0 * c_strut
        c_y = 2.0 * c_strut

        # Rotor In-Plane Lead-Lag Inertial Coupling
        s_zeta = 4.0 * (self.m_blade * self.e_hinge) # First mass moment
        i_zeta = 4.0 * (self.m_blade * (0.280 ** 2) / 3.0) # Blade polar inertia

        # Mass Matrix M (4x4)
        M = np.diag([m_x, m_y, i_zeta, i_zeta])
        M[0, 3] = s_zeta
        M[1, 2] = -s_zeta
        M[2, 1] = -s_zeta
        M[3, 0] = s_zeta

        # Damping Matrix C (4x4) including Coriolis terms
        C = np.diag([c_x, c_y, 0.05 * i_zeta, 0.05 * i_zeta])
        C[2, 3] = 2.0 * i_zeta * self.omega_rot
        C[3, 2] = -2.0 * i_zeta * self.omega_rot

        # Stiffness Matrix K (4x4)
        k_lead_lag = i_zeta * (self.omega_zeta ** 2 - self.omega_rot ** 2)
        K = np.diag([k_x, k_y, k_lead_lag, k_lead_lag])

        # Convert to 1st-order State-Space Matrix A (8x8): d/dt [q, q_dot] = A * [q, q_dot]
        M_inv = np.linalg.inv(M)
        A_sys = np.zeros((8, 8))
        A_sys[0:4, 4:8] = np.eye(4)
        A_sys[4:8, 0:4] = -M_inv @ K
        A_sys[4:8, 4:8] = -M_inv @ C

        eigenvalues = np.linalg.eigvals(A_sys)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        max_real_eigenvalue = float(np.max(real_parts))
        is_resonant_unstable = max_real_eigenvalue >= 0.0

        # Modal damping ratios: zeta_i = -Re(lambda) / |lambda|
        natural_frequencies = np.abs(eigenvalues)
        damping_ratios = -real_parts / np.maximum(natural_frequencies, 1e-4)

        return {
            "strut_compression_stroke_mm": round(strut_compression_m * 1000.0, 1),
            "pneumatic_strut_stiffness_n_m": round(k_strut, 1),
            "hydraulic_damping_coeff_ns_m": round(c_strut, 2),
            "dominant_modal_growth_rate_real": round(max_real_eigenvalue, 4),
            "min_modal_damping_ratio": round(float(np.min(damping_ratios)), 4),
            "rotor_lead_lag_frequency_hz": round(self.omega_zeta / (2.0 * math.pi), 2),
            "ground_resonance_critical": is_resonant_unstable,
            "damping_verdict": "ELASTODYNAMICALLY_STABLE" if not is_resonant_unstable else "CRITICAL_GROUND_RESONANCE_DIVERGENCE"
        }

if __name__ == "__main__":
    analyzer = LandingGearGroundResonanceAnalyzer()

    print("==========================================================================")
    print("      LANDING GEAR ELASTODYNAMICS & GROUND RESONANCE MODAL AUDIT          ")
    print("==========================================================================")

    # Evaluate ground resonance across compression strokes (0.5cm to 4.0cm) during touchdown
    for stroke_cm in [0.5, 1.5, 2.5, 3.8]:
        report = analyzer.evaluate_ground_resonance_stability(
            strut_compression_m=stroke_cm * 1e-2,
            strut_velocity_ms=0.25
        )
        print(f"Stroke: {report['strut_compression_stroke_mm']:4.1f} mm | Stiffness: {report['pneumatic_strut_stiffness_n_m']:7.1f} N/m | "
              f"Damping: {report['hydraulic_damping_coeff_ns_m']:6.2f} N·s/m | Max Re(λ): {report['dominant_modal_growth_rate_real']:+7.4f} | "
              f"Status: {report['damping_verdict']}")
    print("==========================================================================\n")