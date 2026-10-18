import math
import numpy as np
from typing import Dict, List, Tuple

class InFlightBatteryEISTracker:
    def __init__(
        self,
        frequencies_hz: List[float] = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0],
        sampling_rate_hz: int = 10000,
        nominal_r0_milliohms: float = 12.5,
        nominal_rct_milliohms: float = 18.0
    ):
        self.frequencies = frequencies_hz
        self.fs = sampling_rate_hz
        self.r0_base = nominal_r0_milliohms
        self.rct_base = nominal_rct_milliohms
        self.last_nyquist_spectrum: List[Dict] = []

    def compute_dft_impedance_point(
        self,
        current_signal_a: np.ndarray,
        voltage_signal_v: np.ndarray,
        freq_hz: float
    ) -> Tuple[float, float]:
        """
        Calculates complex impedance Z(omega) = V(omega) / I(omega) via single-frequency DFT.
        Returns: (Z_real_milliOhms, Z_imag_milliOhms)
        """
        n_samples = len(current_signal_a)
        t = np.arange(n_samples) / self.fs
        
        # Discrete Fourier kernel
        dft_kernel = np.exp(-2j * np.pi * freq_hz * t)
        
        i_fourier = np.sum(current_signal_a * dft_kernel) / n_samples
        v_fourier = np.sum(voltage_signal_v * dft_kernel) / n_samples

        z_complex = (v_fourier / (i_fourier + 1e-9)) * 1000.0 # Convert to milli-Ohms
        return float(np.real(z_complex)), float(np.imag(z_complex))

    def extract_equivalent_circuit_parameters(
        self,
        nyquist_points: List[Tuple[float, float, float]] # (freq, Z_real, Z_imag)
    ) -> Dict:
        """
        Fits Randles equivalent circuit: [R0, Rct, Cdl, Warburg_sigma]
        """
        z_reals = [p[1] for p in nyquist_points]
        z_imags = [p[2] for p in nyquist_points]
        freqs = [p[0] for p in nyquist_points]

        # 1. Bulk Ohmic Resistance R0 (Highest frequency real intercept)
        r0_est = min(z_reals)

        # 2. Charge Transfer Resistance Rct (Semicircle diameter)
        # Find apex frequency where -Z_imag is maximized
        apex_idx = int(np.argmin(z_imags)) # z_imags is negative in capacitive quadrant
        apex_freq = freqs[apex_idx]
        max_imag_val = abs(z_imags[apex_idx])
        
        # Diameter of semicircular arc ~ 2 * max(-Z_imag)
        rct_est = 2.0 * max_imag_val

        # 3. Double-Layer Capacitance Cdl = 1 / (omega_apex * Rct)
        omega_apex = 2.0 * math.pi * apex_freq
        c_dl_farads = 1.0 / (omega_apex * (rct_est * 1e-3))

        # Degradation & Lithium Plating Diagnostic Flags
        r0_growth_pct = ((r0_est - self.r0_base) / self.r0_base) * 100.0
        rct_growth_pct = ((rct_est - self.rct_base) / self.rct_base) * 100.0

        status = "ELECTROCHEMICALLY_HEALTHY"
        if rct_growth_pct >= 65.0:
            status = "CRITICAL_LITHIUM_PLATING_DENDRITE_RISK"
        elif r0_growth_pct >= 40.0:
            status = "WARNING_BULK_ELECTROLYTE_OXIDATION"

        return {
            "bulk_resistance_r0_mohm": round(r0_est, 2),
            "charge_transfer_resistance_rct_mohm": round(rct_est, 2),
            "double_layer_capacitance_f": round(c_dl_farads, 3),
            "semicircle_apex_frequency_hz": apex_freq,
            "r0_impedance_growth_pct": round(r0_growth_pct, 1),
            "rct_impedance_growth_pct": round(rct_growth_pct, 1),
            "electrochemical_diagnostic_status": status
        }

if __name__ == "__main__":
    eis_tracker = InFlightBatteryEISTracker()

    print("==========================================================================")
    print("      IN-FLIGHT ELECTROCHEMICAL IMPEDANCE SPECTROSCOPY (EIS) AUDIT        ")
    print("==========================================================================")

    # Synthesize multi-sine perturbation response for a degraded pack (R0=14.2mOhm, Rct=26.5mOhm)
    duration_s = 1.0
    N = int(eis_tracker.fs * duration_s)
    t_vec = np.arange(N) / eis_tracker.fs

    spectrum_results = []
    
    for f in eis_tracker.frequencies:
        # 1.0A sinusoidal perturbation current
        i_t = 14.5 + 0.8 * np.sin(2.0 * math.pi * f * t_vec)
        
        # Randles model impedance response
        omega = 2.0 * math.pi * f
        c_dl = 0.12 # Farads
        z_dl = 1.0 / (1j * omega * c_dl)
        z_randles = (14.2e-3) + (26.5e-3 * z_dl) / (26.5e-3 + z_dl)
        
        v_base = 22.4 - 14.5 * 0.0142
        v_t = v_base - (0.8 * abs(z_randles) * np.sin(2.0 * math.pi * f * t_vec + np.angle(z_randles))) + np.random.normal(0, 0.001, N)

        z_r, z_i = eis_tracker.compute_dft_impedance_point(i_t, v_t, f)
        spectrum_results.append((f, z_r, z_i))

    ecm_report = eis_tracker.extract_equivalent_circuit_parameters(spectrum_results)

    print("Nyquist Spectrum Sample Points (Z' + j Z''):")
    for pt in spectrum_results[::2]:
        print(f"  • Frequency {pt[0]:6.1f} Hz : Z = {pt[1]:5.2f} {pt[2]:+5.2f}j mΩ")

    print("\nExtracted Electrochemical Randles Parameters:")
    for k, v in ecm_report.items():
        print(f"  • {k:38s}: {v}")
    print("==========================================================================\n")