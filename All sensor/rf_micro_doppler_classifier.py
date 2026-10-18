import math
import numpy as np
import scipy.signal as signal
from typing import Dict, Tuple, List

class RFMicroDopplerClassifier:
    def __init__(
        self,
        radar_carrier_freq_ghz: float = 24.125, # 24 GHz K-band radar sensor
        sampling_rate_hz: int = 16000,
        stft_nperseg: int = 256,
        stft_noverlap: int = 224
    ):
        self.fc = radar_carrier_freq_ghz * 1e9
        self.c = 299792458.0
        self.lambda_c = self.c / self.fc
        self.fs = sampling_rate_hz
        self.nperseg = stft_nperseg
        self.noverlap = stft_noverlap

    def generate_synthetic_micro_doppler_iq(
        self,
        target_type: str = "QUADCOPTER",
        duration_s: float = 0.5,
        radial_velocity_ms: float = 8.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes target body return and micro-Doppler rotor blade tip reflections.
        """
        n_samples = int(self.fs * duration_s)
        t = np.arange(n_samples) / self.fs

        # 1. Main Body Doppler Shift
        fd_body = 2.0 * radial_velocity_ms / self.lambda_c
        body_signal = 1.0 * np.exp(1j * 2.0 * np.pi * fd_body * t)

        # 2. Micro-Doppler Blade Components
        micro_signal = np.zeros(n_samples, dtype=complex)

        if target_type == "QUADCOPTER":
            # 4 rotors, 2 blades each, 4800 RPM (80 Hz rotation, 160 Hz BPF), R_tip = 0.19m
            omega_rot = 2.0 * math.pi * 80.0
            r_tip = 0.1905
            fd_tip_max = (4.0 * math.pi * self.fc / self.c) * 80.0 * r_tip

            for r_idx in range(4):
                phase_offset = r_idx * (math.pi / 2.0)
                # Blade flash phase modulation
                flash = 0.18 * np.exp(1j * (2.0 * np.pi * fd_body * t + (fd_tip_max / 80.0) * np.sin(omega_rot * t + phase_offset)))
                micro_signal += flash

        elif target_type == "BIRD_AVIAN_CLUTTER":
            # Wing flapping frequency ~ 4.5 Hz with low harmonic spread
            omega_flap = 2.0 * math.pi * 4.5
            flap_mod = 0.35 * np.exp(1j * (2.0 * np.pi * fd_body * t + 8.0 * np.sin(omega_flap * t)))
            micro_signal += flap_mod

        elif target_type == "FIXED_WING_DRONE":
            # Single nose propeller at 6500 RPM (108.3 Hz), R_tip = 0.12m
            omega_rot = 2.0 * math.pi * 108.3
            fd_tip_max = (4.0 * math.pi * self.fc / self.c) * 108.3 * 0.12
            micro_signal = 0.25 * np.exp(1j * (2.0 * np.pi * fd_body * t + (fd_tip_max / 108.3) * np.sin(omega_rot * t)))

        total_iq = body_signal + micro_signal + (np.random.normal(0, 0.08, n_samples) + 1j * np.random.normal(0, 0.08, n_samples))
        return t, total_iq

    def analyze_micro_doppler_signature(self, iq_samples: np.ndarray) -> Dict:
        """
        Extracts STFT spectrogram, body velocity, micro-Doppler bandwidth, and classifies target.
        """
        f, t, zxx = signal.stft(
            iq_samples,
            fs=self.fs,
            window='hann',
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            return_onesided=False
        )

        # Shift zero-frequency to center
        f_shifted = np.fft.fftshift(f)
        spectrogram = np.fft.fftshift(np.abs(zxx), axes=0)
        power_spec = np.mean(spectrogram, axis=1)

        # 1. Estimate Bulk Radial Velocity from Peak Doppler
        peak_freq_idx = int(np.argmax(power_spec))
        fd_peak = f_shifted[peak_freq_idx]
        v_radial_est = (fd_peak * self.lambda_c) / 2.0

        # 2. Extract Micro-Doppler Bandwidth Spread (10 dB threshold above noise floor)
        noise_floor = np.percentile(power_spec, 15)
        signal_mask = power_spec > (noise_floor * 3.5)
        if np.any(signal_mask):
            f_active = f_shifted[signal_mask]
            bw_micro_doppler_hz = float(np.max(f_active) - np.min(f_active))
        else:
            bw_micro_doppler_hz = 50.0

        # 3. Harmonic Autocorrelation to Extract Blade Flash Repetition Rate
        temporal_envelope = np.max(spectrogram, axis=0)
        autocorr = signal.correlate(temporal_envelope - np.mean(temporal_envelope),
                                    temporal_envelope - np.mean(temporal_envelope),
                                    mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peaks, _ = signal.find_peaks(autocorr[5:], distance=10)

        flash_freq_hz = 0.0
        if len(peaks) > 0:
            sample_period = (t[1] - t[0]) * peaks[0]
            flash_freq_hz = 1.0 / max(sample_period, 1e-4)

        # 4. Feature-Based Threat Classification Rule
        classification = "UNKNOWN_RADAR_TARGET"
        confidence = 0.50

        if bw_micro_doppler_hz > 1800.0 and flash_freq_hz >= 120.0:
            classification = "MULTI_ROTOR_UAV_INTRUDER"
            confidence = 0.94
        elif bw_micro_doppler_hz > 1000.0 and flash_freq_hz >= 90.0:
            classification = "FIXED_WING_UAV_INTRUDER"
            confidence = 0.88
        elif bw_micro_doppler_hz < 450.0 and flash_freq_hz <= 15.0:
            classification = "AVIAN_BIOLOGICAL_CLUTTER"
            confidence = 0.91

        return {
            "radar_carrier_ghz": self.fc / 1e9,
            "estimated_radial_velocity_ms": round(float(v_radial_est), 2),
            "micro_doppler_bandwidth_hz": round(bw_micro_doppler_hz, 1),
            "detected_blade_flash_rate_hz": round(flash_freq_hz, 1),
            "target_classification": classification,
            "classification_confidence": round(confidence, 2),
            "threat_action": "ENGAGE_DAA_TACTICAL_SEPARATION" if "UAV_INTRUDER" in classification else "SUPPRESS_RADAR_CLUTTER"
        }

if __name__ == "__main__":
    radar_classifier = RFMicroDopplerClassifier()

    print("==========================================================================")
    print("      MULTI-STATIC RADAR MICRO-DOPPLER INTRUDER CLASSIFIER AUDIT          ")
    print("==========================================================================")

    # 1. Test Multi-Rotor Drone Signature
    _, iq_quad = radar_classifier.generate_synthetic_micro_doppler_iq("QUADCOPTER", radial_velocity_ms=8.5)
    res_quad = radar_classifier.analyze_micro_doppler_signature(iq_quad)
    print(f"Target #1 (Synthetic Quadcopter) -> Class: {res_quad['target_classification']:26s} | "
          f"Conf: {res_quad['classification_confidence']*100:.0f}% | V_rad: {res_quad['estimated_radial_velocity_ms']:+4.1f} m/s | "
          f"BW_mD: {res_quad['micro_doppler_bandwidth_hz']:6.1f} Hz | Flash Rate: {res_quad['detected_blade_flash_rate_hz']:5.1f} Hz")

    # 2. Test Bird / Biological Clutter Signature
    _, iq_bird = radar_classifier.generate_synthetic_micro_doppler_iq("BIRD_AVIAN_CLUTTER", radial_velocity_ms=4.2)
    res_bird = radar_classifier.analyze_micro_doppler_signature(iq_bird)
    print(f"Target #2 (Synthetic Bird Flock) -> Class: {res_bird['target_classification']:26s} | "
          f"Conf: {res_bird['classification_confidence']*100:.0f}% | V_rad: {res_bird['estimated_radial_velocity_ms']:+4.1f} m/s | "
          f"BW_mD: {res_bird['micro_doppler_bandwidth_hz']:6.1f} Hz | Flash Rate: {res_bird['detected_blade_flash_rate_hz']:5.1f} Hz")
    print("==========================================================================\n")