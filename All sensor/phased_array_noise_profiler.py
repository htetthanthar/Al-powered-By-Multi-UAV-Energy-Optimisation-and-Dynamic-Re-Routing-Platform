import math
import numpy as np
from typing import Dict, List, Tuple

class GroundAcousticBeamformer:
    def __init__(
        self,
        num_microphones: int = 8,
        array_radius_m: float = 0.75,
        sampling_rate_hz: int = 48000,
        speed_of_sound_ms: float = 343.0
    ):
        self.num_mics = num_microphones
        self.r_array = array_radius_m
        self.fs = sampling_rate_hz
        self.c_sound = speed_of_sound_ms

        # Microphone array geometry in ground plane (Circular planar array)
        self.mic_positions = []
        for m in range(num_microphones):
            angle = m * (2.0 * math.pi / num_microphones)
            self.mic_positions.append(np.array([
                self.r_array * math.cos(angle),
                self.r_array * math.sin(angle),
                0.0
            ]))

    def compute_steering_delays_samples(
        self,
        uav_pos_ned_m: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Calculates geometric time-of-flight sample delays relative to array origin.
        """
        r_uav = np.array(uav_pos_ned_m)
        d_center = np.linalg.norm(r_uav)
        delays_samples = np.zeros(self.num_mics)

        for m in range(self.num_mics):
            d_mic = np.linalg.norm(r_uav - self.mic_positions[m])
            tau_sec = (d_mic - d_center) / self.c_sound
            delays_samples[m] = tau_sec * self.fs

        return delays_samples

    def process_beamformed_audio_frame(
        self,
        multichannel_audio: np.ndarray,
        uav_pos_ned_m: Tuple[float, float, float]
    ) -> Dict:
        """
        Executes Delay-and-Sum beamforming steered towards UAV 3D coordinates.
        multichannel_audio shape: (num_mics, num_samples)
        """
        num_samples = multichannel_audio.shape[1]
        delays_samples = self.compute_steering_delays_samples(uav_pos_ned_m)
        
        # Shift and sum across microphone channels
        beamformed_signal = np.zeros(num_samples)
        
        for m in range(self.num_mics):
            shift = int(round(delays_samples[m]))
            if shift > 0:
                aligned = np.pad(multichannel_audio[m, :-shift], (shift, 0), mode='constant')
            elif shift < 0:
                aligned = np.pad(multichannel_audio[m, -shift:], (0, -shift), mode='constant')
            else:
                aligned = multichannel_audio[m]
            
            beamformed_signal += aligned

        beamformed_signal /= self.num_mics

        # Root-Mean-Square (RMS) Acoustic Pressure
        rms_pressure = float(np.sqrt(np.mean(beamformed_signal ** 2)))
        p_ref = 20e-6 # 20 uPa reference threshold
        
        spl_linear_db = 20.0 * math.log10(max(rms_pressure / p_ref, 1.0))
        # A-weighting correction for blade-passing fundamental frequency
        spl_a_weighted_dba = max(20.0, spl_linear_db - 16.1)

        # Community noise compliance check (< 55 dBA limit in residential zones)
        is_compliant = spl_a_weighted_dba <= 55.0

        return {
            "steered_uav_coords_ned_m": uav_pos_ned_m,
            "rms_acoustic_pressure_pa": round(rms_pressure, 5),
            "beamformed_spl_linear_db": round(spl_linear_db, 2),
            "ground_noise_exposure_laeq_dba": round(spl_a_weighted_dba, 2),
            "municipal_noise_limit_dba": 55.0,
            "community_noise_compliant": is_compliant
        }

if __name__ == "__main__":
    array = GroundAcousticBeamformer(num_microphones=8)

    # Simulate multichannel acoustic audio from a drone cruising at (x=30m, y=40m, z=-25m)
    uav_loc = (30.0, 40.0, -25.0)
    fs = 48000
    n_samples = 4800 # 100 ms frame
    t = np.linspace(0, 0.1, n_samples)
    
    # Fundamental BPF: 160 Hz + Broadband Turbulent Noise
    bpf_tone = 0.045 * np.sin(2.0 * math.pi * 160.0 * t) # Pressure in Pa
    
    synthetic_audio = np.zeros((8, n_samples))
    delays = array.compute_steering_delays_samples(uav_loc)
    
    for m in range(8):
        shift = int(round(delays[m]))
        noise = np.random.normal(0, 0.005, n_samples)
        synthetic_audio[m] = np.roll(bpf_tone, shift) + noise

    profile = array.process_beamformed_audio_frame(synthetic_audio, uav_loc)

    print("==========================================================================")
    print("      GROUND PHASED ARRAY ACOUSTIC BEAMFORMER NOISE PROFILER AUDIT       ")
    print("==========================================================================")
    for k, v in profile.items():
        print(f"  • {k:32s}: {v}")
    print("==========================================================================\n")