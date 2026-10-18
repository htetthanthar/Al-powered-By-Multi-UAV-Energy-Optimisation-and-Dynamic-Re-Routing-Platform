import math
import numpy as np
from typing import Dict, List, Tuple

class RotorSynchrophasingController:
    def __init__(
        self,
        num_rotors: int = 4,
        blade_count: int = 2,
        rotor_arm_radius_m: float = 0.325,
        nominal_rpm: float = 4800.0,
        speed_of_sound_ms: float = 343.0
    ):
        self.num_rotors = num_rotors
        self.b = blade_count
        self.r_arm = rotor_arm_radius_m
        self.rpm = nominal_rpm
        self.c_sound = speed_of_sound_ms

        # Rotor positions in body frame: (x, y, z)
        self.rotor_locs = [
            ( self.r_arm * 0.707,  self.r_arm * 0.707, 0.0), # Motor 1 (Front-Right, CCW)
            (-self.r_arm * 0.707,  self.r_arm * 0.707, 0.0), # Motor 2 (Rear-Right, CW)
            (-self.r_arm * 0.707, -self.r_arm * 0.707, 0.0), # Motor 3 (Rear-Left, CCW)
            ( self.r_arm * 0.707, -self.r_arm * 0.707, 0.0)  # Motor 4 (Front-Left, CW)
        ]
        
        # Blade Passing Frequency (BPF)
        self.omega_rot_rad_s = (self.rpm / 60.0) * 2.0 * math.pi
        self.bpf_rad_s = self.b * self.omega_rot_rad_s
        self.k_acoustic = self.bpf_rad_s / self.c_sound # Acoustic wavenumber

    def compute_farfield_spl_dba(
        self,
        phase_offsets_rad: np.ndarray,
        ground_receiver_ned: Tuple[float, float, float] = (0.0, 0.0, 25.0)
    ) -> float:
        """
        Integrates acoustic pressure phasors from all 4 rotors at receiver position r:
          P_total = sum_k A_k * exp(j * (phi_k - k_ac * d_k))
        """
        gx, gy, gz = ground_receiver_ned
        real_sum = 0.0
        imag_sum = 0.0

        for idx, (rx, ry, rz) in enumerate(self.rotor_locs):
            dist_k = math.sqrt((gx - rx)**2 + (gy - ry)**2 + (gz - rz)**2)
            # Phase delay = Geometric propagation phase + Motor azimuthal phase
            total_phase = phase_offsets_rad[idx] - self.k_acoustic * dist_k
            
            # Amplitude proportional to 1/r
            amp = 1.0 / dist_k
            real_sum += amp * math.cos(total_phase)
            imag_sum += amp * math.sin(total_phase)

        pressure_amp = math.sqrt(real_sum**2 + imag_sum**2)
        
        # Relative Sound Pressure Level in dBA
        p_ref = 20e-6
        spl_relative_dba = 20.0 * math.log10(max(pressure_amp * 0.02 / p_ref, 1.0)) - 16.1
        return spl_relative_dba

    def optimize_destructive_phase_angles(
        self,
        ground_receiver_ned: Tuple[float, float, float]
    ) -> Dict:
        """
        Finds optimal phase offsets [theta_1, theta_2, theta_3, theta_4]
        to achieve destructive interference directly below the aircraft.
        """
        # Baseline: In-phase rotors (all phase offsets = 0)
        baseline_phase = np.array([0.0, 0.0, 0.0, 0.0])
        spl_baseline = self.compute_farfield_spl_dba(baseline_phase, ground_receiver_ned)

        # Optimize motor phase offsets via coordinate descent
        best_phases = np.zeros(4)
        best_spl = spl_baseline

        for iteration in range(25):
            for motor_id in range(1, 4): # Motor 0 fixed as reference
                for test_phase in np.linspace(0, 2 * math.pi, 36):
                    trial_phases = best_phases.copy()
                    trial_phases[motor_id] = test_phase
                    spl_trial = self.compute_farfield_spl_dba(trial_phases, ground_receiver_ned)
                    
                    if spl_trial < best_spl:
                        best_spl = spl_trial
                        best_phases[motor_id] = test_phase

        noise_attenuation_db = spl_baseline - best_spl

        return {
            "ground_target_receiver_ned": ground_receiver_ned,
            "bpf_frequency_hz": round((self.bpf_rad_s / (2.0 * math.pi)), 1),
            "baseline_in_phase_spl_dba": round(spl_baseline, 2),
            "optimized_synchrophased_spl_dba": round(best_spl, 2),
            "noise_reduction_dba": round(noise_attenuation_db, 2),
            "optimal_motor_phase_angles_deg": [round(math.degrees(p) % 360.0, 1) for p in best_phases],
            "acoustic_cancellation_active": noise_attenuation_db >= 3.0
        }

if __name__ == "__main__":
    anc = RotorSynchrophasingController(nominal_rpm=4800.0)
    opt = anc.optimize_destructive_phase_angles(ground_receiver_ned=(0.0, 0.0, 25.0))

    print("==========================================================================")
    print("      ACTIVE ROTOR SYNCHROPHASING & ACOUSTIC CANCELLATION AUDIT           ")
    print("==========================================================================")
    for k, v in opt.items():
        print(f"  • {k:35s}: {v}")
    print("==========================================================================\n")