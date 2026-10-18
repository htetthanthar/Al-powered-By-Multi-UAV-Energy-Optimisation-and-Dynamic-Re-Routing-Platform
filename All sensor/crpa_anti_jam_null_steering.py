# crpa_anti_jam_null_steering.py
import math
import numpy as np
from typing import Dict

class CRPAAntiJammingBeamformer:
    def __init__(self, num_elements: int = 4, element_spacing_m: float = 0.095):
        self.M = num_elements
        self.d = element_spacing_m
        self.fc = 1575.42e6
        self.lambda_c = 299792458.0 / self.fc
        half_d = self.d / 2.0
        self.element_positions = np.array([
            [ half_d,  half_d, 0.0],
            [-half_d,  half_d, 0.0],
            [-half_d, -half_d, 0.0],
            [ half_d, -half_d, 0.0]
        ])

    def compute_steering_vector(self, az_deg: float, el_deg: float) -> np.ndarray:
        az_rad, el_rad = math.radians(az_deg), math.radians(el_deg)
        k_vec = (2.0 * math.pi / self.lambda_c) * np.array([
            math.cos(el_rad) * math.cos(az_rad),
            math.cos(el_rad) * math.sin(az_rad),
            math.sin(el_rad)
        ])
        steering = np.zeros(self.M, dtype=complex)
        for m in range(self.M):
            steering[m] = np.exp(1j * np.dot(self.element_positions[m], k_vec))
        return steering

    def compute_sample_covariance_matrix(self, snapshots: np.ndarray) -> np.ndarray:
        n_samples = snapshots.shape[1]
        return (snapshots @ snapshots.conj().T) / n_samples

    def compute_mvdr_weights(self, R_xx: np.ndarray, sat_az: float, sat_el: float) -> np.ndarray:
        a_s = self.compute_steering_vector(sat_az, sat_el)
        r_inv = np.linalg.pinv(R_xx + 1e-4 * np.eye(self.M))
        w = r_inv @ a_s
        return w / (np.dot(a_s.conj(), w) + 1e-12)

    def evaluate_beamformer_spatial_attenuation(
        self,
        weights: np.ndarray,
        jam_az: float,
        jam_el: float,
        sat_az: float,
        sat_el: float
    ) -> Dict:
        a_j = self.compute_steering_vector(jam_az, jam_el)
        a_s = self.compute_steering_vector(sat_az, sat_el)
        
        jam_gain = 20.0 * math.log10(max(abs(np.dot(weights.conj(), a_j)), 1e-6))
        sat_gain = 20.0 * math.log10(max(abs(np.dot(weights.conj(), a_s)), 1e-6))
        
        return {
            "jammer_null_depth_db": round(jam_gain, 2),
            "satellite_gain_db": round(sat_gain, 2),
            "anti_jamming_mitigation_active": jam_gain <= -20.0
        }

    def evaluate_null_depth(
        self,
        jam_az: float = 210.0,
        jam_el: float = 10.0,
        sat_az: float = 45.0,
        sat_el: float = 60.0,
        **kwargs
    ) -> Dict:
        a_s = self.compute_steering_vector(sat_az, sat_el)
        a_j = self.compute_steering_vector(jam_az, jam_el)
        
        # Synthesize interference-plus-noise covariance matrix
        R_xx = np.outer(a_j, a_j.conj()) * 1000.0 + np.eye(self.M)
        w = np.linalg.inv(R_xx) @ a_s
        w = w / np.dot(a_s.conj(), w)
        
        null_gain = 20.0 * math.log10(max(abs(np.dot(w.conj(), a_j)), 1e-6))
        return {
            "jammer_null_depth_db": round(null_gain, 2),
            "anti_jamming_mitigation_active": null_gain <= -20.0
        }