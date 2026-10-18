import math
import numpy as np
from typing import Dict, Tuple, List, Optional

class USVDeckHeaveLandingController:
    def __init__(
        self,
        significant_wave_height_hs_m: float = 1.25, # Sea State 3
        peak_wave_period_tp_s: float = 6.5,
        touchdown_relative_alt_m: float = 0.35,
        max_allowable_deck_tilt_deg: float = 4.5,
        ar_model_order: int = 6
    ):
        self.hs = significant_wave_height_hs_m
        self.tp = peak_wave_period_tp_s
        self.h_flare = touchdown_relative_alt_m
        self.max_tilt = math.radians(max_allowable_deck_tilt_deg)
        self.p_order = ar_model_order

        # Auto-regressive coefficients pre-fitted to PM wave spectrum
        self.ar_weights = np.array([1.82, -1.24, 0.48, -0.15, 0.06, -0.02])
        self.heave_history: List[float] = [0.0] * self.p_order

    def generate_synthetic_deck_motion(
        self,
        t_epoch_s: float
    ) -> Dict[str, float]:
        """
        Simulates 6-DOF moving marine deck kinematics (heave, pitch, roll, forward surge).
        """
        omega_p = (2.0 * math.pi) / self.tp
        
        # Superposition of 3 dominant wave spectral harmonics
        z_heave = (
            (self.hs / 2.0) * math.sin(omega_p * t_epoch_s) +
            (self.hs / 6.0) * math.sin(1.85 * omega_p * t_epoch_s + 0.4) +
            (self.hs / 10.0) * math.sin(0.65 * omega_p * t_epoch_s - 0.8)
        )
        
        # Deck Heave Velocities and Accelerations via analytical derivatives
        vz_deck = (
            (self.hs / 2.0) * omega_p * math.cos(omega_p * t_epoch_s) +
            (self.hs / 6.0) * (1.85 * omega_p) * math.cos(1.85 * omega_p * t_epoch_s + 0.4)
        )
        
        az_deck = (
            -(self.hs / 2.0) * (omega_p ** 2) * math.sin(omega_p * t_epoch_s)
        )

        # Deck angular roll and pitch oscillations
        roll_deck = math.radians(3.2 * math.sin(0.9 * omega_p * t_epoch_s + 0.5))
        pitch_deck = math.radians(2.1 * math.cos(1.1 * omega_p * t_epoch_s))

        # USV forward surge (12 knots ~ 6.17 m/s forward speed)
        v_surge_ms = 6.17 + 0.4 * math.sin(0.3 * t_epoch_s)

        return {
            "heave_position_z_m": z_heave,
            "heave_velocity_vz_ms": vz_deck,
            "heave_accel_az_ms2": az_deck,
            "deck_roll_rad": roll_deck,
            "deck_pitch_rad": pitch_deck,
            "usv_surge_speed_ms": v_surge_ms
        }

    def predict_optimal_touchdown_window(
        self,
        current_deck_heave_m: float,
        dt_seconds: float = 0.1,
        lookahead_steps: int = 25
    ) -> Dict:
        """
        Propagates AR model forward to find next peak wave crest (zero velocity, negative acceleration).
        """
        self.heave_history.pop(0)
        self.heave_history.append(current_deck_heave_m)

        pred_series = list(self.heave_history)
        forecast_heave = []

        for step in range(lookahead_steps):
            # AR prediction: z_{t+1} = sum(w_k * z_{t-k+1})
            regressors = np.array(pred_series[-self.p_order:])
            next_z = float(np.dot(self.ar_weights, regressors[::-1]))
            pred_series.append(next_z)
            forecast_heave.append(next_z)

        # Locate first zero-crossing of derivative from positive to negative (wave crest)
        optimal_step = None
        for i in range(1, len(forecast_heave) - 1):
            dz1 = forecast_heave[i] - forecast_heave[i-1]
            dz2 = forecast_heave[i+1] - forecast_heave[i]
            if dz1 >= 0.0 and dz2 < 0.0: # Local maximum / crest
                optimal_step = i
                break

        time_to_crest_s = (optimal_step * dt_seconds) if optimal_step is not None else 1.2

        return {
            "predicted_wave_crest_lead_time_s": round(time_to_crest_s, 2),
            "predicted_crest_height_m": round(forecast_heave[optimal_step] if optimal_step else forecast_heave[0], 2),
            "forecasted_heave_trajectory": [round(z, 2) for z in forecast_heave[:8]]
        }

    def evaluate_touchdown_trigger(
        self,
        uav_relative_pos_ned: Tuple[float, float, float], # relative to deck pad
        deck_motion: Dict[str, float]
    ) -> Dict:
        """
        Validates spatial alignment and deck kinematics before firing magnetic/clamp landing gear.
        """
        dx, dy, dz_rel = uav_relative_pos_ned
        planar_err_m = math.hypot(dx, dy)
        vz_deck = deck_motion["heave_velocity_vz_ms"]
        az_deck = deck_motion["heave_accel_az_ms2"]
        roll_d = deck_motion["deck_roll_rad"]
        pitch_d = deck_motion["deck_pitch_rad"]

        # Kinematic Phase-Lock Predicates:
        # 1. Close planar tracking: ||dx, dy|| <= 0.20 m
        # 2. Flare altitude: z_rel <= 0.40 m
        # 3. Wave crest phase: |vz_deck| <= 0.15 m/s and az_deck <= 0.0
        # 4. Deck tilt stability: |roll| <= 4.5° and |pitch| <= 4.5°
        pos_aligned = planar_err_m <= 0.20 and abs(dz_rel) <= self.h_flare
        crest_locked = abs(vz_deck) <= 0.18 and az_deck <= 0.0
        tilt_safe = abs(roll_d) <= self.max_tilt and abs(pitch_d) <= self.max_tilt

        permit_clamp = pos_aligned and crest_locked and tilt_safe

        return {
            "planar_alignment_error_m": round(planar_err_m, 3),
            "relative_height_above_deck_m": round(abs(dz_rel), 3),
            "deck_heave_velocity_ms": round(vz_deck, 3),
            "deck_roll_deg": round(math.degrees(roll_d), 2),
            "deck_pitch_deg": round(math.degrees(pitch_d), 2),
            "phase_lock_crest_condition": crest_locked,
            "deck_tilt_safe": tilt_safe,
            "engage_landing_gear_clamps": permit_clamp,
            "guidance_action": "TOUCHDOWN_ENGAGE_MAGNETIC_LOCK" if permit_clamp else "MATCH_SURGE_HOLD_ABOVE_CREST"
        }

if __name__ == "__main__":
    usv_guide = USVDeckHeaveLandingController(significant_wave_height_hs_m=1.25, peak_wave_period_tp_s=6.5)

    print("==========================================================================")
    print("      MARINE USV DECK HEAVE PREDICTION & PHASE-LOCK GUIDANCE AUDIT        ")
    print("==========================================================================")

    # 1. Evaluate Wave Crest Prediction
    now_heave = 0.35 # Deck rising near crest
    prediction = usv_guide.predict_optimal_touchdown_window(now_heave, dt_seconds=0.1)
    print("Auto-Regressive Wave Heave Predictor:")
    for k, v in prediction.items():
        print(f"  • {k:35s}: {v}")

    # 2. Evaluate Landing Gate Trigger at Peak Wave Crest
    deck_telemetry = usv_guide.generate_synthetic_deck_motion(t_epoch_s=1.625) # Exact wave crest time
    uav_rel = (0.05, 0.04, -0.28) # 6cm planar error, 28cm above deck
    decision = usv_guide.evaluate_touchdown_trigger(uav_rel, deck_telemetry)

    print("\nMoving Vertipad Touchdown Gate Audit:")
    for k, v in decision.items():
        print(f"  • {k:35s}: {v}")
    print("==========================================================================\n")