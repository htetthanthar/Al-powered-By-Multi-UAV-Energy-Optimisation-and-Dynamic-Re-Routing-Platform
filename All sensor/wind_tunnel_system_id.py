import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "reports/system_id"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def aerodynamic_power_model(
    X: np.ndarray,
    cd0_parasite_area: float,
    induced_power_factor_k: float,
    profile_power_base_w: float
) -> np.ndarray:
    """
    Theoretical Aerodynamic Power Equation:
      P_total = P_induced(k, m, v_air) + P_parasite(cd0, v_air) + P_profile(P0, v_air)
    X is a 2D array: X[:, 0] = v_airspeed_ms, X[:, 1] = total_mass_kg
    """
    v_air = X[:, 0]
    mass = X[:, 1]
    
    rho = 1.225
    g = 9.80665
    disk_area = 4.0 * math.pi * ((15.0 * 0.0254 / 2.0) ** 2) if 'math' in globals() else 0.456

    weight = mass * g

    # 1. Induced Power (Glauert Forward Model)
    # v_induced approx = T / (2 * rho * A * v_air)
    v_air_safe = np.maximum(v_air, 1.0)
    p_induced = induced_power_factor_k * (weight ** 2) / (2.0 * rho * disk_area * v_air_safe)

    # 2. Parasite Power: 0.5 * rho * v^3 * (CD0 * A)
    p_parasite = 0.5 * rho * (v_air ** 3) * cd0_parasite_area

    # 3. Blade Profile Drag Power: P0 * (1 + 3 * (v / v_tip)^2)
    p_profile = profile_power_base_w * (1.0 + 3.0 * ((v_air / 45.0) ** 2))

    return p_induced + p_parasite + p_profile + 25.0  # +25W avionics base

def fit_aerodynamic_parameters(csv_path: str = "data/flight_telemetry.csv") -> dict:
    """
    Executes non-linear least squares parameter estimation over telemetry records.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        from generate_datasets import generate_flight_telemetry
        generate_flight_telemetry(n_samples=2000)
        df = pd.read_csv(csv_path)

    # Calculate airspeed
    v_ground = df["ground_speed_ms"].values
    w_speed = df["wind_speed_ms"].values
    w_angle = df["wind_angle_rad"].values
    payload = df["payload_kg"].values
    total_mass = 4.5 + payload

    v_air_x = v_ground - w_speed * np.cos(w_angle)
    v_air_y = - w_speed * np.sin(w_angle)
    v_air = np.sqrt(v_air_x**2 + v_air_y**2)

    power_actual = df["power_consumption_watts"].values

    X_data = np.column_stack([v_air, total_mass])

    # Initial parameter guesses: [CD0*A, k_induced, P0_profile]
    p0 = [0.035, 1.15, 75.0]
    bounds = ([0.005, 0.8, 40.0], [0.15, 2.0, 150.0])

    popt, pcov = curve_fit(aerodynamic_power_model, X_data, power_actual, p0=p0, bounds=bounds)

    cd0_fitted, k_fitted, p0_fitted = popt
    perr = np.sqrt(np.diag(pcov))

    # Compute R^2 of fitted physical model
    power_pred = aerodynamic_power_model(X_data, *popt)
    r2_score = 1.0 - (np.sum((power_actual - power_pred)**2) / np.sum((power_actual - np.mean(power_actual))**2))

    return {
        "cd0_parasite_area_m2": round(float(cd0_fitted), 5),
        "cd0_std_error": round(float(perr[0]), 5),
        "induced_power_factor_k": round(float(k_fitted), 4),
        "k_std_error": round(float(perr[1]), 4),
        "profile_power_base_watts": round(float(p0_fitted), 2),
        "p0_std_error": round(float(perr[2]), 2),
        "fitted_physical_r2": round(float(r2_score), 4)
    }

if __name__ == "__main__":
    fit_results = fit_aerodynamic_parameters()
    print("==========================================================================")
    print("      NON-LINEAR SYSTEM IDENTIFICATION & AERODYNAMIC PARAMETER FIT        ")
    print("==========================================================================")
    for k, v in fit_results.items():
        print(f"  • {k:28s}: {v}")
    print("==========================================================================\n")