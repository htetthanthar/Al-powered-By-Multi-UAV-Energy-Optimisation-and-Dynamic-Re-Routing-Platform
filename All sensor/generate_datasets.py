import os
import math
import numpy as np
import pandas as pd
import requests

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 1. UAV PHYSICAL TELEMETRY DATASET GENERATOR
# -------------------------------------------------------------------------
def generate_flight_telemetry(n_samples=15000):
    np.random.seed(42)
    
    # Feature distributions
    ground_speed_ms = np.random.uniform(12.0, 22.0, n_samples)  # Around 65 km/h (~18 m/s)
    payload_kg = np.random.uniform(0.0, 2.0, n_samples)         # 0 to 2 kg
    wind_speed_ms = np.random.exponential(3.0, n_samples)       # Mean ~3 m/s wind
    wind_speed_ms = np.clip(wind_speed_ms, 0.0, 15.0)
    wind_angle_rad = np.random.uniform(0.0, 2 * math.pi, n_samples)
    delta_altitude_m = np.random.uniform(-10.0, 10.0, n_samples) # Climb/descent gradients

    # Aerodynamic Constants (Representative Quadcopter: 4.5 kg base mass)
    M_BASE = 4.5
    G = 9.81
    RHO = 1.225         # Air density (kg/m^3)
    ROTOR_RADIUS = 0.25 # Meters
    N_ROTORS = 4
    DISK_AREA = N_ROTORS * math.pi * (ROTOR_RADIUS ** 2)
    CD_FRONT = 0.35     # Parasite drag coefficient
    FRONTAL_AREA = 0.08 # m^2

    power_watts = []
    for i in range(n_samples):
        v_g = ground_speed_ms[i]
        m_p = payload_kg[i]
        w_s = wind_speed_ms[i]
        w_theta = wind_angle_rad[i]
        dh = delta_altitude_m[i]

        # Calculate effective airspeed vector
        v_air_x = v_g - w_s * math.cos(w_theta)
        v_air_y = - w_s * math.sin(w_theta)
        v_air = math.sqrt(v_air_x**2 + v_air_y**2)

        total_mass = M_BASE + m_p
        weight_force = total_mass * G

        # Induced Power (Hover/Lift component)
        p_induced = (weight_force ** 1.5) / math.sqrt(2 * RHO * DISK_AREA)

        # Profile Drag Power (Blade rotation resistance)
        p_profile = 75.0 * (1 + 3 * ((v_air / 45.0) ** 2))

        # Parasite Drag Power (Fuselage resistance against relative wind)
        p_parasite = 0.5 * CD_FRONT * FRONTAL_AREA * RHO * (v_air ** 3)

        # Climb/Descent Gravitational Work
        p_climb = total_mass * G * (dh / 10.0)

        # Base system avionics draw
        p_avionics = 25.0

        p_total = p_induced + p_profile + p_parasite + p_climb + p_avionics + np.random.normal(0, 3.0)
        power_watts.append(max(p_total, 120.0))

    df_telemetry = pd.DataFrame({
        "ground_speed_ms": ground_speed_ms,
        "payload_kg": payload_kg,
        "wind_speed_ms": wind_speed_ms,
        "wind_angle_rad": wind_angle_rad,
        "delta_altitude_m": delta_altitude_m,
        "power_consumption_watts": power_watts
    })

    file_path = os.path.join(DATA_DIR, "flight_telemetry.csv")
    df_telemetry.to_csv(file_path, index=False)
    print(f"Generated {n_samples} flight telemetry samples -> {file_path}")

# -------------------------------------------------------------------------
# 2. METEOROLOGICAL TIME-SERIES GENERATOR
# -------------------------------------------------------------------------
def generate_weather_history(timesteps=8760):
    """Generates 1 full year of hourly meteorological observations."""
    np.random.seed(101)
    
    time_index = pd.date_range(start="2025-01-01", periods=timesteps, freq="h")
    
    # Base seasonal and diurnal temperature curves
    base_temp = 28.0 + 4.0 * np.sin(np.linspace(0, 2 * math.pi * 365, timesteps))
    diurnal_temp = 3.0 * np.sin(np.linspace(0, 2 * math.pi * timesteps / 24, timesteps))
    temperature = base_temp + diurnal_temp + np.random.normal(0, 0.8, timesteps)
    
    # Atmospheric pressure (hPa)
    pressure = 1013.25 - (temperature - 25.0) * 0.4 + np.random.normal(0, 1.2, timesteps)
    
    # Relative Humidity (%)
    humidity = np.clip(75.0 - (diurnal_temp * 5.0) + np.random.normal(0, 5.0, timesteps), 40.0, 100.0)
    
    # Wind components (u: East-West, v: North-South)
    u_wind = np.random.normal(1.5, 3.0, timesteps)
    v_wind = np.random.normal(-0.5, 2.8, timesteps)
    
    # Precipitation: sporadic convective events
    rain_prob = np.where((humidity > 85.0) & (pressure < 1010.0), 0.45, 0.05)
    rain_event = (np.random.rand(timesteps) < rain_prob).astype(int)

    df_weather = pd.DataFrame({
        "timestamp": time_index,
        "temperature_c": np.round(temperature, 2),
        "pressure_hpa": np.round(pressure, 2),
        "humidity_pct": np.round(humidity, 2),
        "wind_u_ms": np.round(u_wind, 2),
        "wind_v_ms": np.round(v_wind, 2),
        "rain_flag": rain_event
    })

    file_path = os.path.join(DATA_DIR, "weather_history.csv")
    df_weather.to_csv(file_path, index=False)
    print(f"Generated {timesteps} hourly weather time-series records -> {file_path}")

if __name__ == "__main__":
    generate_flight_telemetry()
    generate_weather_history()