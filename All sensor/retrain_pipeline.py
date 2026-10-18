import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from dynamic_planner import EnergySurrogateMLP

CHECKPOINT_PATH = "checkpoints/best_energy_mlp.pth"
DRIFT_LOG_PATH = "data/drift_monitor.log"

def detect_concept_drift(baseline_errors, live_errors, alpha=0.05):
    """
    Performs a two-sample Kolmogorov-Smirnov test to detect distribution drift
    between baseline training residuals and live flight prediction errors.
    """
    stat, p_value = ks_2samp(baseline_errors, live_errors)
    is_drift = p_value < alpha
    return is_drift, stat, p_value

def fine_tune_energy_surrogate(model, new_telemetry_df, epochs=5, lr=1e-4):
    """
    Executes continual fine-tuning on newly collected telemetry data.
    """
    print("[*] Initiating Continuous Fine-Tuning Pipeline...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    features = torch.tensor(new_telemetry_df[[
        "ground_speed_ms", "payload_kg", "wind_speed_ms", "wind_angle_rad", "delta_altitude_m"
    ]].values, dtype=torch.float32)

    targets = torch.tensor(new_telemetry_df[["power_consumption_watts"]].values, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(features, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"    Fine-Tuning Epoch {epoch+1}/{epochs} - Loss (MSE): {epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"[+] Fine-tuned model successfully saved -> {CHECKPOINT_PATH}")

def monitor_telemetry_stream():
    np.random.seed(99)
    # Simulated baseline residuals from original validation set
    baseline_residuals = np.random.normal(0, 6.5, 500)

    # Simulated live streaming telemetry with motor degradation (drift)
    live_telemetry_df = pd.read_csv("data/flight_telemetry.csv").sample(300).copy()
    # Introduce structural physical drift: increased drag / motor resistance
    live_telemetry_df["power_consumption_watts"] += np.random.normal(25.0, 8.0, len(live_telemetry_df))

    # Evaluate current model predictions
    model = EnergySurrogateMLP()
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()

    inputs = torch.tensor(live_telemetry_df[[
        "ground_speed_ms", "payload_kg", "wind_speed_ms", "wind_angle_rad", "delta_altitude_m"
    ]].values, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(inputs).numpy().flatten()

    live_residuals = live_telemetry_df["power_consumption_watts"].values - predictions

    drift_detected, ks_stat, p_val = detect_concept_drift(baseline_residuals, live_residuals)

    print("========================================================")
    print("           MLOPS MODEL MONITORING & DRIFT AUDIT         ")
    print("========================================================")
    print(f"Kolmogorov-Smirnov Statistic : {ks_stat:.4f}")
    print(f"P-Value                      : {p_val:.4e}")
    print(f"Concept Drift Status         : {'DRIFT DETECTED' if drift_detected else 'NOMINAL'}")
    print("========================================================")

    if drift_detected:
        print("[!] Warning: Statistical model drift exceeds safety threshold.")
        fine_tune_energy_surrogate(model, live_telemetry_df)
    else:
        print("[+] Model calibration is within valid operational bounds.")

if __name__ == "__main__":
    monitor_telemetry_stream()