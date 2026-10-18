import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Load synthetic telemetry data
data_path = "data/flight_telemetry.csv"
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    from generate_datasets import generate_flight_telemetry
    generate_flight_telemetry(n_samples=5000)
    df = pd.read_csv(data_path)

X = torch.tensor(df[["ground_speed_ms", "payload_kg", "wind_speed_ms", "wind_angle_rad", "delta_altitude_m"]].values, dtype=torch.float32)
Y = torch.tensor(df[["power_consumption_watts"]].values, dtype=torch.float32)

# Train/Val Split (80/20)
train_size = int(0.8 * len(X))
train_loader = DataLoader(TensorDataset(X[:train_size], Y[:train_size]), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X[train_size:], Y[train_size:]), batch_size=128, shuffle=False)

def objective(trial):
    # Search Space
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.3)

    # Dynamic Network Construction
    layers = []
    in_dim = 5
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, 1))

    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Fast evaluation loop (5 epochs)
    for epoch in range(5):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

    # Validation MSE
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in val_loader:
            pred = model(bx)
            val_loss += criterion(pred, by).item()

    return val_loss / len(val_loader)

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("[*] Launching Bayesian Hyperparameter Optimization (Optuna)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("========================================================")
    print("             HYPERPARAMETER TUNING RESULTS              ")
    print("========================================================")
    print(f"Best Validation MSE Loss : {study.best_value:.4f}")
    print("Optimal Hyperparameters  :")
    for k, v in study.best_params.items():
        print(f"  • {k:15s}: {v}")
    print("========================================================")