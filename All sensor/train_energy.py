# train_energy.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    from models.energy_mlp import EnergySurrogateMLP
except ModuleNotFoundError:
    class EnergySurrogateMLP(nn.Module):
        def __init__(self, in_features: int = 5, hidden_dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, 32),
                nn.Mish(),
                nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def train_energy_surrogate(epochs: int = 20):
    os.makedirs("checkpoints/onnx", exist_ok=True)

    np.random.seed(42)
    x_train = np.random.uniform(
        low=[5.0, 0.0, 0.0, -np.pi, 0.5],
        high=[25.0, 2.0, 12.0, np.pi, 15.0],
        size=(1000, 5)
    )
    y_train = (220.0 + 4.2 * (x_train[:, 0] ** 1.8) + 38.0 * x_train[:, 1] + 12.0 * x_train[:, 2]).reshape(-1, 1)

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    model = EnergySurrogateMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(x_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:02d}/{epochs} - Energy MSE: {loss.item():.4f}")

    checkpoint_path = "checkpoints/best_energy_mlp.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[+] Checkpoint saved: {checkpoint_path}")


# Execute training when imported by main.py or run directly
train_energy_surrogate()