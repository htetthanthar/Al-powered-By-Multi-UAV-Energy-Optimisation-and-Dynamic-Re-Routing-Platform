# train_weather.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    from models.weather_lstm import BidirectionalWeatherLSTM
except ModuleNotFoundError:
    class BidirectionalWeatherLSTM(nn.Module):
        def __init__(self, in_features: int = 4, hidden_dim: int = 32, num_layers: int = 2, out_steps: int = 6):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=in_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
            self.fc = nn.Linear(hidden_dim * 2, out_steps * 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])


def train_weather_forecaster(epochs: int = 15):
    os.makedirs("checkpoints/onnx", exist_ok=True)

    np.random.seed(42)
    batch_size = 128
    seq_len = 12
    in_features = 4
    out_steps = 6

    x_train = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
    y_train = np.random.randn(batch_size, out_steps * 2).astype(np.float32)

    x_tensor = torch.tensor(x_train)
    y_tensor = torch.tensor(y_train)

    model = BidirectionalWeatherLSTM(in_features=in_features, hidden_dim=32, num_layers=2, out_steps=out_steps)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(x_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {loss.item():.4f}")

    checkpoint_path = "checkpoints/best_weather_lstm.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[+] Weather LSTM checkpoint successfully saved to {checkpoint_path}")


if __name__ == "__main__":
    train_weather_forecaster()