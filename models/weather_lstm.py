# models/weather_lstm.py
import torch
import torch.nn as nn

class BidirectionalWeatherLSTM(nn.Module):
    def __init__(self, in_features: int = 4, hidden_dim: int = 32, num_layers: int = 2, out_steps: int = 6):
        super(BidirectionalWeatherLSTM, self).__init__()
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