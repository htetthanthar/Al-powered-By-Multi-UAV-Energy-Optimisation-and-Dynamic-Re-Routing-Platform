# models/energy_mlp.py
import torch
import torch.nn as nn

class EnergySurrogateMLP(nn.Module):
    def __init__(self, in_features: int = 5, hidden_dim: int = 64):
        super(EnergySurrogateMLP, self).__init__()
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