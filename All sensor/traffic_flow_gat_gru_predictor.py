import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class GATv2CorridorLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4):
        super(GATv2CorridorLayer, self).__init__()
        self.num_heads = num_heads
        self.out_per_head = out_features // num_heads

        self.w_node = nn.Linear(in_features, out_features, bias=False)
        self.a_att = nn.Parameter(torch.FloatTensor(num_heads, 2 * self.out_per_head))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_node.weight)
        nn.init.xavier_uniform_(self.a_att)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: (batch, num_nodes, in_features)
        adj_mask: (num_nodes, num_nodes) - Binary adjacency
        """
        b_size, n_nodes, _ = h.shape
        # Linear projection: (b, n, heads, out_per_head)
        h_proj = self.w_node(h).view(b_size, n_nodes, self.num_heads, self.out_per_head)

        # Pairwise combination for GATv2
        h_i = h_proj.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1) # (b, n, n, heads, out_per_head)
        h_j = h_proj.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)

        # Concat along head dimension: (b, n, n, heads, 2 * out_per_head)
        h_cat = torch.cat([h_i, h_j], dim=-1)
        
        # Attention score: (b, heads, n, n)
        scores = (h_cat * self.a_att).sum(dim=-1).permute(0, 3, 1, 2)
        scores = self.leaky_relu(scores)

        # Mask non-connected nodes with -inf
        mask_expanded = adj_mask.unsqueeze(0).unsqueeze(0).expand(b_size, self.num_heads, -1, -1)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1) # (b, heads, n, n)

        # Aggregate neighbor features: (b, heads, n, out_per_head)
        h_out = torch.matmul(attn_weights, h_proj.permute(0, 2, 1, 3))
        # Concatenate heads: (b, n, out_features)
        h_out = h_out.permute(0, 2, 1, 3).contiguous().view(b_size, n_nodes, -1)

        return h_out, attn_weights

class SpatioTemporalAirspaceGATGRU(nn.Module):
    def __init__(self, num_corridors: int = 8, in_dim: int = 4, hidden_dim: int = 32, forecast_steps: int = 6):
        """
        Features per corridor node: [current_drone_count, mean_speed_ms, wind_hazard_index, queue_delay_s]
        Target forecast: [predicted_traffic_density, collision_probability_index] over next K steps.
        """
        super(SpatioTemporalAirspaceGATGRU, self).__init__()
        self.num_nodes = num_corridors
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps

        self.gat_layer = GATv2CorridorLayer(in_features=in_dim, out_features=hidden_dim, num_heads=4)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.predictor_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2 * forecast_steps) # (density, collision_risk) x K steps
        )

    def forward(self, x_seq: torch.Tensor, adj_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x_seq: (batch_size, seq_len, num_nodes, in_dim)
        Returns forecasted metrics across all corridor nodes.
        """
        b_size, seq_len, n_nodes, _ = x_seq.shape
        h_state = torch.zeros(b_size * n_nodes, self.hidden_dim, device=x_seq.device)

        last_attn = None
        for t in range(seq_len):
            xt = x_seq[:, t, :, :] # (batch, num_nodes, in_dim)
            h_spatial, last_attn = self.gat_layer(xt, adj_mask)
            
            # Reshape for GRU cell update
            h_in = h_spatial.view(b_size * n_nodes, self.hidden_dim)
            h_state = self.gru_cell(h_in, h_state)

        # Regress forecast horizon
        predictions_raw = self.predictor_head(h_state).view(b_size, n_nodes, self.forecast_steps, 2)
        
        # Split density (positive ReLU) and collision risk (Sigmoid probability)
        pred_density = F.relu(predictions_raw[..., 0])
        pred_risk = torch.sigmoid(predictions_raw[..., 1])

        return {
            "forecast_traffic_density": pred_density,
            "forecast_collision_risk_index": pred_risk,
            "spatial_attention_matrix": last_attn
        }

if __name__ == "__main__":
    model = SpatioTemporalAirspaceGATGRU(num_corridors=6, in_dim=4, hidden_dim=32, forecast_steps=6)
    model.eval()

    # 6 Urban Flight Corridors interconnected graph
    adj = torch.tensor([
        [1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 1]
    ], dtype=torch.float32)

    # Input: 12 historical 2.5-min intervals (30 min history) across 6 corridors
    dummy_input = torch.randn(1, 12, 6, 4)
    dummy_input[..., 0] = torch.abs(dummy_input[..., 0]) * 4.0 # 0 to 8 drones

    with torch.no_grad():
        out = model(dummy_input, adj)

    print("==========================================================================")
    print("      SPATIO-TEMPORAL GAT-GRU URBAN CORRIDOR CONGESTION PREDICTOR AUDIT   ")
    print("==========================================================================")
    print(f"Forecast Horizon Steps      : 6 Steps (T+5min to T+30min)")
    print(f"Airspace Corridor Nodes     : 6 Interconnected Flight Lanes\n")
    
    densities = out["forecast_traffic_density"][0].numpy()
    risks = out["forecast_collision_risk_index"][0].numpy()

    for c_id in range(6):
        print(f"Corridor #{c_id+1} Forecast -> "
              f"Density (T+15m): {densities[c_id, 2]:4.1f} drones/km | "
              f"Density (T+30m): {densities[c_id, 5]:4.1f} drones/km | "
              f"Peak Collision Risk: {np.max(risks[c_id])*100:4.1f}%")
    print("==========================================================================\n")