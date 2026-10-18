import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, num_nodes, in_features)
        adj_norm: (num_nodes, num_nodes) Normalized Laplacian Matrix: D^(-1/2) * (A + I) * D^(-1/2)
        """
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_norm, support) + self.bias
        return output

class SpatioTemporalWindGNN(nn.Module):
    def __init__(self, num_nodes: int = 16, in_dim: int = 4, hidden_dim: int = 64, out_dim: int = 2, seq_len: int = 12):
        """
        Features per node: [wind_u, wind_v, temperature, pressure]
        Target prediction: [future_wind_u, future_wind_v] at t+1..t+K across all nodes.
        """
        super(SpatioTemporalWindGNN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Spatial Feature Extraction
        self.gcn1 = GraphConvolutionLayer(in_dim, hidden_dim)
        self.gcn2 = GraphConvolutionLayer(hidden_dim, hidden_dim)

        # Temporal Sequence Modeling
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Output Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x_seq: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (batch_size, seq_len, num_nodes, in_dim)
        Returns: (batch_size, num_nodes, out_dim)
        """
        batch_size, seq_len, num_nodes, in_dim = x_seq.shape
        spatial_embeddings = []

        for t in range(seq_len):
            xt = x_seq[:, t, :, :]  # (batch_size, num_nodes, in_dim)
            h_spatial = F.relu(self.gcn1(xt, adj_norm))
            h_spatial = F.dropout(h_spatial, p=0.1, training=self.training)
            h_spatial = self.gcn2(h_spatial, adj_norm)  # (batch_size, num_nodes, hidden_dim)
            spatial_embeddings.append(h_spatial.unsqueeze(1))

        # Shape: (batch_size, seq_len, num_nodes, hidden_dim)
        spatial_seq = torch.cat(spatial_embeddings, dim=1)

        # Reshape to treat each node's sequence independently through temporal GRU
        # (batch_size * num_nodes, seq_len, hidden_dim)
        gru_in = spatial_seq.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, seq_len, self.hidden_dim)
        _, h_n = self.temporal_gru(gru_in)  # h_n[-1]: (batch_size * num_nodes, hidden_dim)

        # Map back to node space and regress
        h_last = h_n[-1].view(batch_size, num_nodes, self.hidden_dim)
        out_uv = self.regressor(h_last)  # (batch_size, num_nodes, out_dim)
        return out_uv

def compute_normalized_laplacian(adj: np.ndarray) -> torch.Tensor:
    """Computes renormalized symmetric graph Laplacian: D^(-1/2) * (A + I) * D^(-1/2)."""
    adj_loop = adj + np.eye(adj.shape[0])
    row_sum = np.array(adj_loop.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt.dot(adj_loop).dot(d_mat_inv_sqrt)
    return torch.tensor(adj_normalized, dtype=torch.float32)

if __name__ == "__main__":
    N_NODES = 8
    SEQ_LEN = 12
    BATCH_SIZE = 16

    # Random distance-based adjacency matrix
    coords = np.random.uniform(0, 10, (N_NODES, 2))
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    sigma = 3.0
    adj_raw = np.exp(- (dist_matrix ** 2) / (sigma ** 2))
    adj_raw[adj_raw < 0.2] = 0.0
    np.fill_diagonal(adj_raw, 0.0)

    adj_norm = compute_normalized_laplacian(adj_raw)

    model = SpatioTemporalWindGNN(num_nodes=N_NODES, in_dim=4, hidden_dim=32, out_dim=2, seq_len=SEQ_LEN)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, N_NODES, 4)

    output = model(dummy_input, adj_norm)
    print(f"ST-GNN Input Shape  : {dummy_input.shape}")
    print(f"ST-GNN Output Shape : {output.shape} (Batch, Nodes, [u, v])")