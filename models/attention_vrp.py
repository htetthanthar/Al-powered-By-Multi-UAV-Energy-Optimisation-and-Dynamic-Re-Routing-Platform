# models/attention_vrp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionVRP(nn.Module):
    def __init__(self, node_dim: int = 4, embed_dim: int = 64, n_heads: int = 4):
        super(GraphAttentionVRP, self).__init__()
        self.node_embed = nn.Linear(node_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.decoder_q = nn.Linear(embed_dim, embed_dim)
        self.pointer = nn.Linear(embed_dim, 1)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        h = self.node_embed(node_features)
        attn_out, _ = self.mha(h, h, h)
        h_agg = h + attn_out
        logits = self.pointer(torch.tanh(h_agg)).squeeze(-1)
        return F.softmax(logits, dim=-1)