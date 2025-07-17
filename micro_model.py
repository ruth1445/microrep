import torch
import torch.nn as nn
from microlayer import MiCRoLayer

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

class MiCRoTransformer(nn.Module):
    def __init__(self, num_layers=8, hidden_size=512):
        super().__init__()
        self.layers = nn.ModuleList()
        self.embed = nn.Embedding(32128, hidden_size)  # T5 vocab size

        for i in range(num_layers):
            if 1 < i < num_layers - 1:  # MiCRo in the middle
                self.layers.append(MiCRoLayer(hidden_size))
            else:
                self.layers.append(TransformerBlock(hidden_size))

        self.head = nn.Linear(hidden_size, 32128)

    def forward(self, input_ids):
        x = self.embed(input_ids)  # (B, L, D)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

