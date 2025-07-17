import torch.nn as nn

class Router(nn.Module):
    def __init__(self, hidden_dim=512, num_experts=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        return self.mlp(x)

