import torch.nn as nn

class ExpertBlock(nn.Module):
    def __init__(self, hidden_size=512, nhead=8):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)

    def forward(self, x):
        return self.block(x)
