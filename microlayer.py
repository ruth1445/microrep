import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch.nn as nn
from experts.language_expert import ExpertBlock as LanguageExpert
from experts.logic_expert import ExpertBlock as LogicExpert
from experts.social_expert import ExpertBlock as SocialExpert
from experts.world_expert import ExpertBlock as WorldExpert
import importlib.util
import sys
import os

router_path = os.path.join(os.path.dirname(__file__), "router.py")
spec = importlib.util.spec_from_file_location("router", router_path)
router_module = importlib.util.module_from_spec(spec)
sys.modules["router"] = router_module
spec.loader.exec_module(router_module)

Router = router_module.Router



class MiCRoLayer(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.router = Router(hidden_dim=hidden_size, num_experts=4)
        
        self.experts = nn.ModuleList([
            LanguageExpert(hidden_size),
            LogicExpert(hidden_size),
            SocialExpert(hidden_size),
            WorldExpert(hidden_size)
        ])

        # Load pretrained weights
        expert_paths = ["models/language.pt", "models/logic.pt", "models/social.pt", "models/world.pt"]
        for i, path in enumerate(expert_paths):
            self.experts[i].load_state_dict(torch.load(path))
            for param in self.experts[i].parameters():
                param.requires_grad = False  # freeze

        self.router.load_state_dict(torch.load("models/router.pt"))
        for param in self.router.parameters():
            param.requires_grad = False  # freeze

    def forward(self, x):
        """
        x: tensor of shape (batch, seq_len, hidden)
        """
        # Get router logits
        logits = self.router(x)  # (B, L, 4)
        top_expert = torch.argmax(logits, dim=-1)  # (B, L)

        # Route each token to its selected expert
        B, L, D = x.shape
        out = torch.zeros_like(x)

        for i in range(B):
            for j in range(L):
                expert_id = top_expert[i, j].item()
                token = x[i, j].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                out[i, j] = self.experts[expert_id](token).squeeze(0).squeeze(0)

        return out

