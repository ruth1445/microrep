import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import argparse
import os

# Load expert module based on domain
def get_expert(domain):
    if domain == "language":
        from expert_blocks.language_expert import ExpertBlock
    elif domain == "logic":
        from expert_blocks.logic_expert import ExpertBlock
    elif domain == "social":
        from expert_blocks.social_expert import ExpertBlock
    elif domain == "world":
        from expert_blocks.world_expert import ExpertBlock
    else:
        raise ValueError("Invalid domain name.")
    return ExpertBlock()

class ExpertDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.data = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        x = self.tokenizer(ex["input"], padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze(0)
        y = self.tokenizer(ex["target"], padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze(0)
        return x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    dataset = ExpertDataset(f"data/{args.domain}.jsonl", tokenizer)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = get_expert(args.domain)
    linear_head = nn.Linear(512, tokenizer.vocab_size)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(linear_head.parameters()), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        for input_ids, target_ids in loader:
            input_ids = input_ids.float()  # Fake embeddings for now
            hidden = model(input_ids)
            logits = linear_head(hidden)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), f"models/{args.domain}.pt")
