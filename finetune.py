import torch
from datasets import load_dataset
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from micro_model import MiCRoTransformer
import torch.nn as nn

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess(example):
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    tgt = example.get("output", "")
    prompt = instruction + "\n" + inp
    return {
        "input_ids": tokenizer(prompt, padding="max_length", truncation=True, max_length=128, return_tensors="pt").input_ids.squeeze(0),
        "labels": tokenizer(tgt, padding="max_length", truncation=True, max_length=128, return_tensors="pt").input_ids.squeeze(0)
    }

print("⏳ Loading dataset...")
ds = load_dataset("Muennighoff/flan", split="train[:500]")
ds = ds.map(preprocess)

loader = DataLoader(ds, batch_size=2)

model = MiCRoTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

print("🚀 Starting finetuning...")
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} — Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/micro_finetuned.pt")
print("✅ Saved finetuned model!")

