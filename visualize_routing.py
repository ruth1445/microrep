import torch
from transformers import T5Tokenizer
from microlayer import MiCRoLayer

# Load router + experts into a single MiCRoLayer
layer = MiCRoLayer()
layer.eval()

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 📝 Input your sentence here
text = "Lily said she was fine after the test, but her tone suggested otherwise."

tokens = tokenizer(text, return_tensors="pt")
input_ids = tokens["input_ids"]
decoded = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

with torch.no_grad():
    embeds = layer.embed(input_ids) if hasattr(layer, "embed") else torch.randn(1, input_ids.size(1), 512)
    router_logits = layer.router(embeds)
    top_expert = torch.argmax(router_logits, dim=-1).squeeze()  # shape: (seq_len,)

# Map expert IDs to names and colors
expert_names = ["Lang", "Logic", "Social", "World"]
colors = ["\033[94m", "\033[92m", "\033[95m", "\033[93m"]  # blue, green, magenta, yellow
RESET = "\033[0m"

print("\n🧠 Routing Visualization:")
for token, expert_id in zip(decoded, top_expert):
    eid = expert_id.item()
    expert = expert_names[eid]
    color = colors[eid]
    print(f"{color}{token:<12} → {expert}{RESET}")

