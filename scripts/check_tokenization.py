#!/usr/bin/env python3
"""Check token length distribution to set max_length for training."""
import json
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from rich.console import Console

console = Console()
MODEL_NAME = "distilbert-base-uncased"

dataset = load_from_disk("data/processed/final_dataset")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

texts = dataset["train"]["text"][:5000]
lengths = [len(tokenizer(t)["input_ids"]) for t in texts]

p50 = int(np.percentile(lengths, 50))
p95 = int(np.percentile(lengths, 95))
p99 = int(np.percentile(lengths, 99))
max_len = int(np.max(lengths))

console.print(f"\nToken length analysis (n={len(texts)} samples):")
console.print(f"  p50:  {p50}")
console.print(f"  p95:  {p95}")
console.print(f"  p99:  {p99}")
console.print(f"  max:  {max_len}")

recommended = 128 if p95 <= 128 else 256
console.print(f"\n[green]Recommended max_length: {recommended}[/green]")

with open("data/processed/tokenization_config.json", "w") as f:
    json.dump({"recommended_max_length": recommended, "p95": p95, "p99": p99}, f)
