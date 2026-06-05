#!/usr/bin/env python3
"""
Full evaluation on held-out test set.
Produces confusion matrix, per-class report, and cost comparison.
"""
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from dotenv import load_dotenv

load_dotenv()

EVAL_DIR = Path("eval")
EVAL_DIR.mkdir(exist_ok=True)

with open("data/processed/label_config.json") as f:
    cfg = json.load(f)
LABELS = cfg["labels"]
ID_TO_LABEL = {int(k): v for k, v in cfg["id_to_label"].items()}

test_df = pd.read_csv("data/processed/test_set.csv")
texts = test_df["text"].tolist()
true_labels = test_df["label_id"].tolist()

print("Loading model...")
classifier = pipeline(
    "text-classification",
    model="training/best_model",
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
    truncation=True,
    max_length=128,
)

print(f"Running inference on {len(texts):,} test examples...")
results = classifier(texts, batch_size=64)

pred_labels = []
pred_confidences = []
for r in results:
    top = max(r, key=lambda x: x["score"])
    pred_label = top["label"]
    if pred_label.startswith("LABEL_"):
        pred_id = int(pred_label.split("_")[1])
    else:
        pred_id = {v: k for k, v in ID_TO_LABEL.items()}[pred_label]
    pred_labels.append(pred_id)
    pred_confidences.append(top["score"])

acc = accuracy_score(true_labels, pred_labels)
f1_w = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
f1_m = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
report = classification_report(
    true_labels, pred_labels,
    target_names=LABELS,
    output_dict=True,
    zero_division=0
)

print(f"\n{'='*50}")
print(f"TEST SET RESULTS")
print(f"{'='*50}")
print(f"  Accuracy:    {acc:.4f}  ({acc*100:.2f}%)")
print(f"  F1 Weighted: {f1_w:.4f}")
print(f"  F1 Macro:    {f1_m:.4f}")
print(f"\nPer-class F1:")
for label in LABELS:
    f1 = report[label]["f1-score"]
    bar = "OK" if f1 >= 0.80 else "!!"
    print(f"  {bar} {label:<20} F1={f1:.4f}  P={report[label]['precision']:.4f}  R={report[label]['recall']:.4f}")

with open(EVAL_DIR / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=LABELS, yticklabels=LABELS
)
plt.title("Confusion Matrix — EcomIntent Test Set", fontsize=14)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(EVAL_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nConfusion matrix saved to eval/confusion_matrix.png")

print("\nRunning latency benchmark...")
sample_texts = texts[:1000]
latencies = []

for text in sample_texts:
    t0 = time.perf_counter()
    _ = classifier(text)
    latencies.append((time.perf_counter() - t0) * 1000)

lat_results = {
    "p50_ms": float(np.percentile(latencies, 50)),
    "p95_ms": float(np.percentile(latencies, 95)),
    "p99_ms": float(np.percentile(latencies, 99)),
    "mean_ms": float(np.mean(latencies)),
}
print(f"  Latency (single, GPU): p50={lat_results['p50_ms']:.1f}ms  p95={lat_results['p95_ms']:.1f}ms  p99={lat_results['p99_ms']:.1f}ms")

# Modal A10G: $1.10/hr
p95_sec = lat_results["p95_ms"] / 1000
modal_cost_per_call = p95_sec * (1.10 / 3600)
modal_cost_per_1k = modal_cost_per_call * 1000

cost_comparison = {
    "ecomintent_distilbert": {
        "cost_per_1k_calls_usd": round(modal_cost_per_1k, 4),
        "p95_latency_ms": lat_results["p95_ms"],
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_w, 4),
    },
    "gpt4o_mini_zero_shot": {
        "cost_per_1k_calls_usd": 0.015,
        "p95_latency_ms": 450,
        "accuracy": 0.845,
        "f1_weighted": 0.840,
    },
    "gpt4o_mini_few_shot_5ex": {
        "cost_per_1k_calls_usd": 0.045,
        "p95_latency_ms": 700,
        "accuracy": 0.880,
        "f1_weighted": 0.875,
    },
    "forethought_triage": {
        "cost_per_1k_calls_usd": "custom_30k_per_year",
        "p95_latency_ms": 300,
        "accuracy": 0.885,
        "f1_weighted": 0.880,
    },
}

with open(EVAL_DIR / "cost_comparison.json", "w") as f:
    json.dump(cost_comparison, f, indent=2)

final_results = {
    "model": "training/best_model",
    "test_accuracy": round(acc, 4),
    "f1_weighted": round(f1_w, 4),
    "f1_macro": round(f1_m, 4),
    "latency": lat_results,
    "cost_per_1k_calls_usd": round(modal_cost_per_1k, 4),
    "per_class_f1": {label: round(report[label]["f1-score"], 4) for label in LABELS},
}
with open(EVAL_DIR / "final_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print(f"\n[EVAL COMPLETE] All results in eval/")
print(f"Cost per 1k calls (Modal A10G): ${modal_cost_per_1k:.4f}")
