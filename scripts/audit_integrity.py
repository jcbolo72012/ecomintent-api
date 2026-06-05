#!/usr/bin/env python3
"""
Data integrity audit — checks for leakage, template collapse,
near-duplicates, and anything else that could inflate eval numbers.
"""
import json
import hashlib
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("SECTION 1 — EXACT OVERLAP BETWEEN SPLITS")
print("=" * 60)

ds = load_from_disk("data/processed/final_dataset")
train_texts = set(ds["train"]["text"])
val_texts   = set(ds["validation"]["text"])
test_texts  = set(ds["test"]["text"])

print(f"  Train size: {len(ds['train']):,}")
print(f"  Val size:   {len(ds['validation']):,}")
print(f"  Test size:  {len(ds['test']):,}")
print()
tv = len(train_texts & val_texts)
tt = len(train_texts & test_texts)
vt = len(val_texts & test_texts)
print(f"  Train ∩ Val:  {tv}  {'CLEAN' if tv == 0 else 'LEAKAGE DETECTED'}")
print(f"  Train ∩ Test: {tt}  {'CLEAN' if tt == 0 else 'LEAKAGE DETECTED'}")
print(f"  Val   ∩ Test: {vt}  {'CLEAN' if vt == 0 else 'LEAKAGE DETECTED'}")


print()
print("=" * 60)
print("SECTION 2 — NEAR-DUPLICATE DETECTION (lowercased, stripped)")
print("=" * 60)

def norm(t):
    return " ".join(t.lower().strip().split())

train_norm = set(norm(t) for t in ds["train"]["text"])
test_norm  = set(norm(t) for t in ds["test"]["text"])
near = train_norm & test_norm
print(f"  Near-exact (case/whitespace fold) train ∩ test: {len(near)}")
if near:
    for ex in list(near)[:3]:
        print(f"    -> {ex[:80]}")


print()
print("=" * 60)
print("SECTION 3 — LABEL DISTRIBUTION CONSISTENCY")
print("=" * 60)

with open("data/processed/label_config.json") as f:
    cfg = json.load(f)
id2label = {int(k): v for k, v in cfg["id_to_label"].items()}

for split_name in ["train", "validation", "test"]:
    labels = ds[split_name]["label"]
    counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)
    print(f"\n  {split_name.upper()} ({total:,}):")
    for lid, cnt in counts.items():
        print(f"    {id2label[lid]:<22} {cnt:>5,}  ({cnt/total*100:.1f}%)")


print()
print("=" * 60)
print("SECTION 4 — BITEXT TEMPLATE ANALYSIS (source data diversity)")
print("=" * 60)

raw_retail  = pd.read_parquet("data/raw/bitext_retail.parquet")
raw_support = pd.read_parquet("data/raw/bitext_support.parquet")
raw_retail.columns  = [c.lower() for c in raw_retail.columns]
raw_support.columns = [c.lower() for c in raw_support.columns]

for name, df in [("retail", raw_retail), ("support", raw_support)]:
    text_col = "instruction" if "instruction" in df.columns else df.columns[0]
    texts = df[text_col].dropna()
    # Check unique ratio
    unique_ratio = texts.nunique() / len(texts)
    # Word count distribution
    wc = texts.str.split().str.len()
    print(f"\n  {name.upper()} dataset ({len(texts):,} rows):")
    print(f"    Unique text ratio: {unique_ratio:.4f}  ({texts.nunique():,} unique / {len(texts):,} total)")
    print(f"    Word count: mean={wc.mean():.1f}  p50={wc.median():.0f}  p95={wc.quantile(0.95):.0f}  max={wc.max()}")
    # Sample a few examples per intent to check template diversity
    intent_col = "intent" if "intent" in df.columns else df.columns[-1]
    intents = df[intent_col].value_counts().head(3).index.tolist()
    print(f"    Sample from top intent '{intents[0]}':")
    samples = df[df[intent_col] == intents[0]][text_col].sample(min(4, len(df[df[intent_col] == intents[0]])), random_state=42).tolist()
    for s in samples:
        print(f"      -> {s[:80]}")


print()
print("=" * 60)
print("SECTION 5 — TEST SET SAMPLE INSPECTION")
print("=" * 60)

test_df = pd.read_csv("data/processed/test_set.csv")
with open("data/processed/label_config.json") as f:
    cfg = json.load(f)
id2label = {int(k): v for k, v in cfg["id_to_label"].items()}
test_df["label_name"] = test_df["label_id"].map(id2label)

for label_name in cfg["labels"]:
    subset = test_df[test_df["label_name"] == label_name]
    samples = subset["text"].sample(min(3, len(subset)), random_state=42).tolist()
    print(f"\n  {label_name} ({len(subset)} test examples):")
    for s in samples:
        print(f"    -> {s[:90]}")


print()
print("=" * 60)
print("SECTION 6 — EVAL PIPELINE INTEGRITY CHECK")
print("=" * 60)

# Check that eval loaded the correct split (test, not train)
test_csv = pd.read_csv("data/processed/test_set.csv")
train_set = set(ds["train"]["text"])
test_texts_in_train = test_csv["text"].isin(train_set).sum()
print(f"  Test CSV rows that appear in training set: {test_texts_in_train}")
print(f"  Test CSV total rows: {len(test_csv)}")
print(f"  Verdict: {'CLEAN — no leakage' if test_texts_in_train == 0 else 'LEAKAGE DETECTED'}")

print()
print("AUDIT COMPLETE.")
