#!/usr/bin/env python3
"""
Merges all data sources, applies label mapping, deduplicates,
and creates train/val/test splits.
"""
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()
random_state = 42
np.random.seed(random_state)

DATA_RAW = Path("data/raw")
DATA_SYNTH = Path("data/synthetic")
DATA_PROC = Path("data/processed")
DATA_PROC.mkdir(exist_ok=True)

with open(DATA_PROC / "label_config.json") as f:
    label_config = json.load(f)

LABEL_MAP = label_config["source_to_canonical"]
LABEL_TO_ID = label_config["label_to_id"]


def map_label(src_label: str) -> str:
    return LABEL_MAP.get(src_label.lower().strip(), "OTHER")


def normalize_text(text: str) -> str:
    return str(text).strip()


dfs = []

# Prefer parquet over CSV for the same stem (avoid double-loading)
_seen_stems = set()
bitext_files = []
for fpath in sorted(DATA_RAW.glob("bitext_*.parquet")) + sorted(DATA_RAW.glob("bitext_*.csv")):
    if fpath.stem not in _seen_stems:
        _seen_stems.add(fpath.stem)
        bitext_files.append(fpath)

if bitext_files:
    for fpath in bitext_files:
        df = pd.read_parquet(fpath) if fpath.suffix == ".parquet" else pd.read_csv(fpath)
        # Normalize column names
        if "instruction" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"instruction": "text"})
        if "intent" in df.columns and "label" not in df.columns:
            df = df.rename(columns={"intent": "label"})
        if "source" not in df.columns:
            df["source"] = fpath.stem
        df["canonical_label"] = df["label"].apply(map_label)
        df["text"] = df["text"].apply(normalize_text)
        dfs.append(df[["text", "canonical_label", "source"]])
        console.print(f"  Loaded {fpath.name}: {len(df):,} rows")
else:
    console.print("[yellow]No Bitext raw files found — synthetic-only mode.[/yellow]")

synth_all = DATA_SYNTH / "synthetic_all.parquet"
if synth_all.exists():
    df_synth = pd.read_parquet(synth_all)
    df_synth["canonical_label"] = df_synth["label"]
    dfs.append(df_synth[["text", "canonical_label", "source"]])
    console.print(f"  Loaded synthetic: {len(df_synth):,} rows")

if not dfs:
    console.print("[red]ERROR: No data sources found. Run generate_synthetic.py first.[/red]")
    raise SystemExit(1)

df_all = pd.concat(dfs, ignore_index=True)
console.print(f"\nBefore dedup: {len(df_all):,} total rows")

df_all["text_hash"] = df_all["text"].str.lower().str.strip().apply(
    lambda x: hashlib.md5(x.encode()).hexdigest()
)
df_all = df_all.drop_duplicates(subset="text_hash").drop(columns="text_hash")
console.print(f"After dedup:  {len(df_all):,} total rows")

df_all["word_count"] = df_all["text"].str.split().str.len()
df_all = df_all[(df_all["word_count"] >= 3) & (df_all["word_count"] <= 200)]
console.print(f"After length filter: {len(df_all):,} total rows")

df_all["label"] = df_all["canonical_label"]
df_all["label_id"] = df_all["label"].map(LABEL_TO_ID)

bad = df_all[df_all["label_id"].isna()]
if len(bad) > 0:
    console.print(f"[yellow]WARNING: {len(bad)} rows with unmapped labels → dropping[/yellow]")
    df_all = df_all.dropna(subset=["label_id"])

df_all["label_id"] = df_all["label_id"].astype(int)

# Cap dominant classes to keep imbalance ratio ≤ 4:1
# Floor is the smallest class count; cap everything at 4x that
min_count = df_all["label"].value_counts().min()
cap = min_count * 4
oversize = df_all["label"].value_counts()[df_all["label"].value_counts() > cap]
if len(oversize) > 0:
    console.print(f"\n[yellow]Capping {len(oversize)} over-represented classes at {cap:,} (4x min={min_count:,}):[/yellow]")
    for lbl in oversize.index:
        console.print(f"  {lbl}: {oversize[lbl]:,} -> {cap:,}")
    df_all = (
        df_all.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), cap), random_state=random_state), include_groups=False)
        .reset_index(drop=True)
    )

console.print("\n[cyan]Class distribution:[/cyan]")
dist = df_all["label"].value_counts()
for label, count in dist.items():
    pct = count / len(df_all) * 100
    bar = "#" * int(pct / 2)
    console.print(f"  {label:<20} {count:>5,}  {pct:5.1f}%  {bar}")

dist.to_json(DATA_PROC / "class_distribution.json")

max_count = dist.max()
min_count = dist.min()
ratio = max_count / min_count
if ratio > 4.0:
    console.print(f"[yellow]NEEDS_REVIEW: Class imbalance ratio {ratio:.1f} > 4.0[/yellow]")

X = df_all["text"].values
y = df_all["label_id"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=random_state
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
)


def make_hf_dataset(texts, labels):
    return Dataset.from_dict({"text": list(texts), "label": list(labels)})


dataset = DatasetDict({
    "train": make_hf_dataset(X_train, y_train),
    "validation": make_hf_dataset(X_val, y_val),
    "test": make_hf_dataset(X_test, y_test),
})

dataset.save_to_disk(str(DATA_PROC / "final_dataset"))
console.print(f"\n[green]Dataset saved:[/green]")
console.print(f"  Train:      {len(dataset['train']):,}")
console.print(f"  Validation: {len(dataset['validation']):,}")
console.print(f"  Test:       {len(dataset['test']):,}")

test_df = pd.DataFrame({"text": X_test, "label_id": y_test})
test_df["label"] = test_df["label_id"].map({v: k for k, v in LABEL_TO_ID.items()})
test_df.to_csv(DATA_PROC / "test_set.csv", index=False)
console.print(f"  Test CSV exported to data/processed/test_set.csv")
