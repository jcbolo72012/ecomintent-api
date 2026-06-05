#!/usr/bin/env python3
"""
Processes manually-downloaded Bitext CSV files into normalized parquet format.
Place the CSVs in data/raw/ before running:
  data/raw/bitext_retail.csv
  data/raw/bitext_support.csv
"""
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()
DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "csv_name": "bitext_retail.csv",
        "name": "bitext_retail",
        "text_col": "instruction",
        "label_col": "intent",
    },
    {
        "csv_name": "bitext_support.csv",
        "name": "bitext_support",
        "text_col": "instruction",
        "label_col": "intent",
    },
]

stats = {}
for ds_config in DATASETS:
    csv_path = DATA_RAW / ds_config["csv_name"]
    if not csv_path.exists():
        console.print(f"[yellow]SKIP: {csv_path} not found — download it manually from HuggingFace[/yellow]")
        continue

    console.print(f"\n[cyan]Loading {csv_path.name}...[/cyan]")
    df = pd.read_csv(csv_path)
    console.print(f"  Columns: {list(df.columns)}")

    # Detect text/label columns flexibly
    text_col = ds_config["text_col"] if ds_config["text_col"] in df.columns else df.columns[0]
    label_col = ds_config["label_col"] if ds_config["label_col"] in df.columns else "intent"
    if label_col not in df.columns:
        # fallback: look for any column with 'intent' in name
        candidates = [c for c in df.columns if "intent" in c.lower()]
        label_col = candidates[0] if candidates else df.columns[-1]

    console.print(f"  Using text_col={text_col!r}, label_col={label_col!r}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna()
    df["source"] = ds_config["name"]

    out_path = DATA_RAW / f"{ds_config['name']}.parquet"
    df.to_parquet(out_path, index=False)

    label_dist = df["label"].value_counts().to_dict()
    stats[ds_config["name"]] = {
        "total_rows": len(df),
        "num_labels": len(label_dist),
        "label_distribution": label_dist,
    }
    console.print(f"  [green]OK[/green] {len(df):,} rows, {len(label_dist)} labels -> {out_path}")

with open(DATA_RAW / "download_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

console.print(f"\n[green]Done. Stats saved to data/raw/download_stats.json[/green]")
