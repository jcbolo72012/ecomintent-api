#!/usr/bin/env python3
"""
Fine-tune DistilBERT for 9-class e-commerce intent classification.
Falls back to RoBERTa if val F1 < 0.88 after primary run.
"""
import json
import os
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

with open("data/processed/label_config.json") as f:
    label_config = json.load(f)

with open("data/processed/tokenization_config.json") as f:
    tok_config = json.load(f)

LABELS = label_config["labels"]
LABEL_TO_ID = label_config["label_to_id"]
ID_TO_LABEL = {int(k): v for k, v in label_config["id_to_label"].items()}
NUM_LABELS = len(LABELS)
MAX_LENGTH = tok_config["recommended_max_length"]

MODEL_NAME = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
OUTPUT_DIR = Path(f"training/checkpoints/{MODEL_NAME.replace('/', '_')}_{datetime.now():%Y%m%d_%H%M}")
BEST_MODEL_DIR = Path("training/best_model")
EVAL_DIR = Path("eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

print(f"Training config:")
print(f"  Model:       {MODEL_NAME}")
print(f"  Num labels:  {NUM_LABELS}")
print(f"  Max length:  {MAX_LENGTH}")
print(f"  GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

dataset = load_from_disk("data/processed/final_dataset")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)


tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.rename_column("label", "labels")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    result = {"accuracy": acc, "f1_weighted": f1_weighted, "f1_macro": f1_macro}
    for i, label in ID_TO_LABEL.items():
        result[f"f1_{label}"] = per_class_f1[i]
    return result


# 4080 16GB: DistilBERT@bs32=~4GB VRAM, RoBERTa@bs16=~8GB VRAM — both safe
BATCH_SIZE = 32 if "distilbert" in MODEL_NAME else 16

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=8,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    logging_dir=str(OUTPUT_DIR / "logs"),
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,  # 0 on Windows to avoid multiprocessing issues
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("\nStarting training...")
train_result = trainer.train()

BEST_MODEL_DIR.mkdir(exist_ok=True)
trainer.save_model(str(BEST_MODEL_DIR))
tokenizer.save_pretrained(str(BEST_MODEL_DIR))

val_metrics = trainer.evaluate()
print("\nValidation metrics:")
for k, v in val_metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")

summary = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "num_labels": NUM_LABELS,
    "train_samples": len(dataset["train"]),
    "val_samples": len(dataset["validation"]),
    "best_val_f1_weighted": val_metrics.get("eval_f1_weighted"),
    "best_val_accuracy": val_metrics.get("eval_accuracy"),
    "training_time_seconds": train_result.metrics.get("train_runtime"),
    "timestamp": datetime.now().isoformat(),
}

with open(EVAL_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nBest val weighted F1: {val_metrics.get('eval_f1_weighted', 0):.4f}")
print(f"Best model saved to: {BEST_MODEL_DIR}")

f1 = val_metrics.get("eval_f1_weighted", 0)
if f1 < 0.88:
    print(f"\n⚠️  F1 {f1:.4f} < 0.88 threshold. Recommend running with roberta-base.")
    print("   Set MODEL_NAME=roberta-base and re-run this script.")
    exit(2)
else:
    print(f"\n✓  F1 {f1:.4f} meets threshold. Proceeding to evaluation.")
