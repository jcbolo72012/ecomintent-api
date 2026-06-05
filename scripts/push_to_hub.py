#!/usr/bin/env python3
"""Push fine-tuned model and model card to HuggingFace Hub."""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

HF_USERNAME = os.environ["HF_USERNAME"]
HF_TOKEN = os.environ["HF_TOKEN"]
REPO_NAME = os.environ.get("HF_MODEL_REPO", "ecomintent-distilbert")
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi(token=HF_TOKEN)

try:
    create_repo(REPO_ID, private=False, exist_ok=True)
    print(f"Repo: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"Repo already exists or error: {e}")

api.upload_folder(
    folder_path="training/best_model",
    repo_id=REPO_ID,
    commit_message="Add fine-tuned ecommerce intent classifier",
)

if Path("MODEL_CARD.md").exists():
    api.upload_file(
        path_or_fileobj="MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        commit_message="Add model card",
    )

api.upload_file(
    path_or_fileobj="data/processed/label_config.json",
    path_in_repo="label_config.json",
    repo_id=REPO_ID,
)

if Path("eval/final_results.json").exists():
    api.upload_file(
        path_or_fileobj="eval/final_results.json",
        path_in_repo="eval_results.json",
        repo_id=REPO_ID,
    )

if Path("eval/confusion_matrix.png").exists():
    api.upload_file(
        path_or_fileobj="eval/confusion_matrix.png",
        path_in_repo="confusion_matrix.png",
        repo_id=REPO_ID,
    )

print(f"\nModel live at: https://huggingface.co/{REPO_ID}")
