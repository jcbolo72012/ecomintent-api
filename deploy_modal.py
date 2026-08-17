#!/usr/bin/env python3
"""
Modal.com serverless deployment for EcomIntent API.
Uses A10G GPU with scale-to-zero (container_idle_timeout=300s).
Cost: ~$1.10/hr active, $0/hr idle.
"""
import os
import modal

app = modal.App("ecomintent-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "transformers==4.47.0",
        "fastapi==0.115.6",
        "uvicorn==0.34.0",
        "pydantic==2.10.4",
        "slowapi==0.1.9",
        "python-multipart==0.0.20",
        "huggingface_hub==0.27.0",
        "python-dotenv==1.0.1",
    ])
    .add_local_dir("api", remote_path="/root/api")  # bundle FastAPI app into image
)

model_volume = modal.Volume.from_name("ecomintent-model-weights", create_if_missing=True)
settings_volume = modal.Volume.from_name("ecomintent-settings", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token")


@app.function(
    image=image,
    gpu="A10G",
    scaledown_window=300,  # scale to zero after 5 min idle
    volumes={
        "/model": model_volume,
        "/settings": settings_volume,
    },
    secrets=[hf_secret],
    timeout=60,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def fastapi_app():
    os.environ["MODEL_PATH"] = "/model/best_model"
    os.environ["LABEL_CONFIG_PATH"] = "/model/best_model/label_config.json"
    from api.main import app as fastapi_application
    return fastapi_application


@app.function(
    image=image,
    volumes={"/model": model_volume},
    secrets=[hf_secret],
    timeout=600,
)
def upload_model(hf_model_id: str):
    """
    Download model from HuggingFace Hub into the Modal Volume.
    Run once after pushing the model to HuggingFace:
      modal run deploy_modal.py::upload_model --hf-model-id YOUR_USERNAME/ecomintent-distilbert
    """
    from huggingface_hub import snapshot_download
    import shutil
    from pathlib import Path

    print(f"Downloading {hf_model_id} to Modal volume...")
    cache_dir = snapshot_download(repo_id=hf_model_id)
    dest = Path("/model/best_model")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_dir, str(dest), dirs_exist_ok=True)
    print("Model uploaded to Modal volume.")
    model_volume.commit()


if __name__ == "__main__":
    with app.run():
        print("App deployed locally")
