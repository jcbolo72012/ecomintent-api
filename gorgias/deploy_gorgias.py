#!/usr/bin/env python3
"""
Modal.com serverless deployment for the EcomIntent Gorgias integration.

This is a SEPARATE Modal app from the inference API (ecomintent-api).
CPU-only — no GPU required for this service layer.

Deploy:
    modal deploy gorgias/deploy_gorgias.py

One-time webhook-secret setup:
    modal run gorgias/deploy_gorgias.py::create_webhook_secret

Required Modal secret (create once):
    modal secret create ecomintent-gorgias-secrets \\
        GORGIAS_APP_ID=<your-app-id> \\
        GORGIAS_APP_SECRET=<your-app-secret> \\
        APP_BASE_URL=https://john-72391--ecomintent-gorgias-app.modal.run \\
        INFERENCE_API_URL=https://john-72391--ecomintent-api-fastapi-app.modal.run \\
        STRIPE_SECRET_KEY=sk_live_... \\
        STRIPE_PRICE_ID=price_... \\
        STRIPE_PAYMENT_LINK=https://buy.stripe.com/... \\
        STRIPE_WEBHOOK_SECRET=whsec_...
"""
import os
import modal

# ── App & Image ───────────────────────────────────────────────────────────────

app = modal.App("ecomintent-gorgias")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "fastapi==0.115.6",
        "uvicorn==0.34.0",
        "pydantic==2.10.4",
        "httpx==0.27.0",
        "python-dotenv==1.0.1",
        "stripe==11.1.0",  # already included
        "python-multipart==0.0.20",
        "itsdangerous==2.2.0",   # HMAC signing helper
    ])
    .add_local_dir("gorgias", remote_path="/root/gorgias")
)

# ── Persistent volume (tokens, settings, logs) ────────────────────────────────

gorgias_volume = modal.Volume.from_name("ecomintent-gorgias-data", create_if_missing=True)

# ── Secrets ───────────────────────────────────────────────────────────────────
# Create with:
#   modal secret create ecomintent-gorgias-secrets \
#       GORGIAS_APP_ID=... GORGIAS_APP_SECRET=... APP_BASE_URL=... \
#       INFERENCE_API_URL=... STRIPE_SECRET_KEY=... STRIPE_PRICE_ID=... \
#       STRIPE_PAYMENT_LINK=... STRIPE_WEBHOOK_SECRET=...

gorgias_secret = modal.Secret.from_name("ecomintent-gorgias-secrets")


# ── Main ASGI endpoint ────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=None,                    # CPU-only — no ML inference here
    volumes={"/gorgias": gorgias_volume},
    secrets=[gorgias_secret],
    scaledown_window=300,        # Scale to zero after 5 min idle
    timeout=30,
    min_containers=0,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    """Serve the EcomIntent Gorgias FastAPI app."""
    # Import here so that the module path resolves inside the container
    import sys
    sys.path.insert(0, "/root")
    from gorgias.main import app as application  # noqa: E402
    return application


# ── One-time setup: HMAC webhook secret ──────────────────────────────────────

@app.function(
    image=image,
    gpu=None,
    volumes={"/gorgias": gorgias_volume},
    secrets=[gorgias_secret],
    timeout=60,
)
def create_webhook_secret():
    """
    One-time setup function.

    Generates a cryptographically random HMAC secret used to verify
    incoming Gorgias webhook payloads and persists it to the Modal Volume
    at /gorgias/webhook_secret.txt.

    Run with:
        modal run gorgias/deploy_gorgias.py::create_webhook_secret

    After running, copy the printed secret into your Gorgias app's
    webhook configuration (Shared Secret field).
    """
    import secrets
    from pathlib import Path

    secret_path = Path("/gorgias/webhook_secret.txt")

    if secret_path.exists():
        existing = secret_path.read_text().strip()
        print(f"Webhook secret already exists: {existing[:8]}... (not overwritten)")
        print("To regenerate, delete /gorgias/webhook_secret.txt from the volume first.")
        return

    # 32 bytes → 64-char hex string
    webhook_secret = secrets.token_hex(32)
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret_path.write_text(webhook_secret)

    # Commit to Modal Volume so the ASGI function can read it
    gorgias_volume.commit()

    app_base_url = os.environ.get("APP_BASE_URL", "<APP_BASE_URL not set>")
    print("\n" + "=" * 60)
    print("Webhook secret created and stored in Modal Volume.")
    print(f"\nSecret (first 8 chars): {webhook_secret[:8]}...")
    print(f"\nWebhook endpoint base URL:")
    print(f"  {app_base_url}/webhook/{{account_id}}")
    print("\nConfigure this URL + the secret in your Gorgias app dashboard.")
    print("=" * 60 + "\n")
