"""
Per-account settings management for the Gorgias integration.
Stores settings in Modal Volume at /gorgias/settings/{account_id}.json
"""
import os
import json
import httpx
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(tags=["settings"])

SETTINGS_DIR = Path("/gorgias/settings")
INFERENCE_API_URL = "https://john-72391--ecomintent-api-fastapi-app.modal.run"

INTENT_LABELS = [
    "WISMO", "RETURN_REQUEST", "EXCHANGE_REQUEST", "CANCEL_ORDER",
    "DAMAGED_ITEM", "BILLING_DISPUTE", "PRODUCT_QUESTION", "ACCOUNT_ISSUE", "OTHER",
]

DEFAULT_SETTINGS = {
    "api_key": "",
    "confidence_threshold": 0.70,
    "enabled_intents": INTENT_LABELS,
    "plan": "free",
    "stripe_customer_id": "",
    "created_at": "",
    "webhook_stats": {
        "total_classified": 0,
        "total_errors": 0,
        "last_processed_at": None,
    },
}


class SettingsUpdate(BaseModel):
    api_key: Optional[str] = None
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    enabled_intents: Optional[list[str]] = None


class TestConnectionRequest(BaseModel):
    api_key: str


def load_settings(account_id: str) -> dict:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = SETTINGS_DIR / f"{account_id}.json"
    if not path.exists():
        settings = dict(DEFAULT_SETTINGS)
        settings["created_at"] = datetime.now(timezone.utc).isoformat()
        save_settings(account_id, settings)
        return settings
    with open(path) as f:
        return json.load(f)


def save_settings(account_id: str, data: dict):
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_DIR / f"{account_id}.json", "w") as f:
        json.dump(data, f, indent=2)


@router.post("/settings/test")
async def test_connection(body: TestConnectionRequest):
    """Test that the EcomIntent API is reachable and responding."""
    try:
        resp = httpx.get(
            f"{INFERENCE_API_URL}/health",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "status": "ok",
                "message": f"Connected — model v{data.get('version', '1.0.0')}",
                "device": data.get("device", "unknown"),
            }
        return {"status": "error", "message": f"API returned {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}


@router.get("/settings/{account_id}")
async def get_settings(account_id: str):
    settings = load_settings(account_id)
    # Inject payment link with account_id so Stripe can match payment to merchant
    base_link = os.environ.get(
        "STRIPE_PAYMENT_LINK", "https://buy.stripe.com/00waEY1Cr6081mT3qWcMM00"
    )
    settings["stripe_payment_link"] = f"{base_link}?client_reference_id={account_id}"
    return settings


@router.post("/settings/{account_id}")
async def update_settings(account_id: str, body: SettingsUpdate):
    settings = load_settings(account_id)
    if body.api_key is not None:
        settings["api_key"] = body.api_key
    if body.confidence_threshold is not None:
        settings["confidence_threshold"] = body.confidence_threshold
    if body.enabled_intents is not None:
        valid = [i for i in body.enabled_intents if i in INTENT_LABELS]
        settings["enabled_intents"] = valid
    save_settings(account_id, settings)
    return {"status": "saved", "settings": settings}


