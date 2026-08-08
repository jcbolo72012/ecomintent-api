"""
Webhook receiver for Gorgias ticket events.
Validates HMAC signature, classifies ticket text, applies intent tag.
"""
import os
import json
import hmac
import base64
import hashlib
import logging
import httpx
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, Request, Response
from dotenv import load_dotenv

from gorgias.oauth import get_valid_token, load_tokens
from gorgias.settings import load_settings

load_dotenv()

router = APIRouter(tags=["webhook"])

GORGIAS_APP_SECRET = os.environ.get("GORGIAS_APP_SECRET", "")
INFERENCE_API_URL = "https://john-72391--ecomintent-api-fastapi-app.modal.run"
LOG_DIR = Path("/gorgias/logs")

logger = logging.getLogger("ecomintent.gorgias.webhook")


def verify_signature(body: bytes, signature_header: str) -> bool:
    """Validate Gorgias HMAC-SHA256 webhook signature."""
    if not signature_header:
        return False
    expected = base64.b64encode(
        hmac.new(
            GORGIAS_APP_SECRET.encode(),
            body,
            hashlib.sha256,
        ).digest()
    ).decode()
    return hmac.compare_digest(expected, signature_header)


def log_classification(account_id: str, ticket_id: int, text: str, intent: str, confidence: float, success: bool):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": account_id,
        "ticket_id": ticket_id,
        "text_preview": text[:80],
        "intent": intent,
        "confidence": confidence,
        "success": success,
    }
    with open(LOG_DIR / "classifications.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def extract_ticket_text(payload: dict) -> tuple[str, int]:
    """Extract ticket body text and ticket ID from Gorgias webhook payload."""
    ticket_id = 0
    text = ""
    try:
        obj = payload.get("event", {}).get("object", payload.get("data", {}).get("object", {}))
        ticket_id = obj.get("id", 0)
        # Try message body first
        messages = obj.get("messages", [])
        if messages:
            msg = messages[0]
            text = msg.get("body_text") or msg.get("body_html", "")
            # Strip basic HTML if body_text not available
            if "<" in text:
                import re
                text = re.sub(r"<[^>]+>", " ", text).strip()
        # Fall back to subject
        if not text:
            text = obj.get("subject", "")
    except Exception:
        pass
    return text.strip()[:512], ticket_id


def apply_tag(gorgias_domain: str, access_token: str, ticket_id: int, tag: str) -> bool:
    """Apply an intent tag to a Gorgias ticket.

    Confirmed endpoint: POST /api/tickets/{ticket_id}/tags  body {"names": [...]}
    Tags are created on the fly if they don't exist yet, so no pre-provisioning
    of the tag itself is strictly required for the tag to apply.
    """
    try:
        resp = httpx.post(
            f"https://{gorgias_domain}/api/tickets/{ticket_id}/tags",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"names": [tag]},
            timeout=8,
        )
        if resp.status_code not in (200, 201, 204):
            logger.error(
                "Tag apply failed for ticket %s: %s %s",
                ticket_id, resp.status_code, resp.text,
            )
        return resp.status_code in (200, 201, 204)
    except Exception as e:
        logger.error(f"Tag apply failed for ticket {ticket_id}: {e}")
        return False


@router.post("/gorgias/webhook/{account_id}")
async def receive_webhook(account_id: str, request: Request):
    """
    Receive a Gorgias ticket event, classify the text, and apply an intent tag.
    Always returns 200 to prevent Gorgias from disabling the webhook.
    """
    body = await request.body()

    # TEMPORARY debug — remove after confirming signature model in sandbox
    logger.info("WEBHOOK HEADERS: %s", dict(request.headers))
    logger.info("WEBHOOK BODY: %s", body[:1000])

    # Validate signature
    sig = request.headers.get("X-Gorgias-Hmac-Sha256", "")
    if not verify_signature(body, sig):
        logger.warning(f"Invalid signature for account {account_id}")
        return Response(status_code=200)  # Return 200 to avoid webhook disable

    # Parse payload
    try:
        payload = json.loads(body)
    except Exception:
        return Response(status_code=200)

    # Load account tokens and settings
    tokens = load_tokens(account_id)
    if not tokens:
        logger.warning(f"No tokens found for account {account_id}")
        return Response(status_code=200)

    settings = load_settings(account_id)
    threshold = settings.get("confidence_threshold", 0.70)
    enabled_intents = settings.get("enabled_intents", [])

    # Extract ticket text
    text, ticket_id = extract_ticket_text(payload)
    if not text or not ticket_id:
        return Response(status_code=200)

    # Get valid access token
    access_token = get_valid_token(account_id)
    if not access_token:
        logger.warning(f"Could not get valid token for account {account_id}")
        return Response(status_code=200)

    # Classify
    intent = "OTHER"
    confidence = 0.0
    tag = "ei-unclassified"
    success = False

    try:
        resp = httpx.post(
            f"{INFERENCE_API_URL}/classify",
            json={"text": text, "threshold": threshold},
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            intent = result.get("intent", "OTHER")
            confidence = result.get("confidence", 0.0)
            below = result.get("below_threshold", False)

            if not below and intent in enabled_intents:
                tag = f"ei-{intent.lower()}"
            else:
                tag = "ei-unclassified"

            # Apply tag to ticket
            success = apply_tag(
                tokens["gorgias_domain"], access_token, ticket_id, tag
            )
    except Exception as e:
        logger.error(f"Classification error for account {account_id}: {e}")

    log_classification(account_id, ticket_id, text, intent, confidence, success)

    # Update webhook stats
    settings["webhook_stats"]["total_classified"] = settings["webhook_stats"].get("total_classified", 0) + 1
    if not success:
        settings["webhook_stats"]["total_errors"] = settings["webhook_stats"].get("total_errors", 0) + 1
    settings["webhook_stats"]["last_processed_at"] = datetime.now(timezone.utc).isoformat()

    from gorgias.settings import save_settings
    save_settings(account_id, settings)

    return Response(status_code=200)
