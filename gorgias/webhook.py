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

WEBHOOK_SECRET_FILE = Path("/gorgias/webhook_secret.txt")


def _candidate_secrets() -> dict[str, str]:
    """
    All secrets Gorgias *might* be signing with. Used during verification to
    determine the real one. Returns {label: secret} for non-empty candidates.
    """
    candidates = {}
    if GORGIAS_APP_SECRET:
        candidates["app_secret"] = GORGIAS_APP_SECRET
    try:
        if WEBHOOK_SECRET_FILE.exists():
            file_secret = WEBHOOK_SECRET_FILE.read_text().strip()
            if file_secret:
                candidates["file_secret"] = file_secret
    except OSError:
        pass
    return candidates


def verify_signature(body: bytes, signature_header: str) -> bool:
    """Validate Gorgias HMAC-SHA256 webhook signature.

    DIAGNOSTIC MODE: tries every candidate secret and logs which one matched.
    Once you've confirmed the real secret from the logs, collapse this back to a
    single hmac.compare_digest against that secret.
    """
    if not signature_header:
        logger.warning("No signature header present on webhook")
        return False

    candidates = _candidate_secrets()
    if not candidates:
        logger.error("No webhook secret configured at all")
        return False

    for label, secret in candidates.items():
        expected = base64.b64encode(
            hmac.new(secret.encode(), body, hashlib.sha256).digest()
        ).decode()
        if hmac.compare_digest(expected, signature_header):
            logger.info("Webhook signature matched using: %s", label)
            return True

    logger.warning(
        "Webhook signature matched NO candidate. Tried: %s. "
        "Header value (first 16 chars): %s",
        list(candidates.keys()), signature_header[:16],
    )
    return False


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
    """Extract ticket body text and ticket ID from the Gorgias webhook payload.

    The integration's `form` config (see provisioning.py) sends a FLAT payload:
      { "ticket_id": ..., "from_agent": ..., "body_text": ..., "subject": ... }
    So we read those top-level fields directly. Falls back to the older nested
    shapes just in case a differently-configured integration ever calls in.
    """
    import re

    # Primary: the flat shape our `form` template produces
    ticket_id = payload.get("ticket_id") or 0
    text = payload.get("body_text") or ""

    # Fallback: nested event/object shape (older/other integrations)
    if not text and not ticket_id:
        try:
            obj = payload.get("event", {}).get("object") \
                  or payload.get("data", {}).get("object", {})
            ticket_id = obj.get("id", 0)
            messages = obj.get("messages", [])
            if messages:
                m = messages[0]
                text = m.get("body_text") or m.get("body_html", "")
            if not text:
                text = obj.get("subject", "")
        except Exception:
            pass

    # Fall back to subject if body came through empty
    if not text:
        text = payload.get("subject", "")

    # Strip HTML if we got an HTML body
    if text and "<" in text:
        text = re.sub(r"<[^>]+>", " ", text).strip()

    # Coerce ticket_id to int (templates may deliver it as a string)
    try:
        ticket_id = int(ticket_id)
    except (ValueError, TypeError):
        ticket_id = 0

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
