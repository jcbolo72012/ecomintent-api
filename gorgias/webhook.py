"""
Webhook receiver for Gorgias ticket events.
Classifies inbound customer ticket text and applies an intent tag.

Note on auth: Gorgias HTTP integrations do NOT sign webhook payloads (no HMAC
header is sent). Auth is the unguessable per-account webhook URL plus a check
that the account_id maps to a stored token. If a signature header IS ever
present, we still verify it and reject tampering — but its absence is expected.
"""
import os
import json
import hmac
import base64
import hashlib
import logging
import re
import httpx
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request, Response
from dotenv import load_dotenv

from gorgias.oauth import get_valid_token, load_tokens
from gorgias.settings import load_settings, save_settings

load_dotenv()

router = APIRouter(tags=["webhook"])

GORGIAS_APP_SECRET = os.environ.get("GORGIAS_APP_SECRET", "")
INFERENCE_API_URL = "https://john-72391--ecomintent-api-fastapi-app.modal.run"
LOG_DIR = Path("/gorgias/logs")

logger = logging.getLogger("ecomintent.gorgias.webhook")

WEBHOOK_SECRET_FILE = Path("/gorgias/webhook_secret.txt")

# EcomIntent model intent -> Gorgias tag name. Explicit map so tags match the
# ones registered on install (avoids ei-return_request vs ei-return drift).
INTENT_TO_TAG = {
    "WISMO":            "ei-wismo",
    "RETURN_REQUEST":   "ei-return",
    "EXCHANGE_REQUEST": "ei-exchange",
    "CANCEL_ORDER":     "ei-cancel",
    "DAMAGED_ITEM":     "ei-damaged",
    "BILLING_DISPUTE":  "ei-billing",
    "PRODUCT_QUESTION": "ei-product-q",
    "ACCOUNT_ISSUE":    "ei-account",
    "OTHER":            "ei-other",
}
UNCLASSIFIED_TAG = "ei-unclassified"


def _candidate_secrets() -> dict[str, str]:
    """All secrets Gorgias *might* sign with, if it signs at all."""
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
    """Validate a Gorgias HMAC-SHA256 signature IF one is present.

    Returns True when a present signature matches a candidate secret.
    Returns False when the signature is present but matches nothing.
    A MISSING signature is handled by the caller (expected for HTTP integrations),
    so this only needs to speak to the present-but-checkable case.
    """
    if not signature_header:
        return False

    candidates = _candidate_secrets()
    if not candidates:
        return False

    for label, secret in candidates.items():
        expected = base64.b64encode(
            hmac.new(secret.encode(), body, hashlib.sha256).digest()
        ).decode()
        if hmac.compare_digest(expected, signature_header):
            logger.info("Webhook signature matched using: %s", label)
            return True

    logger.warning(
        "Webhook signature present but matched NO candidate. Tried: %s",
        list(candidates.keys()),
    )
    return False


def log_classification(account_id, ticket_id, text, intent, confidence, success):
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

    The integration `form` config sends a FLAT payload with string values:
      { "ticket_id": "123", "from_agent": "False", "body_text": "...", "subject": "..." }
    Falls back to nested event/object shapes just in case.
    """
    # Primary: flat shape our `form` template produces
    ticket_id = payload.get("ticket_id") or 0
    text = payload.get("body_text") or ""

    # Fallback: nested event/object shape
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

    if not text:
        text = payload.get("subject", "")

    # Strip HTML if present
    if text and "<" in text:
        text = re.sub(r"<[^>]+>", " ", text).strip()

    # Coerce ticket_id to int (templates deliver it as a string)
    try:
        ticket_id = int(ticket_id)
    except (ValueError, TypeError):
        ticket_id = 0

    return text.strip()[:512], ticket_id


def apply_tag(gorgias_domain: str, access_token: str, ticket_id: int, tag: str) -> bool:
    """Apply an intent tag to a Gorgias ticket.

    Confirmed endpoint: POST /api/tickets/{ticket_id}/tags  body {"names": [...]}
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
    ALWAYS returns 200 so Gorgias never disables the webhook.
    """
    body = await request.body()

    # TEMPORARY debug — remove once the loop is confirmed working end-to-end
    logger.info("WEBHOOK HEADERS: %s", dict(request.headers))
    logger.info("WEBHOOK BODY: %s", body[:1000])

    # ── Auth ──────────────────────────────────────────────────────────────
    # Gorgias HTTP integrations don't sign payloads. If a signature IS present
    # we verify it; if it's absent we rely on the unguessable per-account URL +
    # a known-account check below. Never 4xx here — always 200.
    sig = request.headers.get("X-Gorgias-Hmac-Sha256", "")
    if sig and not verify_signature(body, sig):
        logger.warning("Invalid signature for account %s", account_id)
        return Response(status_code=200)

    # ── Parse ─────────────────────────────────────────────────────────────
    try:
        payload = json.loads(body)
    except Exception:
        return Response(status_code=200)

    # Skip agent-authored messages. Gorgias sends from_agent as a STRING.
    if str(payload.get("from_agent", "")).lower() == "true":
        logger.info("Skipping agent message for account %s", account_id)
        return Response(status_code=200)

    # ── Account context ───────────────────────────────────────────────────
    tokens = load_tokens(account_id)
    if not tokens:
        logger.warning("No tokens found for account %s", account_id)
        return Response(status_code=200)

    settings = load_settings(account_id)
    threshold = settings.get("confidence_threshold", 0.70)
    # Default to ALL intents enabled if the account has none configured, so a
    # fresh install tags normally instead of sending everything to unclassified.
    enabled_intents = settings.get("enabled_intents") or list(INTENT_TO_TAG.keys())

    # ── Extract ───────────────────────────────────────────────────────────
    text, ticket_id = extract_ticket_text(payload)
    if not text or not ticket_id:
        logger.info("No usable text/ticket_id for account %s", account_id)
        return Response(status_code=200)

    access_token = get_valid_token(account_id)
    if not access_token:
        logger.warning("Could not get valid token for account %s", account_id)
        return Response(status_code=200)

    # ── Classify ──────────────────────────────────────────────────────────
    intent = "OTHER"
    confidence = 0.0
    tag = UNCLASSIFIED_TAG
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
                tag = INTENT_TO_TAG.get(intent, UNCLASSIFIED_TAG)
            else:
                tag = UNCLASSIFIED_TAG

            success = apply_tag(
                tokens["gorgias_domain"], access_token, ticket_id, tag
            )
            logger.info(
                "Classified ticket %s: intent=%s conf=%.3f tag=%s applied=%s",
                ticket_id, intent, confidence, tag, success,
            )
        else:
            logger.error("Classify API returned %s: %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Classification error for account %s: %s", account_id, e)

    log_classification(account_id, ticket_id, text, intent, confidence, success)

    # ── Stats ─────────────────────────────────────────────────────────────
    stats = settings.setdefault("webhook_stats", {})
    stats["total_classified"] = stats.get("total_classified", 0) + 1
    if not success:
        stats["total_errors"] = stats.get("total_errors", 0) + 1
    stats["last_processed_at"] = datetime.now(timezone.utc).isoformat()
    save_settings(account_id, settings)

    return Response(status_code=200)