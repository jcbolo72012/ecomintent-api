"""
On-install provisioning. Runs once, right after OAuth succeeds:
  1. Create the 9 ei-* intent tags (Gorgias requires tags to exist before they
     can be applied to a ticket — you cannot create-on-apply).
  2. Register the webhook HTTP integration so Gorgias calls us on new messages.
"""
import os
import logging
import httpx

logger = logging.getLogger("ecomintent.gorgias.provisioning")

APP_BASE_URL = os.environ.get("APP_BASE_URL", "")

# The 9 intent tags, matching INTENT_TO_TAG in webhook.py, plus the fallback.
EI_TAGS = [
    "ei-wismo", "ei-return", "ei-exchange", "ei-cancel", "ei-damaged",
    "ei-billing", "ei-product-q", "ei-account", "ei-other", "ei-unclassified",
]


def create_tags(gorgias_domain: str, access_token: str, account_id: str) -> int:
    """Create the ei-* tags. Idempotent: a tag that already exists (409/422)
    counts as success. Returns the number of tags confirmed present."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    ok = 0
    for name in EI_TAGS:
        try:
            resp = httpx.post(
                f"https://{gorgias_domain}/api/tags",
                headers=headers,
                json={"name": name},
                timeout=10,
            )
            # 200/201 = created; 409/422 = already exists — both are fine.
            if resp.status_code in (200, 201, 409, 422):
                ok += 1
            else:
                logger.warning(
                    "Tag create '%s' for %s returned %s %s",
                    name, account_id, resp.status_code, resp.text,
                )
        except Exception as e:
            logger.warning("Tag create '%s' for %s error: %s", name, account_id, e)
    logger.info("Provisioned %d/%d tags for %s", ok, len(EI_TAGS), account_id)
    return ok


def register_webhook_integration(gorgias_domain: str, access_token: str, account_id: str) -> bool:
    webhook_url = f"{APP_BASE_URL}/gorgias/webhook/{account_id}"
    payload = {
        "name": "EcomIntent",
        "type": "http",
        "http": {
            "url": webhook_url,
            "method": "POST",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "triggers": {
                "ticket-message-created": True,
            },
            "form": {
                "ticket_id": "{{ticket.id}}",
                "from_agent": "{{ticket.messages[-1].from_agent}}",
                "body_text": "{{ticket.messages[-1].body_text}}",
                "subject": "{{ticket.subject}}",
            },
        },
    }
    try:
        resp = httpx.post(
            f"https://{gorgias_domain}/api/integrations",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            logger.info("Webhook integration registered for %s", account_id)
            return True
        logger.error(
            "Integration registration failed for %s: %s %s",
            account_id, resp.status_code, resp.text,
        )
        return False
    except Exception as e:
        logger.error("Integration registration error for %s: %s", account_id, e)
        return False


def provision(gorgias_domain: str, access_token: str, account_id: str) -> bool:
    """Full on-install provisioning: tags first (so tagging works immediately),
    then the webhook integration. Call this from the OAuth callback."""
    create_tags(gorgias_domain, access_token, account_id)
    return register_webhook_integration(gorgias_domain, access_token, account_id)