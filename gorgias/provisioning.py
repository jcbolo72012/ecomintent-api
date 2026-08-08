"""
On-install provisioning: register the webhook integration so Gorgias calls us
on new ticket messages. Runs once, right after OAuth succeeds.
"""
import os
import logging
import httpx

logger = logging.getLogger("ecomintent.gorgias.provisioning")

APP_BASE_URL = os.environ.get("APP_BASE_URL", "")

# TODO(verify) against developers.gorgias.com "Create an integration" +
# "The Integration object": the exact `http` config shape, the trigger-event
# field name, and the event slug for "customer message created". The structure
# below matches the confirmed POST /api/integrations endpoint; confirm the inner
# http object field names by inspecting one real integration in the sandbox.
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
            # Gorgias templates the request body from ticket context.
            # This sends the ticket id + latest message so the webhook can classify.
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
