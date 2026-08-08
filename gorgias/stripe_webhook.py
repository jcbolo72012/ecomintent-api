"""
Stripe webhook handler for EcomIntent Gorgias integration.
Handles subscription lifecycle events to upgrade/downgrade merchant plans.

The merchant's Gorgias account_id is passed as client_reference_id on the
Stripe Payment Link URL so we can match payments to accounts.
"""
import os
import json
import logging
import stripe
from fastapi import APIRouter, Request, Response
from dotenv import load_dotenv

from gorgias.settings import load_settings, save_settings

load_dotenv()

router = APIRouter(tags=["stripe"])
logger = logging.getLogger("ecomintent.gorgias.stripe")

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


def upgrade_account(account_id: str, stripe_customer_id: str):
    """Set merchant plan to pro."""
    settings = load_settings(account_id)
    settings["plan"] = "pro"
    settings["stripe_customer_id"] = stripe_customer_id
    save_settings(account_id, settings)
    logger.info(f"Upgraded account {account_id} to Pro (customer={stripe_customer_id})")


def downgrade_account(account_id: str):
    """Set merchant plan back to free."""
    settings = load_settings(account_id)
    settings["plan"] = "free"
    save_settings(account_id, settings)
    logger.info(f"Downgraded account {account_id} to Free")


def find_account_by_customer(customer_id: str) -> str | None:
    """Search settings files for a matching Stripe customer ID."""
    from pathlib import Path
    settings_dir = Path("/gorgias/settings")
    if not settings_dir.exists():
        return None
    for f in settings_dir.glob("*.json"):
        try:
            with open(f) as fp:
                s = json.load(fp)
            if s.get("stripe_customer_id") == customer_id:
                return f.stem  # filename without .json = account_id
        except Exception:
            continue
    return None


@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Receive and process Stripe webhook events.
    Always returns 200 to prevent Stripe from disabling the endpoint.
    """
    body = await request.body()
    sig = request.headers.get("stripe-signature", "")

    # Validate signature
    try:
        event = stripe.Webhook.construct_event(body, sig, WEBHOOK_SECRET)
    except stripe.errors.SignatureVerificationError:
        logger.warning("Invalid Stripe webhook signature")
        return Response(status_code=400, content="Invalid signature")
    except Exception as e:
        logger.error(f"Webhook parse error: {e}")
        return Response(status_code=200)

    event_type = event["type"]
    logger.info(f"Stripe event received: {event_type}")

    try:
        if event_type == "checkout.session.completed":
            session = event["data"]["object"]
            account_id = session.get("client_reference_id")
            customer_id = session.get("customer")
            if account_id and customer_id:
                upgrade_account(account_id, customer_id)
            else:
                logger.warning(f"checkout.session.completed missing account_id or customer_id")

        elif event_type == "customer.subscription.created":
            sub = event["data"]["object"]
            customer_id = sub.get("customer")
            # Try to find account by customer ID (set during checkout)
            account_id = find_account_by_customer(customer_id)
            if account_id:
                upgrade_account(account_id, customer_id)
            else:
                logger.warning(f"subscription.created: no account found for customer {customer_id}")

        elif event_type == "customer.subscription.deleted":
            sub = event["data"]["object"]
            customer_id = sub.get("customer")
            account_id = find_account_by_customer(customer_id)
            if account_id:
                downgrade_account(account_id)
            else:
                logger.warning(f"subscription.deleted: no account found for customer {customer_id}")

        elif event_type == "invoice.payment_failed":
            invoice = event["data"]["object"]
            customer_id = invoice.get("customer")
            attempt = invoice.get("attempt_count", 1)
            logger.warning(f"Payment failed for customer {customer_id} (attempt {attempt})")
            # Downgrade after 3 failed attempts (Stripe's default retry schedule)
            if attempt >= 3:
                account_id = find_account_by_customer(customer_id)
                if account_id:
                    downgrade_account(account_id)

    except Exception as e:
        logger.error(f"Error processing {event_type}: {e}")

    return Response(status_code=200)
