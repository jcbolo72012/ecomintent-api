"""
Gorgias OAuth2 flow handler.
Implements authorization, callback, token storage, and refresh.
"""
import os
import json
import time
import hmac
import hashlib
import httpx
import secrets as _secrets
from urllib.parse import urlencode
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/oauth", tags=["oauth"])

GORGIAS_APP_ID = os.environ.get("GORGIAS_APP_ID", "")
GORGIAS_APP_SECRET = os.environ.get("GORGIAS_APP_SECRET", "")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "")
REDIRECT_URI = f"{APP_BASE_URL}/oauth/callback"
TOKEN_DIR = Path("/gorgias/tokens")


def save_tokens(account_id: str, data: dict):
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_DIR / f"{account_id}.json", "w") as f:
        json.dump(data, f, indent=2)


def load_tokens(account_id: str) -> dict | None:
    path = TOKEN_DIR / f"{account_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def refresh_access_token(gorgias_domain: str, refresh_tok: str) -> dict | None:
    try:
        resp = httpx.post(
            f"https://{gorgias_domain}/oauth/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_tok,
                "client_id": GORGIAS_APP_ID,
                "client_secret": GORGIAS_APP_SECRET,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token", refresh_tok),
                "expires_at": time.time() + data.get("expires_in", 604800),
            }
    except Exception:
        pass
    return None


def get_valid_token(account_id: str) -> str | None:
    """Return a valid access token, refreshing if expiring within 5 minutes."""
    tokens = load_tokens(account_id)
    if not tokens:
        return None
    if tokens.get("expires_at", 0) - time.time() < 300:
        refreshed = refresh_access_token(
            tokens["gorgias_domain"], tokens["refresh_token"]
        )
        if refreshed:
            tokens.update(refreshed)
            save_tokens(account_id, tokens)
        else:
            return None
    return tokens.get("access_token")


@router.get("/authorize")
async def authorize(account: str):
    """Kick off OAuth flow. account = Gorgias subdomain (e.g. 'mystore')."""
    if not account:
        raise HTTPException(400, "account parameter required")
    params = {
        "response_type": "code",
        "client_id": GORGIAS_APP_ID,
        "scope": "openid email profile offline write:all",
        "redirect_uri": REDIRECT_URI,
        "state": account,
        "nonce": _secrets.token_hex(8),
    }
    url = f"https://{account}.gorgias.com/oauth/authorize?{urlencode(params)}"
    return RedirectResponse(url)


@router.get("/callback")
async def callback(code: str = None, state: str = None, error: str = None):
    """Handle OAuth callback from Gorgias."""
    if error:
        return HTMLResponse(
            f"<h2>Authorization failed: {error}</h2><p>Please try again.</p>",
            status_code=400,
        )
    if not code or not state:
        raise HTTPException(400, "Missing code or state parameter")

    gorgias_domain = f"{state}.gorgias.com"
    resp = httpx.post(
        f"https://{gorgias_domain}/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": GORGIAS_APP_ID,
            "client_secret": GORGIAS_APP_SECRET,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=15,
    )
    if resp.status_code != 200:
        import logging
        logging.getLogger("ecomintent.gorgias.oauth").error(
            "Token exchange failed: status=%s body=%s", resp.status_code, resp.text
        )
        raise HTTPException(400, f"Token exchange failed ({resp.status_code}): {resp.text}")

    token_data = resp.json()
    account_id = state

    save_tokens(account_id, {
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token", ""),
        "account_id": account_id,
        "gorgias_domain": gorgias_domain,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": time.time() + token_data.get("expires_in", 604800),
    })

    # ── NEW: register the webhook so Gorgias starts sending ticket events ──
    from gorgias.provisioning import register_webhook_integration
    register_webhook_integration(
        gorgias_domain, token_data["access_token"], account_id
    )

    return RedirectResponse(f"{APP_BASE_URL}/settings?account={account_id}")
