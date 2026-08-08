"""
EcomIntent Gorgias Integration — FastAPI application entry point.

Mounts:
  GET  /health          → liveness probe
  GET  /settings        → serves settings.html iframe page
  GET  /privacy         → privacy policy (required by Gorgias app listing)
  /oauth/*              → OAuth 2.0 install/callback flow  (oauth.py)
  /settings/*           → merchant settings CRUD + test   (settings.py)
  /webhook/*            → Gorgias ticket webhook handler   (webhook.py)

Startup:
  Creates /gorgias/{tokens,settings,logs} directories on the Modal Volume.
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ── Structured logging ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ecomintent.gorgias")

# ── Constants ─────────────────────────────────────────────────────────────────

SETTINGS_HTML_PATH = Path("/root/gorgias/ui/settings.html")
VOLUME_DIRS = [
    "/gorgias/tokens",
    "/gorgias/settings",
    "/gorgias/logs",
]


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create required volume directories on startup."""
    for d in VOLUME_DIRS:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        logger.info("Volume directory ready: %s", d)
    logger.info(
        "EcomIntent Gorgias integration started — version %s",
        application.version,
    )
    yield
    logger.info("EcomIntent Gorgias integration shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="EcomIntent — Auto Intent Tagger",
    description=(
        "Gorgias integration: automatically classifies incoming support tickets "
        "into 9 e-commerce intent categories and applies Gorgias tags."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

# Allow all origins so the Gorgias iframe can reach this service from any domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
# Imported here (not at module top) so that failed imports surface with clear
# tracebacks rather than crashing before the logging setup above runs.

try:
    from gorgias.oauth import router as oauth_router                  # noqa: E402
    from gorgias.settings import router as settings_router            # noqa: E402
    from gorgias.webhook import router as webhook_router              # noqa: E402
    from gorgias.stripe_webhook import router as stripe_router        # noqa: E402
except ImportError as e:
    logger.error("Failed to import sub-router: %s", e)
    raise

app.include_router(oauth_router,    tags=["OAuth"])
app.include_router(settings_router, tags=["Settings"])
app.include_router(webhook_router,  tags=["Webhook"])
app.include_router(stripe_router,   tags=["Stripe"])


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Liveness probe",
    response_description="Service status",
)
async def health() -> dict:
    """Returns 200 when the service is running."""
    return {
        "status": "ok",
        "service": "ecomintent-gorgias",
        "version": "1.0.0",
    }


@app.get(
    "/settings",
    response_class=HTMLResponse,
    summary="Settings UI",
    include_in_schema=False,
)
async def settings_page() -> HTMLResponse:
    """
    Serve the merchant settings page embedded in a Gorgias iframe.
    Reads ?account= from the query string to identify the merchant.
    """
    if not SETTINGS_HTML_PATH.exists():
        logger.error("settings.html not found at %s", SETTINGS_HTML_PATH)
        return HTMLResponse(
            content=(
                "<!DOCTYPE html><html><body>"
                "<h2 style='font-family:sans-serif;padding:2rem;color:#dc2626'>"
                "Settings UI not found — please redeploy the app.</h2>"
                "</body></html>"
            ),
            status_code=500,
        )
    try:
        html = SETTINGS_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(
            content=html,
            headers={
                # Allow embedding in Gorgias iframe; Gorgias manages its own CSP.
                "X-Frame-Options": "ALLOWALL",
                "Cache-Control": "no-store",
            },
        )
    except OSError as exc:
        logger.exception("Error reading settings.html: %s", exc)
        return HTMLResponse(content="Error loading settings page.", status_code=500)


@app.get("/setup", response_class=HTMLResponse, include_in_schema=False)
async def setup_guide() -> HTMLResponse:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>EcomIntent Setup Guide</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           max-width: 740px; margin: 48px auto; padding: 0 24px 80px;
           color: #1e293b; line-height: 1.7; }
    h1  { font-size: 26px; font-weight: 800; margin-bottom: 4px; }
    .sub { color: #64748b; font-size: 15px; margin-bottom: 40px; }
    h2  { font-size: 17px; font-weight: 700; margin: 36px 0 10px; color: #0f172a;
          display: flex; align-items: center; gap: 10px; }
    .n  { background: #3b82f6; color: #fff; border-radius: 50%; width: 28px; height: 28px;
          display: inline-flex; align-items: center; justify-content: center;
          font-size: 13px; font-weight: 800; flex-shrink: 0; }
    p   { color: #475569; margin-bottom: 12px; }
    code { background: #f1f5f9; padding: 2px 7px; border-radius: 5px; font-size: 13px; color: #1e293b; }
    .box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
           padding: 16px 20px; margin: 12px 0; }
    .box p { margin: 0; }
    table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 14px; }
    th { background: #f1f5f9; text-align: left; padding: 9px 14px; font-weight: 600; }
    td { padding: 9px 14px; border-top: 1px solid #e2e8f0; color: #475569; }
    .tag { background: #dbeafe; color: #1d4ed8; border-radius: 4px;
           padding: 2px 8px; font-size: 12px; font-weight: 600; font-family: monospace; }
    a { color: #3b82f6; }
    .faq-q { font-weight: 600; color: #1e293b; margin: 20px 0 4px; }
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 40px 0 30px; }
  </style>
</head>
<body>
  <h1>EcomIntent Setup Guide</h1>
  <p class="sub">Auto-tag your Gorgias support tickets with e-commerce intent labels in under 5 minutes.</p>

  <h2><span class="n">1</span> Install the app</h2>
  <p>Find <strong>EcomIntent</strong> in the Gorgias App Store and click <strong>Install</strong>. You'll be redirected to an authorization screen — click <strong>Authorize</strong> to grant access to your Gorgias account.</p>

  <h2><span class="n">2</span> Enter your API key</h2>
  <p>After authorizing you'll land on the EcomIntent settings page. Paste your EcomIntent API key into the <strong>API Key</strong> field and click <strong>Test</strong>. You should see a green <em>"Connected"</em> status.</p>
  <p>Don't have a key yet? Get one free at <a href="https://rapidapi.com/john-UG9kfZiW5/api/ecomintent-e-commerce-intent-classifie" target="_blank">RapidAPI</a>.</p>

  <h2><span class="n">3</span> Configure your settings</h2>
  <div class="box"><p><strong>Confidence threshold</strong> — default 70%. Tickets below this confidence level receive <code>ei-unclassified</code> instead of a specific intent tag. Raise it to reduce false positives; lower it to maximize coverage.</p></div>
  <div class="box"><p><strong>Active intents</strong> — toggle off any intent classes you don't want tagged. Disabled intents fall back to <code>ei-unclassified</code>.</p></div>
  <p>Click <strong>Save Settings</strong> when done.</p>

  <h2><span class="n">4</span> Intent tags applied to your tickets</h2>
  <table>
    <tr><th>Tag</th><th>When it fires</th></tr>
    <tr><td><span class="tag">ei-wismo</span></td><td>Where is my order / tracking / delivery status</td></tr>
    <tr><td><span class="tag">ei-return_request</span></td><td>Customer wants to return for a refund</td></tr>
    <tr><td><span class="tag">ei-exchange_request</span></td><td>Customer wants a different size, color, or variant</td></tr>
    <tr><td><span class="tag">ei-cancel_order</span></td><td>Customer wants to cancel before shipment</td></tr>
    <tr><td><span class="tag">ei-damaged_item</span></td><td>Broken, wrong, or missing item arrived</td></tr>
    <tr><td><span class="tag">ei-billing_dispute</span></td><td>Charge issues, refund status, payment problems</td></tr>
    <tr><td><span class="tag">ei-product_question</span></td><td>Specs, sizing, compatibility, availability</td></tr>
    <tr><td><span class="tag">ei-account_issue</span></td><td>Login, password, account access problems</td></tr>
    <tr><td><span class="tag">ei-unclassified</span></td><td>Below confidence threshold or intent disabled</td></tr>
  </table>

  <h2><span class="n">5</span> Build Gorgias automation rules</h2>
  <p>Go to <strong>Settings → Rules → Add Rule</strong> and use intent tags as triggers:</p>
  <div class="box"><p><strong>Route WISMO tickets:</strong> IF Tag contains <code>ei-wismo</code> → Assign to Fulfillment team + Apply "Tracking" macro</p></div>
  <div class="box"><p><strong>Prioritize damaged items:</strong> IF Tag contains <code>ei-damaged_item</code> → Set priority: Urgent + Notify manager</p></div>
  <div class="box"><p><strong>Auto-send return instructions:</strong> IF Tag contains <code>ei-return_request</code> → Apply "Return Portal" macro</p></div>

  <hr>
  <h2 style="margin-top:0">FAQ</h2>

  <p class="faq-q">How long does tagging take?</p>
  <p>Tags appear within 1–3 seconds of ticket creation. The first ticket of the day may take up to 10 seconds (cold start).</p>

  <p class="faq-q">What if EcomIntent is unreachable?</p>
  <p>The ticket is left untagged and your Gorgias workflows continue unaffected. No tickets are blocked or lost.</p>

  <p class="faq-q">What languages are supported?</p>
  <p>English only in v1. Multilingual support is on the roadmap.</p>

  <p class="faq-q">How do I uninstall?</p>
  <p>Go to Gorgias Settings → Apps → EcomIntent → Remove. All stored data is deleted within 30 days.</p>

  <p class="faq-q">Need help?</p>
  <p>Email <a href="mailto:support@ecomintent.com">support@ecomintent.com</a> — we respond within 1 business day.</p>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/terms", response_class=HTMLResponse, include_in_schema=False)
async def terms_of_service() -> HTMLResponse:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>EcomIntent — Terms of Service</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           max-width: 720px; margin: 48px auto; padding: 0 24px 80px;
           color: #1e293b; line-height: 1.7; }
    h1 { font-size: 24px; font-weight: 800; margin-bottom: 4px; }
    .sub { color: #64748b; font-size: 14px; margin-bottom: 40px; }
    h2 { font-size: 16px; font-weight: 700; margin: 32px 0 8px; color: #0f172a; }
    p  { color: #475569; margin-bottom: 12px; }
    a  { color: #3b82f6; }
  </style>
</head>
<body>
  <h1>EcomIntent — Terms of Service</h1>
  <p class="sub">Last updated: June 2026</p>

  <h2>1. Acceptance of Terms</h2>
  <p>By installing or using the EcomIntent Gorgias integration ("Service"), you agree to these Terms of Service. If you do not agree, do not use the Service.</p>

  <h2>2. Description of Service</h2>
  <p>EcomIntent automatically classifies incoming Gorgias support tickets into e-commerce intent categories and applies corresponding tags. The Service is provided via a Gorgias App Store integration backed by a machine learning inference API.</p>

  <h2>3. Permitted Use</h2>
  <p>You may use the Service to classify e-commerce customer support messages within your Gorgias helpdesk. The Service is intended for commercial use by e-commerce merchants and their support teams.</p>

  <h2>4. Prohibited Use</h2>
  <p>You may not use the Service to process sensitive personal data beyond standard e-commerce support context, attempt to reverse-engineer the underlying model, resell access to the Service, or use the Service in ways that violate applicable law.</p>

  <h2>5. Subscription and Billing</h2>
  <p>The Service is offered with a free 14-day trial followed by a paid subscription at $29/month. Billing is handled through Stripe. You may cancel at any time; cancellation takes effect at the end of the current billing period. No refunds are issued for partial periods.</p>

  <h2>6. Service Availability</h2>
  <p>We aim for high availability but do not guarantee uninterrupted service. The Service is provided "as is" without warranties of any kind. Classification results are probabilistic and should not be the sole basis for automated decisions with significant consequences.</p>

  <h2>7. Data Processing</h2>
  <p>By using the Service you authorize EcomIntent to access ticket text via the Gorgias API for the purpose of classification. Please review our <a href="/privacy">Privacy Policy</a> for full details on data handling.</p>

  <h2>8. Termination</h2>
  <p>We reserve the right to suspend or terminate access to the Service for accounts that violate these terms or abuse the API. You may terminate at any time by uninstalling the app from your Gorgias account.</p>

  <h2>9. Limitation of Liability</h2>
  <p>To the maximum extent permitted by law, EcomIntent's liability to you is limited to the fees paid in the 30 days preceding any claim. We are not liable for indirect, incidental, or consequential damages.</p>

  <h2>10. Changes to Terms</h2>
  <p>We may update these terms from time to time. Continued use of the Service after changes constitutes acceptance of the updated terms.</p>

  <h2>11. Contact</h2>
  <p>Questions about these terms: <a href="mailto:support@ecomintent.com">support@ecomintent.com</a></p>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get(
    "/privacy",
    response_class=HTMLResponse,
    summary="Privacy policy",
    include_in_schema=False,
)
async def privacy_policy() -> HTMLResponse:
    """
    Minimal privacy policy page required by the Gorgias app marketplace listing.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>EcomIntent — Privacy Policy</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           max-width: 720px; margin: 48px auto; padding: 0 24px;
           color: #1e293b; line-height: 1.7; }
    h1   { font-size: 24px; margin-bottom: 8px; }
    h2   { font-size: 16px; margin: 28px 0 8px; color: #334155; }
    p    { margin-bottom: 12px; color: #475569; }
    a    { color: #3b82f6; }
  </style>
</head>
<body>
  <h1>EcomIntent &mdash; Privacy Policy</h1>
  <p><em>Last updated: June 2026</em></p>

  <h2>Data We Access</h2>
  <p>EcomIntent accesses the <strong>subject and message body</strong> of new Gorgias
  support tickets via the Gorgias webhook API for the sole purpose of classifying
  customer intent.</p>

  <h2>Data Processing</h2>
  <p>Ticket text is forwarded to our inference API (hosted on Modal.com) for
  real-time classification. Text is processed transiently and is <strong>not
  stored</strong> beyond brief classification logs used for billing and
  error diagnostics.</p>

  <h2>Data Storage</h2>
  <p>We store: your Gorgias OAuth token (encrypted at rest), your app settings
  (confidence threshold, enabled intents), and summary webhook activity counters
  (no ticket content). All data is stored in Modal Volumes in US-East-1.</p>

  <h2>Data Sharing</h2>
  <p>We do not sell or share your data with third parties. Stripe is used solely
  for payment processing under their own privacy policy.</p>

  <h2>Data Deletion</h2>
  <p>To delete all stored data, uninstall the EcomIntent app from your Gorgias
  account and email <a href="mailto:support@ecomintent.com">support@ecomintent.com</a>.
  We will delete all records within 30 days.</p>

  <h2>Contact</h2>
  <p><a href="mailto:support@ecomintent.com">support@ecomintent.com</a></p>
</body>
</html>"""
    return HTMLResponse(content=html)
