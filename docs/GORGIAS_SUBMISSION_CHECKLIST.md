# Gorgias App Store Submission Checklist

---

## Pre-Submission: Technical Requirements

### OAuth2
- [ ] GET /oauth/authorize redirects to correct Gorgias OAuth URL
- [ ] GET /oauth/callback exchanges code for tokens successfully
- [ ] Tokens stored securely (not in URL params, not in logs)
- [ ] Token refresh works before expiry
- [ ] App handles revoked tokens gracefully (re-prompts OAuth)

### Webhook
- [ ] POST /gorgias/webhook/{account_id} is publicly reachable
- [ ] HMAC-SHA256 signature validated on every request
- [ ] Invalid signatures return 200 (not 401 — prevents Gorgias disabling webhook)
- [ ] Webhook responds within 5 seconds (Gorgias timeout)
- [ ] All exceptions caught — webhook never returns 5xx
- [ ] Webhook handles missing/malformed payloads gracefully

### Settings UI
- [ ] Settings page loads correctly in Gorgias iframe (no CSP errors)
- [ ] Page works on mobile (responsive layout)
- [ ] API key field does not auto-submit on Enter
- [ ] Settings persist across page reloads
- [ ] Confidence threshold slider updates value display in real time

### General
- [ ] All endpoints respond within 30 seconds
- [ ] App handles Gorgias API rate limits (retry with backoff)
- [ ] No hardcoded credentials anywhere in code
- [ ] Privacy policy page is live and accessible

---

## Pre-Submission: App Listing Requirements

- [ ] App name: "EcomIntent — Auto Intent Tagger"
- [ ] Short description ≤ 160 characters
- [ ] Long description explains what it does, how it works, and why it's useful
- [ ] 3 screenshots uploaded — BACKLOG ITEM (replace placeholder screenshots before approval)
      Screenshot 1: Open https://john-72391--ecomintent-gorgias-fastapi-app.modal.run/settings?account=bolognino — screenshot the settings page
      Screenshot 2: Manually tag a ticket in bolognino.gorgias.com with ei-wismo, screenshot the ticket sidebar
      Screenshot 3: Go to Gorgias Settings → Rules → Add Rule, set trigger to ei-wismo, screenshot the rule editor
- [ ] App icon/logo uploaded (use docs/logo.svg)
- [ ] Category set: Productivity / Automation
- [ ] Pricing configured: Free trial + $29/mo
- [ ] Support email set: support@ecomintent.com
- [ ] Privacy policy URL live and accessible
- [ ] Website URL set

---

## Testing Before Submission

### With your own Gorgias account (bolognino.gorgias.com)
- [ ] Complete OAuth install flow end to end
- [ ] Open settings page — confirm it loads in Gorgias iframe
- [ ] Test API key connection — confirm green status
- [ ] Adjust threshold — confirm it saves
- [ ] Toggle an intent off — confirm it saves
- [ ] Create a test ticket: "where is my order"
- [ ] Verify `ei-wismo` tag appears on the ticket within 5 seconds
- [ ] Create a test ticket: "I need to return this"
- [ ] Verify `ei-return_request` tag appears
- [ ] Create a ticket with very short text: "hi"
- [ ] Verify `ei-unclassified` tag appears (below threshold)
- [ ] Check /gorgias/logs/classifications.jsonl — confirm entries are being written

---

## Submission Steps

1. Go to **https://partners.gorgias.com**
2. Open your app → click **Submit for review**
3. Fill in all listing fields (copy from gorgias_app_store_listing.md)
4. Upload 3 screenshots
5. Upload logo (docs/logo.svg)
6. Set OAuth redirect URL: `https://john-72391--ecomintent-gorgias-app.modal.run/oauth/callback`
7. Set webhook URL template: `https://john-72391--ecomintent-gorgias-app.modal.run/gorgias/webhook/{account_id}`
8. Click **Submit**

**Review timeline:** 1–3 weeks. Gorgias will test your OAuth flow and webhook manually.

---

## Common Rejection Reasons

| Reason | Prevention |
|--------|-----------|
| Webhook times out | Ensure Modal container stays warm; response must be < 5s |
| OAuth callback fails | Test with fresh incognito window before submitting |
| Settings page broken in iframe | Test in Gorgias admin, not standalone browser tab |
| Missing privacy policy | Ensure /privacy URL is live |
| Screenshots show localhost | Screenshots must show production URL |
| No error handling | Every code path must return 200; never let exceptions propagate |

---

## Post-Approval Steps

- [ ] Update APP_BASE_URL in .env and Modal secrets to final production URL
- [ ] Re-deploy: `modal deploy gorgias/deploy_gorgias.py`
- [ ] Set up Stripe webhook in Stripe dashboard pointing to `/stripe/webhook`
- [ ] Test real merchant install on bolognino.gorgias.com
- [ ] Add app listing URL to README.md and MODEL_CARD.md
- [ ] Announce in HuggingFace community and dev.to (follow-up post)
- [ ] DM 5 Shopify merchants in relevant Facebook groups offering free beta access
