# Gorgias App Store Listing — EcomIntent Auto Intent Tagger

---

## App Name
EcomIntent — Auto Intent Tagger

## Short Description (≤160 chars)
Auto-tags incoming support tickets with intent labels (WISMO, returns, exchanges, etc.) using AI. Build Gorgias automation rules on top.

---

## Long Description

**Turn every incoming ticket into a routing signal — automatically.**

EcomIntent reads each new support ticket the moment it arrives in Gorgias and applies an intent tag in under a second. No manual triage. No keyword rules. Just clean, consistent labels your automation rules can act on.

**9 labels built specifically for e-commerce support:**
`ei-wismo` · `ei-return_request` · `ei-exchange_request` · `ei-cancel_order` · `ei-damaged_item` · `ei-billing_dispute` · `ei-product_question` · `ei-account_issue` · `ei-unclassified`

**Build automation rules on top of intent tags — no coding required.**
Once tickets are tagged, use Gorgias's built-in Rules to act on them. Examples:
- IF tag = `ei-wismo` → assign to Fulfillment team + apply "Tracking" macro
- IF tag = `ei-return_request` → assign to Returns team + send return portal link
- IF tag = `ei-damaged_item` → set priority = Urgent + notify manager
- IF tag = `ei-unclassified` → assign to General queue for human review

Merchants typically automate 30–40% of their ticket volume within the first week. The tags handle routing; your team handles the actual responses.

**Setup takes under 5 minutes.**
Install the app, enter your EcomIntent API key, adjust the confidence threshold if needed, and you're done. The webhook registers automatically. No developer required, no Zapier middleman, no ongoing maintenance.

**Built for e-commerce, not generic AI.**
EcomIntent was fine-tuned specifically on e-commerce customer support data — it understands the difference between a return (refund) and an exchange (swap), and between a billing dispute and a general complaint. Generic LLMs don't make these distinctions reliably. EcomIntent does.

---

## Screenshots (Describe 3)

**Screenshot 1 — Settings Page**
The EcomIntent settings page embedded in Gorgias admin. Shows:
- API key field with green "Connected" status
- Confidence threshold slider set to 70%
- Per-intent toggle switches, all enabled
- Plan section showing "Upgrade to Pro" button

**Screenshot 2 — Tagged Ticket**
A Gorgias ticket view showing a customer message ("where is my order, it's been 5 days") with the `ei-wismo` tag applied in the Tags sidebar. Shows how the tag appears alongside other Gorgias tags.

**Screenshot 3 — Automation Rule**
Gorgias Rules configuration screen showing a rule: IF tag contains `ei-wismo` THEN assign to "Fulfillment" team AND apply macro "WISMO — Tracking Response". Demonstrates the end-to-end value.

---

## App Details

| Field | Value |
|-------|-------|
| Category | Productivity / Automation |
| Pricing | Free 14-day trial, then $29/month |
| Support email | support@ecomintent.com |
| Privacy policy | https://john-72391--ecomintent-gorgias-app.modal.run/privacy |
| App URL | https://john-72391--ecomintent-gorgias-app.modal.run |
| OAuth redirect | https://john-72391--ecomintent-gorgias-app.modal.run/oauth/callback |
