# EcomIntent API — Launch Summary

## Live Endpoints

| Resource | URL |
|----------|-----|
| Modal API endpoint | https://john-72391--ecomintent-api-fastapi-app.modal.run |
| HuggingFace model | https://huggingface.co/JohnBolognino/ecomintent-distilbert |
| RapidAPI listing | https://rapidapi.com/john-UG9kfZiW5/api/ecomintent-e-commerce-intent-classifie |
| GitHub repo | https://github.com/jcbolo72012/ecomintent-api |

---

## Benchmark Numbers

| Metric | Result | Target | Pass? |
|--------|--------|--------|-------|
| Test accuracy (Bitext held-out) | 99.92% | ≥93% | ✓ |
| Weighted F1 (Bitext held-out) | 0.9992 | ≥0.93 | ✓ |
| P95 latency (Modal A10G, warm) | 3.9ms | ≤10ms | ✓ |
| Worst per-class F1 | 0.9979 (CANCEL_ORDER) | ≥0.80 | ✓ |
| Production smoke test | 9/9 passed @ 1.000 conf | 8/9 | ✓ |
| Training time | 10.9 min (RTX 4080 Laptop) | ≤45 min | ✓ |

> **Note:** 99.92% reflects performance on Bitext synthetic data (same template distribution as training). Real-world accuracy on production tickets estimated at 87–93%.

---

## Estimated Monthly Costs at Scale (Modal A10G = $1.10/hr)

| Monthly calls | Active GPU hours | Monthly cost |
|---------------|-----------------|--------------|
| 10,000 | ~1.4 hrs | ~$1.54 |
| 100,000 | ~14 hrs | ~$15.40 |
| 500,000 | ~70 hrs | ~$77 |
| 1,000,000 | ~140 hrs | ~$154 |
| 5,000,000 | ~700 hrs | ~$770 |

*Assumes 8ms P95 latency per call and 50 concurrent inputs.*

---

## Revenue Projections

| Month | Milestone | MRR |
|-------|-----------|-----|
| 1–2 | Core API on RapidAPI | $300 |
| 2–3 | HuggingFace model card + post | $500 |
| 3–5 | Gorgias plugin (10 merchants) | $900 |
| 5–7 | Gorgias compounds to 30 merchants | $1,700 |
| 7–9 | Shopify app (20 installs) | $2,400 |
| 9–12 | All channels compound | $3,600 |

---

## Remaining Human Tasks (from 02_HUMAN_TODOS.md)

### RapidAPI Listing (B5)
- [ ] Log in to rapidapi.com/provider
- [ ] Click "Add New API"
- [ ] Name: `EcomIntent - E-commerce Intent Classifier`
- [ ] Category: Machine Learning → Text Analysis
- [ ] Copy description from `docs/rapidapi_listing.md`
- [ ] Base URL: [MODAL_ENDPOINT_URL]
- [ ] Add endpoint: `POST /classify` with schema
- [ ] Add endpoint: `POST /classify/batch`
- [ ] Add endpoint: `GET /health`
- [ ] Add endpoint: `GET /intents`
- [ ] Set pricing plans (BASIC/PRO/ULTRA/MEGA)
- [ ] Enable overage billing
- [ ] Test all endpoints in RapidAPI console
- [ ] Submit for review (24hr approval)

### GitHub (B6)
- [ ] Confirm `.env` is NOT committed
- [ ] Update `README.md` with live URLs
- [ ] Add repo topics: nlp, intent-classification, ecommerce, customer-support, api, distilbert
- [ ] Add repo description and website URL
- [ ] `git push origin main`

### HuggingFace (B4)
- [ ] Fill in real eval numbers in `MODEL_CARD.md`
- [ ] Add your name as author
- [ ] Fill in live API URL and RapidAPI URL
- [ ] `python scripts/push_to_hub.py`
- [ ] Verify model loads in HuggingFace inference widget

### Launch Announcement (B7 — Optional)
- [ ] Post to HuggingFace Community (recommended first)
- [ ] Syndicate to dev.to 2 weeks later
