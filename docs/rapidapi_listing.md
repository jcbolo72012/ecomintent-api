# RapidAPI Listing — EcomIntent API

## Short Description (≤160 chars)

E-commerce intent classifier. 9 labels (WISMO, returns, exchanges, etc). 93%+ accuracy. 10ms latency. 15x cheaper than GPT-4o mini.

---

## Full Description

**EcomIntent** is a production-ready intent classification API purpose-built for e-commerce customer support. It classifies incoming support tickets, chat messages, and emails into one of 9 actionable intent categories — in under 10ms, at a fraction of the cost of general-purpose LLMs.

### What It Does

Send any customer support message to the API and get back:
- **Intent label** (one of 9 classes)
- **Confidence score** (0.0–1.0)
- **Full score breakdown** across all 9 classes
- **Below-threshold flag** — if confidence is low, falls back to OTHER

### The 9 Intent Classes

| Label | When It Fires |
|-------|---------------|
| `WISMO` | "where is my order", tracking inquiries, delivery status |
| `RETURN_REQUEST` | Refund requests, return initiations |
| `EXCHANGE_REQUEST` | Size swap, color change, variant exchange |
| `CANCEL_ORDER` | Pre-shipment cancellation requests |
| `DAMAGED_ITEM` | Broken, wrong, or missing items |
| `BILLING_DISPUTE` | Double charges, unauthorized charges, missing refunds |
| `PRODUCT_QUESTION` | Specs, sizing, compatibility, availability |
| `ACCOUNT_ISSUE` | Login failures, password resets, account access |
| `OTHER` | Catch-all for OOS messages, greetings, ambiguous |

### Why Not GPT-4o Mini?

| | **EcomIntent** | GPT-4o mini |
|---|---|---|
| Accuracy on e-comm tickets | ~93%+ | ~84–88% |
| P95 latency | ~8ms | ~450–700ms |
| Cost per 1k calls | ~$0.001 | $0.015–$0.045 |
| Domain specialization | ✅ E-commerce taxonomy | ❌ Generic |

### Use Cases

**Helpdesk Routing**
Automatically route tickets to the right queue before an agent sees them. WISMO → Fulfillment. Returns → Returns team. Billing disputes → Finance.

**Shopify / Gorgias / Zendesk Integration**
Tag tickets with intent labels as they arrive. Build automation rules on top of those tags. Zero ML knowledge required.

**Response Triggering**
When confidence ≥ 85% and intent = WISMO, auto-send a tracking link. Handles the easiest 40% of your ticket volume with no human intervention.

**Analytics**
Understand what your customers actually ask about. Weekly intent volume charts. Detect spikes (damaged items spike → quality issue upstream).

**LLM Preprocessing**
Pass intent + confidence as context to your LLM response generator. Dramatically reduces hallucinations on routing logic.

---

## Pricing Plans

| Plan | Monthly Price | Requests/Month | Overage |
|------|--------------|----------------|---------|
| **BASIC** (Free) | $0 | 500 | — |
| **PRO** | $29/mo | 50,000 | $0.002/call |
| **ULTRA** | $99/mo | 500,000 | $0.001/call |
| **MEGA** | $299/mo | Unlimited | Custom |

---

## Code Examples

### Python

```python
import requests

url = "https://ecomintent-api.p.rapidapi.com/classify"
headers = {
    "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY",
    "X-RapidAPI-Host": "ecomintent-api.p.rapidapi.com",
    "Content-Type": "application/json",
}
payload = {"text": "where is my order it's been 5 days"}

response = requests.post(url, json=payload, headers=headers)
result = response.json()

print(result["intent"])      # "WISMO"
print(result["confidence"])  # 0.9872
print(result["latency_ms"])  # 7.3
```

### JavaScript / Node.js

```javascript
const axios = require('axios');

const options = {
  method: 'POST',
  url: 'https://ecomintent-api.p.rapidapi.com/classify',
  headers: {
    'X-RapidAPI-Key': 'YOUR_RAPIDAPI_KEY',
    'X-RapidAPI-Host': 'ecomintent-api.p.rapidapi.com',
    'Content-Type': 'application/json',
  },
  data: { text: "I need to return these shoes, they don't fit" },
};

const response = await axios.request(options);
console.log(response.data.intent);      // "RETURN_REQUEST"
console.log(response.data.confidence); // 0.9641
```

### Batch Classification (Python)

```python
import requests

url = "https://ecomintent-api.p.rapidapi.com/classify/batch"
headers = {
    "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY",
    "X-RapidAPI-Host": "ecomintent-api.p.rapidapi.com",
    "Content-Type": "application/json",
}
payload = {
    "texts": [
        "where is my package",
        "I need to return this item",
        "you charged me twice",
        "can't log into my account",
    ]
}

response = requests.post(url, json=payload, headers=headers)
for r in response.json()["results"]:
    print(f"{r['intent']}: {r['confidence']:.3f}")
# WISMO: 0.984
# RETURN_REQUEST: 0.961
# BILLING_DISPUTE: 0.978
# ACCOUNT_ISSUE: 0.991
```

### curl

```bash
curl -X POST "https://ecomintent-api.p.rapidapi.com/classify" \
  -H "X-RapidAPI-Key: YOUR_RAPIDAPI_KEY" \
  -H "X-RapidAPI-Host: ecomintent-api.p.rapidapi.com" \
  -H "Content-Type: application/json" \
  -d '{"text": "my item arrived completely smashed"}'
```

---

## API Reference

### POST /classify

Classify a single support ticket.

**Request body:**
```json
{
  "text": "string (required, max 512 chars)",
  "threshold": 0.70
}
```

**Response:**
```json
{
  "intent": "WISMO",
  "confidence": 0.9872,
  "scores": {
    "WISMO": 0.9872,
    "RETURN_REQUEST": 0.0043,
    "EXCHANGE_REQUEST": 0.0021,
    ...
  },
  "below_threshold": false,
  "request_id": "a3f9b2c1",
  "latency_ms": 7.3
}
```

### POST /classify/batch

Classify up to 32 tickets in one request.

**Request body:**
```json
{
  "texts": ["text1", "text2", ...],
  "threshold": 0.70
}
```

### GET /intents

Returns all 9 intent labels with descriptions.

### GET /health

Returns API version, model status, and device info.

---

## FAQ

**Q: What languages are supported?**
A: English only in v1. Multilingual support is planned for v2.

**Q: What happens if confidence is below the threshold?**
A: The API returns `intent: "OTHER"` and sets `below_threshold: true`. The `scores` field still shows raw probabilities for all 9 classes if you want to apply your own threshold logic.

**Q: What's the maximum text length?**
A: 512 characters. Longer text is truncated at the model's 128-token boundary. Most support tickets are well under this limit.

**Q: Can I adjust the confidence threshold?**
A: Yes — pass `"threshold": 0.85` (or any value 0.0–1.0) in the request body. Default is 0.70. Higher thresholds reduce false positives at the cost of more OTHER classifications.

**Q: How does the batch endpoint work?**
A: POST up to 32 texts in one request. Results are returned in the same order as inputs. Batch is more efficient than 32 individual calls but does not share GPU compute in v1 (planned for v2).
