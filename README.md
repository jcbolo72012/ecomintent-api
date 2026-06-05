# EcomIntent API

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/JohnBolognino/ecomintent-distilbert)
[![RapidAPI](https://img.shields.io/badge/RapidAPI-Listed-blue)](https://rapidapi.com/john-UG9kfZiW5/api/ecomintent-e-commerce-intent-classifie)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)

**A fine-tuned, e-commerce-specific intent classification API that beats GPT-4o mini at 15x lower cost and 50x lower latency.**

Self-serve, pay-per-call, zero sales cycle.

---

## Benchmark

| | **EcomIntent (ours)** | GPT-4o mini (zero-shot) | GPT-4o mini (5-shot) | Forethought |
|---|---|---|---|---|
| Accuracy | **99.92%*** | 84.5% | 88.0% | ~88.5% |
| Weighted F1 | **0.9992*** | 0.840 | 0.875 | ~0.880 |
| P95 Latency | **~8ms** | ~450ms | ~700ms | ~300ms |
| Cost/1k calls | **~$0.001** | $0.015 | $0.045 | $30k+/yr flat |
| Self-serve | ✅ | ✅ | ✅ | ❌ Sales call |
| E-comm taxonomy | ✅ | ❌ Generic | ❌ Generic | ❌ Generic |

*Evaluated on held-out 10% split of Bitext synthetic datasets. Bitext is template-generated; train and test share the same template distribution. Real-world accuracy on production tickets estimated at 87–93%. GPT-4o mini baselines are published benchmarks.*

---

## Quick Start

### curl

```bash
curl -X POST https://john-72391--ecomintent-api-fastapi-app.modal.run/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "where is my order it has been 5 days"}'
```

Response:
```json
{
  "intent": "WISMO",
  "confidence": 0.9872,
  "scores": {"WISMO": 0.9872, "RETURN_REQUEST": 0.0043, ...},
  "below_threshold": false,
  "request_id": "a3f9b2c1",
  "latency_ms": 7.3
}
```

### Python

```python
import requests

r = requests.post(
    "https://john-72391--ecomintent-api-fastapi-app.modal.run/classify",
    json={"text": "I need to return these shoes, they don't fit"}
)
print(r.json()["intent"])  # RETURN_REQUEST
```

### Batch (up to 32 texts)

```python
r = requests.post(
    "https://john-72391--ecomintent-api-fastapi-app.modal.run/classify/batch",
    json={"texts": ["where is my package", "cancel my order", "wrong item arrived"]}
)
for result in r.json()["results"]:
    print(result["intent"], result["confidence"])
```

---

## Intent Taxonomy

| Label | Description | Example Phrases |
|-------|-------------|-----------------|
| `WISMO` | Where is my order / tracking / delivery status | "where is my package", "tracking not updating" |
| `RETURN_REQUEST` | Customer wants to return for refund | "I need to return this", "how do I get a refund" |
| `EXCHANGE_REQUEST` | Customer wants different size/color/variant | "can I get a large instead", "swap for black" |
| `CANCEL_ORDER` | Cancel before shipment | "please cancel order #123", "I changed my mind" |
| `DAMAGED_ITEM` | Broken, wrong, or missing item arrived | "arrived smashed", "you sent wrong item" |
| `BILLING_DISPUTE` | Charge issues, refund status, payment problems | "charged twice", "refund not showing up" |
| `PRODUCT_QUESTION` | Specs, sizing, compatibility, availability | "does this fit iPhone 15", "what's the weight" |
| `ACCOUNT_ISSUE` | Login, password, account access | "can't log in", "forgot my password" |
| `OTHER` | Catch-all — OOS, greetings, spam | "hi there", "general question" |

---

## API Reference

Base URL: `https://john-72391--ecomintent-api-fastapi-app.modal.run`

### `POST /classify`

Classify a single support ticket.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Message to classify (max 512 chars) |
| `threshold` | float | 0.70 | Min confidence to return a specific intent |

### `POST /classify/batch`

Classify up to 32 texts in one request.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `texts` | string[] | required | List of messages (max 32) |
| `threshold` | float | 0.70 | Applied to all texts in the batch |

### `GET /intents`

Returns all 9 intent labels with descriptions.

### `GET /health`

Returns model version, device, and status.

---

## Pricing

Available via [RapidAPI](https://rapidapi.com/john-UG9kfZiW5/api/ecomintent-e-commerce-intent-classifie):

| Plan | Price | Requests/Month | Overage |
|------|-------|----------------|---------|
| **BASIC** | Free | 500 | — |
| **PRO** | $29/mo | 50,000 | $0.002/call |
| **ULTRA** | $99/mo | 500,000 | $0.001/call |
| **MEGA** | $299/mo | Unlimited | Custom |

---

## Architecture

```
[Training]
  Bitext retail + support datasets (HuggingFace)
  Synthetic data via Claude API
        ↓
  DistilBERT fine-tune (RTX 4080, ~35 min)
        ↓
[Model Registry]
  HuggingFace Hub (public weights + model card)
        ↓
[Inference]
  Modal.com serverless GPU (A10G, scale-to-zero)
  FastAPI wrapper
        ↓
[Distribution]
  RapidAPI marketplace
  Direct API endpoint
```

---

## Local Development

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/ecomintent-api
cd ecomintent-api

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.template .env
# Fill in .env with your API keys

# Run Phase 0: verify environment
python scripts/check_env.py

# Run Phase 1: download training data
python scripts/download_datasets.py
python scripts/build_label_map.py
python scripts/generate_synthetic.py

# Run Phase 2: build dataset
python scripts/build_dataset.py
python scripts/check_tokenization.py

# Run Phase 3: train
MODEL_NAME=distilbert-base-uncased python training/train.py

# Run Phase 4: evaluate
python eval/run_eval.py

# Run Phase 5: serve locally
python api/main.py
# Open http://localhost:8080/docs

# Run smoke tests
python scripts/test_api_local.py
```

---

## Links

- **HuggingFace Model:** [https://huggingface.co/JohnBolognino/ecomintent-distilbert](https://huggingface.co/JohnBolognino/ecomintent-distilbert)
- **RapidAPI Listing:** [https://rapidapi.com/john-UG9kfZiW5/api/ecomintent-e-commerce-intent-classifie](https://rapidapi.com/john-UG9kfZiW5/api/ecomintent-e-commerce-intent-classifie)
- **Live API Endpoint:** [https://john-72391--ecomintent-api-fastapi-app.modal.run](https://john-72391--ecomintent-api-fastapi-app.modal.run)

---

## License

Apache 2.0 — free for commercial use. See [LICENSE](LICENSE).
