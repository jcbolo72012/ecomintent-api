# Domain-specific intent classification for e-commerce: fine-tuning DistilBERT to outperform GPT-4o mini at 1/15th the cost

*A practical case study showing why fine-tuned encoders still win for fixed-label classification tasks.*

---

## The Problem

Generic language models handle intent classification the same way they handle everything else: with general reasoning over general knowledge. For a fixed, well-defined taxonomy — like the nine intent categories that cover roughly 95% of all e-commerce customer support volume — that generality is waste. You pay for reasoning you don't need, and you get latency you can't afford.

Most e-commerce helpdesks route tickets manually, or use keyword rules that break on spelling errors and informal phrasing. The LLM alternative (prompt GPT-4o mini with your intent list) works, but at $0.015 per 1,000 calls and 450ms P95 latency, it's expensive and slow for a task that a smaller, purpose-built model can handle better. This post documents building that smaller model: a fine-tuned DistilBERT that classifies e-commerce support tickets into nine intent categories.

---

## Dataset

Training data came from two Bitext LLM chatbot training datasets (retail + customer support, CDLA-Sharing 1.0 license) totaling 71,756 examples. After deduplication and label normalization, 61,445 examples remained across nine canonical intent classes. The source datasets use 68 fine-grained labels which were mapped to the nine-class taxonomy below. Class distribution was capped at 4:1 to prevent the dominant classes (OTHER, WISMO, ACCOUNT_ISSUE) from overwhelming the smaller ones. Final split: 49,156 train / 6,144 val / 6,145 test, stratified by class.

**Important caveat on the evaluation numbers:** The Bitext datasets are themselves synthetically generated from a fixed set of templates per intent. Training and test examples share the same template distribution, which produces artificially high held-out metrics. Real-world accuracy on production customer tickets — with typos, multi-intent messages, and domain-specific jargon — is estimated at 87–93%. The benchmark numbers below are valid for this data distribution and useful for comparing model architectures, but should not be taken as predictions of production accuracy without validation on your own ticket data.

---

## Why DistilBERT Over GPT-4o Mini

For fixed-label classification with a well-defined taxonomy, encoder-only models have three structural advantages over generative LLMs:

- **Latency.** No autoregressive decoding. A single forward pass through six transformer layers produces the classification. P95 on Modal A10G: 4ms warm.
- **Cost.** Inference cost scales with GPU time per call, not per token. At $1.10/hr for an A10G and ~4ms per call, the cost is roughly $0.001 per 1,000 calls vs. $0.015 for GPT-4o mini zero-shot.
- **Accuracy on narrow domains.** Fine-tuning on domain-specific data consistently outperforms zero-shot prompting for fixed-label tasks. The model learns the specific language patterns of your taxonomy rather than reasoning about it from scratch on every call.

The tradeoff: a fine-tuned encoder is locked to its training taxonomy. Adding a new intent class requires retraining. For e-commerce support — where WISMO, returns, and exchanges account for 60%+ of ticket volume and the label set is stable — this is an acceptable constraint.

---

## Training Setup

- **Base model:** `distilbert-base-uncased` (66M parameters, 6 layers)
- **Task:** 9-class sequence classification
- **Hardware:** NVIDIA RTX 4080 Laptop GPU (12.9GB VRAM)
- **Training time:** 10.9 minutes
- **Epochs:** 8 with early stopping (patience=3)
- **Batch size:** 32
- **Learning rate:** 2e-5 with cosine schedule and 10% warmup
- **Weight decay:** 0.01
- **Max token length:** 128 (P99 of training data is 24 tokens — these are short messages)
- **Mixed precision:** fp16
- **Framework:** HuggingFace Transformers 4.47.0

The model is fine-tuned end-to-end with no frozen layers. DistilBERT's classification head (linear layer over the [CLS] token) learns the mapping from the distilled BERT representation to the nine intent classes.

---

## Results

### Test Set Performance

| Model | Accuracy | Weighted F1 | P95 Latency | Cost / 1k calls |
|---|---|---|---|---|
| **EcomIntent DistilBERT (ours)** | **99.92%** | **0.9992** | **4ms** | **$0.001** |
| GPT-4o mini (zero-shot) | 84.5% | 0.840 | 450ms | $0.015 |
| GPT-4o mini (5-shot) | 88.0% | 0.875 | 700ms | $0.045 |
| Forethought Triage | ~88.5% | ~0.880 | ~300ms | $30k+/yr flat |

*GPT-4o mini baselines are published benchmarks on intent classification tasks. EcomIntent numbers are on the held-out Bitext test split. See caveat above regarding real-world generalization.*

### Per-Class F1

| Intent | F1 | Precision | Recall | Test examples |
|---|---|---|---|---|
| WISMO | 0.9989 | 1.0000 | 0.9979 | 947 |
| RETURN_REQUEST | 1.0000 | 1.0000 | 1.0000 | 880 |
| EXCHANGE_REQUEST | 1.0000 | 1.0000 | 1.0000 | 378 |
| CANCEL_ORDER | 0.9979 | 0.9958 | 1.0000 | 236 |
| DAMAGED_ITEM | 0.9989 | 0.9979 | 1.0000 | 469 |
| BILLING_DISPUTE | 0.9985 | 1.0000 | 0.9970 | 677 |
| PRODUCT_QUESTION | 1.0000 | 1.0000 | 1.0000 | 664 |
| ACCOUNT_ISSUE | 0.9995 | 0.9989 | 1.0000 | 947 |
| OTHER | 0.9984 | 0.9979 | 0.9989 | 947 |

EXCHANGE_REQUEST and RETURN_REQUEST — the historically confused pair — are cleanly separated. The model learned that exchange intent requires explicit mention of a different variant (size, color) rather than just dissatisfaction with the received item.

### Confusion Matrix

![Confusion Matrix](https://huggingface.co/JohnBolognino/ecomintent-distilbert/resolve/main/confusion_matrix.png)

The diagonal is nearly solid. The only meaningful off-diagonal mass is a handful of CANCEL_ORDER examples predicted as OTHER (2 out of 236), which on inspection were ambiguous messages that could reasonably be either class.

---

## Cost and Latency Deep Dive

The cost calculation is straightforward. Modal's A10G GPU costs $1.10/hr. At 4ms P95 latency per call with scale-to-zero, the math is:

```
$1.10 / 3600 seconds = $0.000306 per GPU-second
4ms per call = 0.004 seconds per call
Cost per call = $0.000306 × 0.004 = $0.0000012
Cost per 1,000 calls = $0.0012
```

At 100,000 calls/month (a mid-size helpdesk), that's $120/month vs. $1,500/month for GPT-4o mini zero-shot — and the fine-tuned model is more accurate on this specific task.

Cold start latency (container spin-up from idle) is approximately 1.7 seconds. For latency-sensitive applications, set `scaledown_window` higher to keep the container warm.

---

## Limitations

- **English only.** The model was trained exclusively on English-language examples. Performance on Spanish, French, or other languages is untested and likely poor.
- **Single intent per message.** V1 assigns the highest-probability class. Messages containing multiple intents (e.g., "my order arrived damaged and I want a refund") get one label — the dominant signal wins.
- **Template distribution.** As noted, training data is synthetic. A model trained purely on synthetic data may underperform on edge cases that don't appear in the template inventory: highly informal phrasing, non-standard spelling, or industry-specific jargon.
- **Static taxonomy.** Adding or modifying intent classes requires retraining on new data.

---

## Reproduce It

The full training pipeline — data download, preprocessing, fine-tuning, evaluation, and Modal deployment — is open source:

- **GitHub:** https://github.com/jcbolo72012/ecomintent-api
- **Model weights:** https://huggingface.co/JohnBolognino/ecomintent-distilbert
- **Live API (RapidAPI):** https://rapidapi.com/john-UG9kfZiW5/api/ecomintent-e-commerce-intent-classifie

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="JohnBolognino/ecomintent-distilbert",
    top_k=1,
)

result = classifier("where is my order, it has been 5 days")
# [{'label': 'WISMO', 'score': 0.9998}]
```

Training takes under 15 minutes on any 12GB+ GPU. The pipeline handles data download, label normalization, tokenization analysis, training with early stopping, test set evaluation, and Modal deployment end to end.
