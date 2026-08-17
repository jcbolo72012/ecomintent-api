#!/usr/bin/env python3
"""
EcomIntent API — FastAPI inference server.
Production-ready with rate limiting, confidence thresholding,
batch processing, and structured logging.
"""
import json
import time
import uuid
import logging
import os
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "training/best_model"))
LABEL_CONFIG_PATH = Path(os.environ.get("LABEL_CONFIG_PATH", "data/processed/label_config.json"))
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
MAX_TEXT_LENGTH = 512
MAX_BATCH_SIZE = 32
VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ecomintent")

with open(LABEL_CONFIG_PATH) as f:
    _cfg = json.load(f)

LABELS = _cfg["labels"]
ID_TO_LABEL = {int(k): v for k, v in _cfg["id_to_label"].items()}
LABEL_TO_ID = _cfg["label_to_id"]

INTENT_DESCRIPTIONS = {
    "WISMO": "Customer asking where their order is or requesting tracking information",
    "RETURN_REQUEST": "Customer wanting to return item(s) for a refund",
    "EXCHANGE_REQUEST": "Customer wanting to exchange for a different size, color, or variant",
    "CANCEL_ORDER": "Customer wanting to cancel an order before it ships",
    "DAMAGED_ITEM": "Customer received a broken, wrong, or damaged item",
    "BILLING_DISPUTE": "Customer has a billing issue, unexpected charge, or refund question",
    "PRODUCT_QUESTION": "Customer asking about product specs, availability, or compatibility",
    "ACCOUNT_ISSUE": "Customer has a login, password, or account access problem",
    "OTHER": "Message doesn't clearly fit a specific intent or is out of scope",
}

classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    logger.info(f"Loading model from {MODEL_PATH}...")
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model=str(MODEL_PATH),
        device=device,
        top_k=None,
        truncation=True,
        max_length=128,
    )
    logger.info(f"Model loaded. Device: {'GPU' if device == 0 else 'CPU'}")
    yield
    logger.info("Shutting down.")


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="EcomIntent API",
    description="E-commerce support ticket intent classification. Fine-tuned on Bitext retail data.",
    version=VERSION,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SETTINGS_DIR = Path("/settings/users")
LOGS_DIR = Path("/settings/logs")


def _user_id(request: Request) -> str:
    """RapidAPI injects X-RapidAPI-User on every proxied request.
    Fall back to a hashed IP for direct/dev calls. Never returns a raw IP."""
    ru = request.headers.get("X-RapidAPI-User")
    if ru:
        return f"rapid_{ru}"
    ip = request.client.host if request.client else "unknown"
    return f"direct_{hashlib.md5(ip.encode()).hexdigest()[:12]}"


def _settings_path(user_id: str) -> Path:
    safe = hashlib.md5(user_id.encode()).hexdigest()
    return SETTINGS_DIR / f"{safe}.json"


def _load_settings(user_id: str) -> dict:
    p = _settings_path(user_id)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"opt_in": False, "created_at": None, "updated_at": None}


def _save_settings(user_id: str, settings: dict):
    from datetime import datetime, timezone
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    settings["updated_at"] = now
    settings.setdefault("created_at", now)
    p = _settings_path(user_id)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(settings, f)
    tmp.replace(p)  # atomic, overwrites on both POSIX and Windows


def _is_opted_in(user_id: str) -> bool:
    return _load_settings(user_id).get("opt_in", False)


def _log_low_confidence(user_id: str, text: str, intent: str, confidence: float):
    """Store a HASHED low-confidence example. Only called when opted in.
    Raw text is never written — only its MD5 hash."""
    from datetime import datetime, timezone
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    safe = hashlib.md5(user_id.encode()).hexdigest()
    entry = {
        "text_hash": hashlib.md5(text.lower().strip().encode()).hexdigest(),
        "intent": intent,
        "confidence": round(confidence, 4),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(LOGS_DIR / f"{safe}.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Support ticket text to classify", min_length=1)
    threshold: float = Field(
        DEFAULT_CONFIDENCE_THRESHOLD,
        description="Minimum confidence to return a specific intent (0.0–1.0). Below threshold returns OTHER.",
        ge=0.0, le=1.0
    )

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v):
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text exceeds {MAX_TEXT_LENGTH} characters")
        return v.strip()


class ClassifyBatchRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to classify", min_length=1, max_length=MAX_BATCH_SIZE)
    threshold: float = Field(DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    scores: dict[str, float]
    below_threshold: bool
    request_id: str
    latency_ms: float


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]
    count: int
    request_id: str
    total_latency_ms: float


class SettingsBody(BaseModel):
    opt_in: bool = Field(..., description="True to opt into anonymised training data collection")


def run_inference(text: str, threshold: float) -> dict:
    result = classifier(text)[0]
    scores = {}
    for item in result:
        label = item["label"]
        if label.startswith("LABEL_"):
            label = ID_TO_LABEL[int(label.split("_")[1])]
        scores[label] = round(float(item["score"]), 4)

    top_intent = max(scores, key=scores.get)
    top_confidence = scores[top_intent]
    below_threshold = top_confidence < threshold

    return {
        "intent": "OTHER" if below_threshold else top_intent,
        "confidence": top_confidence,
        "scores": scores,
        "below_threshold": below_threshold,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "version": VERSION,
        "device": "gpu" if torch.cuda.is_available() else "cpu",
        "labels": LABELS,
    }


@app.get("/intents")
async def get_intents():
    return {
        "intents": LABELS,
        "count": len(LABELS),
        "descriptions": INTENT_DESCRIPTIONS,
    }


@app.get("/settings")
async def get_settings(request: Request):
    uid = _user_id(request)
    s = _load_settings(uid)
    return {"opt_in": s.get("opt_in", False),
            "message": "Opted in" if s.get("opt_in") else "Not opted in (default)"}


@app.post("/settings")
async def update_settings(request: Request, body: SettingsBody):
    uid = _user_id(request)
    s = _load_settings(uid)
    s["opt_in"] = body.opt_in
    _save_settings(uid, s)
    return {"opt_in": body.opt_in,
            "message": f"Successfully {'opted in' if body.opt_in else 'opted out'}. Effective immediately."}


@app.delete("/settings/data")
async def delete_training_data(request: Request):
    from datetime import datetime, timezone
    uid = _user_id(request)
    safe = hashlib.md5(uid.encode()).hexdigest()
    log_file = LOGS_DIR / f"{safe}.jsonl"
    n = 0
    if log_file.exists():
        with open(log_file) as f:
            n = sum(1 for _ in f)
        log_file.unlink()
    return {"deleted": True, "examples_deleted": n,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": f"Deleted {n} stored examples. Raw text was never stored."}


@app.post("/classify", response_model=ClassifyResponse)
@limiter.limit("100/second")
async def classify(request: Request, body: ClassifyRequest):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    result = run_inference(body.text, body.threshold)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Opt-in gated training-data capture. Default OFF — nothing hits disk unless
    # the user explicitly opted in. Only a HASH is stored, never raw text.
    if result["confidence"] < 0.75:
        uid = _user_id(request)
        if _is_opted_in(uid):
            try:
                _log_low_confidence(uid, body.text, result["intent"], result["confidence"])
            except Exception as e:
                logger.warning(f"opt-in log write failed: {e}")  # never affect the response

    return ClassifyResponse(**result, request_id=request_id, latency_ms=latency_ms)


@app.post("/classify/batch", response_model=BatchClassifyResponse)
@limiter.limit("20/second")
async def classify_batch(request: Request, body: ClassifyBatchRequest):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    uid = _user_id(request)
    should_log = _is_opted_in(uid)

    results = []
    for text in body.texts:
        text_clean = text[:MAX_TEXT_LENGTH].strip()
        r = run_inference(text_clean, body.threshold)
        if should_log and r["confidence"] < 0.75:
            try:
                _log_low_confidence(uid, text_clean, r["intent"], r["confidence"])
            except Exception as e:
                logger.warning(f"batch opt-in log write failed: {e}")
        results.append(ClassifyResponse(**r, request_id=request_id, latency_ms=0))

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    for r in results:
        r.latency_ms = round(total_ms / len(results), 2)

    return BatchClassifyResponse(
        results=results, count=len(results),
        request_id=request_id, total_latency_ms=total_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
