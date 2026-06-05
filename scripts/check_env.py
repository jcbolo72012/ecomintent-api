#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv
load_dotenv()

required = {
    "ANTHROPIC_API_KEY": "Anthropic API for synthetic data generation",
    "HF_TOKEN": "HuggingFace token for pushing model",
    "MODAL_TOKEN_ID": "Modal.com token ID for deployment",
    "MODAL_TOKEN_SECRET": "Modal.com token secret for deployment",
    "HF_USERNAME": "Your HuggingFace username (for model repo)",
}

missing = []
for var, desc in required.items():
    val = os.getenv(var)
    if not val:
        missing.append(f"  MISS {var}: {desc}")
        print(f"MISSING: {var} — {desc}")
    else:
        masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "****"
        print(f"  OK {var}: {masked}")

if missing:
    print(f"\n{len(missing)} required variables missing. See 04_ENV_TEMPLATE.md")
    sys.exit(1)
else:
    print("\nAll environment variables present.")
