#!/usr/bin/env python3
"""
Generates synthetic training examples for each intent class using Claude API.
Targets 200-400 examples per intent to fill data gaps.
Total API cost: approximately $2-4 at Sonnet pricing.
"""
import os
import json
import time
import random
import argparse
from pathlib import Path
import anthropic
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SEED_EXAMPLES = {
    "WISMO": [
        "where is my order it's been 5 days",
        "can you check on my delivery please",
        "tracking number not updating what's going on",
        "my package was supposed to arrive yesterday",
        "when will my stuff get here",
    ],
    "RETURN_REQUEST": [
        "I need to return this item it doesn't fit",
        "how do I send this back for a refund",
        "want to return the shoes I bought last week",
        "can I get my money back this doesn't work",
        "I'd like to initiate a return please",
    ],
    "EXCHANGE_REQUEST": [
        "can I get a size large instead of medium",
        "I ordered the blue one but want the black",
        "wrong size, can you send a different one",
        "I need to swap this for a different variant",
        "can I exchange this for the same thing in XL",
    ],
    "CANCEL_ORDER": [
        "please cancel my order I just placed",
        "I need to cancel order #12345 immediately",
        "can you cancel before it ships",
        "I changed my mind please cancel",
        "cancel my order from this morning",
    ],
    "DAMAGED_ITEM": [
        "my item arrived completely broken",
        "the box was damaged and the product inside too",
        "you sent me the wrong item",
        "this arrived cracked and unusable",
        "package was destroyed when it arrived",
    ],
    "BILLING_DISPUTE": [
        "I was charged twice for the same order",
        "there's an unauthorized charge on my account",
        "my refund hasn't shown up yet",
        "I see a charge I don't recognize",
        "why was I billed twice",
    ],
    "PRODUCT_QUESTION": [
        "does this work with iPhone 15",
        "what are the dimensions of this product",
        "is this available in red",
        "does this come with a warranty",
        "what's the weight limit on this",
    ],
    "ACCOUNT_ISSUE": [
        "I can't log into my account",
        "forgot my password how do I reset it",
        "my email changed and I can't access my account",
        "I'm locked out please help",
        "can't remember which email I used to sign up",
    ],
    "OTHER": [
        "hi there",
        "you guys are amazing thank you",
        "I have a general question",
        "not sure who to contact about this",
        "just wanted to leave feedback",
    ],
}

INTENT_DEFINITIONS = {
    "WISMO": "Customer asking where their order is, wanting tracking information, asking about delivery status or estimated arrival time",
    "RETURN_REQUEST": "Customer wanting to return one or more items to get a full or partial refund, not an exchange",
    "EXCHANGE_REQUEST": "Customer wanting to swap an item for a different size, color, or variant — not a refund, they want a replacement",
    "CANCEL_ORDER": "Customer wanting to cancel an order that has not yet shipped. Must mention cancellation explicitly",
    "DAMAGED_ITEM": "Customer received a broken, damaged, defective, or incorrect item. Item arrived in bad condition or wrong item sent",
    "BILLING_DISPUTE": "Customer has a problem with a charge: double billing, unauthorized charge, missing refund, payment failure",
    "PRODUCT_QUESTION": "Customer asking about product specifications, availability, sizing, compatibility, materials, or other product details",
    "ACCOUNT_ISSUE": "Customer cannot log in, needs password reset, account locked, or has any account access problem",
    "OTHER": "Message that doesn't clearly fit any of the above: greetings, general feedback, vague questions, spam, or messages that could be multiple intents",
}

REGISTER_VARIATIONS = [
    "Include some with typos and spelling mistakes",
    "Include some very short messages (under 10 words)",
    "Include some very frustrated/emotional messages in CAPS",
    "Include some very polite, formal messages",
    "Include some with order numbers or reference codes",
    "Include some with emojis",
    "Include some with missing punctuation",
    "Include some run-on sentences without proper spacing",
]

TARGET_PER_INTENT = {
    "WISMO": 1500,
    "RETURN_REQUEST": 1500,
    "EXCHANGE_REQUEST": 2000,  # hardest class — needs most examples
    "CANCEL_ORDER": 1200,
    "DAMAGED_ITEM": 1500,
    "BILLING_DISPUTE": 1500,
    "PRODUCT_QUESTION": 1800,  # broad class — needs diversity
    "ACCOUNT_ISSUE": 1200,
    "OTHER": 2000,             # needs OOS diversity
}


def generate_for_intent(intent: str, count: int = 200) -> list[str]:
    variations = random.sample(REGISTER_VARIATIONS, 3)
    variation_str = "\n".join(f"- {v}" for v in variations)

    prompt = f"""Generate {count} realistic customer support ticket messages for an e-commerce store.

Intent: {intent}
Definition: {INTENT_DEFINITIONS[intent]}

Requirements:
- Every message MUST clearly express the intent: {intent}
- Vary writing style dramatically across the {count} examples
- {variation_str}
- Do NOT include the intent label or any classification in the message
- Do NOT number the messages
- Each message on its own line
- Mix message lengths: some 5 words, some 50 words
- Make them sound like real customers, not AI-generated text

Seed examples for reference (generate NEW ones, don't repeat these):
{chr(10).join(f'- {ex}' for ex in SEED_EXAMPLES[intent])}

Generate exactly {count} messages now, one per line:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) >= 5]
    clean = [l for l in lines if not l[0].isdigit() and not l.startswith("-")]
    return clean[:count]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", type=str, default=None, help="Generate for a single intent only")
    parser.add_argument("--count", type=int, default=None, help="Override count for the specified intent")
    args = parser.parse_args()

    SYNTH_DIR = Path("data/synthetic")
    SYNTH_DIR.mkdir(exist_ok=True)

    intents_to_run = (
        {args.intent: args.count or TARGET_PER_INTENT.get(args.intent, 200)}
        if args.intent
        else TARGET_PER_INTENT
    )

    all_synthetic = []
    for intent, count in intents_to_run.items():
        console.print(f"\n[cyan]Generating {count} examples for {intent}...[/cyan]")
        examples = generate_for_intent(intent, count)
        console.print(f"  ✓ Got {len(examples)} examples")

        for text in examples:
            all_synthetic.append({"text": text, "label": intent, "source": "synthetic_claude"})

        pd.DataFrame([{"text": t, "label": intent} for t in examples]).to_parquet(
            SYNTH_DIR / f"synthetic_{intent.lower()}.parquet"
        )
        time.sleep(1)

    df_synth = pd.DataFrame(all_synthetic)
    out_path = (
        SYNTH_DIR / "synthetic_all.parquet"
        if not args.intent
        else SYNTH_DIR / f"synthetic_{args.intent.lower()}_extra.parquet"
    )
    df_synth.to_parquet(out_path)

    console.print(f"\n[green]Synthetic generation complete: {len(df_synth):,} total examples[/green]")
    print(df_synth["label"].value_counts().to_string())
