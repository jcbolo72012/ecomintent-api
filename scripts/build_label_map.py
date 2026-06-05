#!/usr/bin/env python3
"""
Maps Bitext labels to the canonical 9-class EcomIntent taxonomy.
Every label from both Bitext datasets must be handled.
Labels that don't fit any class map to OTHER.
"""
import json
import pandas as pd
from pathlib import Path

LABELS = [
    "WISMO",
    "RETURN_REQUEST",
    "EXCHANGE_REQUEST",
    "CANCEL_ORDER",
    "DAMAGED_ITEM",
    "BILLING_DISPUTE",
    "PRODUCT_QUESTION",
    "ACCOUNT_ISSUE",
    "OTHER",
]

LABEL_ID = {label: idx for idx, label in enumerate(LABELS)}

LABEL_MAP = {
    # WISMO
    "track_order": "WISMO",
    "track_refund": "WISMO",
    "delivery_period": "WISMO",
    "delivery_options": "WISMO",
    "set_up_shipping_address": "WISMO",
    "shipping_issue": "WISMO",

    # RETURN_REQUEST
    "get_refund": "RETURN_REQUEST",
    "return_merchandise": "RETURN_REQUEST",
    "check_refund_policy": "RETURN_REQUEST",
    "refund_not_showing_up": "RETURN_REQUEST",

    # EXCHANGE_REQUEST
    "exchange": "EXCHANGE_REQUEST",
    "change_order": "EXCHANGE_REQUEST",

    # CANCEL_ORDER
    "cancel_order": "CANCEL_ORDER",
    "check_cancellation_fee": "CANCEL_ORDER",

    # DAMAGED_ITEM
    "damaged_or_defective_item": "DAMAGED_ITEM",
    "wrong_item": "DAMAGED_ITEM",
    "missing_item": "DAMAGED_ITEM",
    "complaint": "DAMAGED_ITEM",

    # BILLING_DISPUTE
    "check_invoice": "BILLING_DISPUTE",
    "get_invoice": "BILLING_DISPUTE",
    "check_payment_methods": "BILLING_DISPUTE",
    "payment_issue": "BILLING_DISPUTE",

    # PRODUCT_QUESTION
    "product_info": "PRODUCT_QUESTION",
    "product_question": "PRODUCT_QUESTION",
    "availability_in_store": "PRODUCT_QUESTION",
    "add_product": "PRODUCT_QUESTION",
    "check_out": "PRODUCT_QUESTION",

    # ACCOUNT_ISSUE
    "create_account": "ACCOUNT_ISSUE",
    "delete_account": "ACCOUNT_ISSUE",
    "edit_account": "ACCOUNT_ISSUE",
    "switch_account": "ACCOUNT_ISSUE",
    "recover_password": "ACCOUNT_ISSUE",
    "registration_problems": "ACCOUNT_ISSUE",
    "login": "ACCOUNT_ISSUE",

    # OTHER
    "contact_customer_service": "OTHER",
    "review": "OTHER",
    "pay": "OTHER",
    "place_order": "OTHER",
    "newsletter_subscription": "OTHER",
    "use_app": "OTHER",

    # --- Extended mappings from Bitext retail & support datasets ---

    # WISMO (additional)
    "track_delivery": "WISMO",
    "delivery_time": "WISMO",
    "delivery_issue": "WISMO",
    "change_shipping_address": "WISMO",
    "shipping_costs": "WISMO",

    # RETURN_REQUEST (additional)
    "return_product": "RETURN_REQUEST",
    "return_product_online": "RETURN_REQUEST",
    "return_product_in_store": "RETURN_REQUEST",
    "return_policy": "RETURN_REQUEST",
    "request_refund": "RETURN_REQUEST",
    "refund_policy": "RETURN_REQUEST",
    "refund_status": "RETURN_REQUEST",

    # EXCHANGE_REQUEST (additional)
    "exchange_product": "EXCHANGE_REQUEST",
    "exchange_product_in_store": "EXCHANGE_REQUEST",

    # DAMAGED_ITEM (additional)
    "damaged_delivery": "DAMAGED_ITEM",
    "product_issue": "DAMAGED_ITEM",

    # BILLING_DISPUTE (additional)
    "payment_methods": "BILLING_DISPUTE",
    "request_invoice": "BILLING_DISPUTE",

    # PRODUCT_QUESTION (additional)
    "product_information": "PRODUCT_QUESTION",
    "availability": "PRODUCT_QUESTION",
    "availability_online": "PRODUCT_QUESTION",
    "sales_period": "PRODUCT_QUESTION",
    "remove_product": "PRODUCT_QUESTION",

    # ACCOUNT_ISSUE (additional)
    "change_account": "ACCOUNT_ISSUE",
    "close_account": "ACCOUNT_ISSUE",
    "open_account": "ACCOUNT_ISSUE",
    "technical_issue": "ACCOUNT_ISSUE",

    # OTHER (additional)
    "contact_human_agent": "OTHER",
    "customer_service": "OTHER",
    "human_agent": "OTHER",
    "order_history": "OTHER",
    "store_location": "OTHER",
    "store_opening_hours": "OTHER",
    "submit_feedback": "OTHER",
    "submit_product_feedback": "OTHER",
    "submit_product_idea": "OTHER",
    "request_right_to_rectification": "OTHER",
}


def map_label(source_label: str) -> str:
    """Map a source label to canonical taxonomy. Default to OTHER."""
    return LABEL_MAP.get(source_label.lower().strip(), "OTHER")


if __name__ == "__main__":
    DATA_RAW = Path("data/raw")
    DATA_PROC = Path("data/processed")
    DATA_PROC.mkdir(exist_ok=True)

    # Scan raw parquet files if they exist; otherwise label_config still valid
    all_source_labels = set()
    for parquet_file in DATA_RAW.glob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        if "label" in df.columns:
            all_source_labels.update(df["label"].unique())

    unmapped = [l for l in all_source_labels if l.lower().strip() not in LABEL_MAP]
    if unmapped:
        print(f"\nWARNING: {len(unmapped)} unmapped source labels -> mapped to OTHER:")
        for l in sorted(unmapped):
            print(f"  - {l}")

    config = {
        "labels": LABELS,
        "label_to_id": LABEL_ID,
        "id_to_label": {str(v): k for k, v in LABEL_ID.items()},
        "source_to_canonical": LABEL_MAP,
        "unmapped_labels": unmapped,
        "num_labels": len(LABELS),
    }
    with open(DATA_PROC / "label_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nLabel config saved: {len(LABELS)} canonical labels.")
    if all_source_labels:
        print(f"Scanned {len(all_source_labels)} source labels from raw parquet files.")
    else:
        print("No raw parquet files found — label config written from hardcoded map (synthetic-only mode).")
    if len(unmapped) > 10:
        print("NEEDS_REVIEW: More than 10 unmapped labels detected.")
