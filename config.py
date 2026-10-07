from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Base SLM used by the experiment.
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

LABELS = [
    "CANCEL_ITEM",
    "CANCEL_ORDER",
    "CHANGE_DELIVERY_ADDRESS",
    "EXCHANGE_ITEM",
    "PAYMENT_ISSUE",
    "REFUND_REQUEST",
    "REFUND_STATUS",
    "RETURN_ITEM",
    "RETURN_ORDER",
    "RETURN_POLICY",
    "SHIPPING_POLICY",
    "TRACK_ORDER",
]

DESCRIPTIONS = {
    "CANCEL_ITEM": "Cancel one item or line from an order while keeping the rest.",
    "CANCEL_ORDER": "Cancel the complete order before dispatch.",
    "CHANGE_DELIVERY_ADDRESS": "Change the delivery or shipping address of an existing order.",
    "EXCHANGE_ITEM": "Exchange a delivered item, for example for another size, colour or replacement.",
    "PAYMENT_ISSUE": "Report a payment, checkout, card, wallet or payment-processing problem.",
    "REFUND_REQUEST": "Request or initiate a new refund that has not already been approved or started.",
    "REFUND_STATUS": "Check progress of a refund that was already requested, approved or issued.",
    "RETURN_ITEM": "Return one item from an order while keeping other items.",
    "RETURN_ORDER": "Return the complete delivered order or all items in it.",
    "RETURN_POLICY": "Ask general return eligibility, deadlines, packaging rules or return fees.",
    "SHIPPING_POLICY": "Ask general shipping destinations, charges, options or delivery times.",
    "TRACK_ORDER": "Locate or check delivery progress of an existing order.",
}


def classification_instructions():
    lines = [
        "You are an intent classifier for the fictional DemoCart store.",
        "Classify the customer request into exactly one allowed intent label.",
        "Return only the label. Do not explain or answer the request.",
        "Pay special attention to whether the customer means one item or the whole order,",
        "and whether a refund is being newly requested or its existing status is being checked.",
        "",
        "Allowed intent labels:",
    ]

    for label in LABELS:
        lines.append(f"{label}: {DESCRIPTIONS[label]}")

    return "\n".join(lines)
