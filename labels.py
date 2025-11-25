"""
Label definitions for PII NER task.
"""

# All entity types we detect
LABELS = [
    "O",
    "B-PHONE", "I-PHONE",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-EMAIL", "I-EMAIL",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-DATE", "I-DATE",
    "B-CITY", "I-CITY",
    "B-LOCATION", "I-LOCATION",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}

# PII categories (must be redacted)
PII_LABELS = {"PHONE", "CREDIT_CARD", "EMAIL", "PERSON_NAME", "DATE"}

def label_is_pii(label: str) -> bool:
    """Check if a label is PII (needs redaction)."""
    # Remove B-/I- prefix
    if "-" in label:
        label = label.split("-", 1)[1]
    return label in PII_LABELS