"""Document-level categories, with optional explicit overrides at ingestion."""

CATEGORIES = ("investing", "taxes", "trading", "insurance", "personal_finance")


def infer_category(source: str) -> str:
    """Use transparent filename rules; these labels are not content classification."""
    name = source.casefold()
    if "irs" in name or "tax" in name:
        return "taxes"
    if "insurance" in name:
        return "insurance"
    if any(word in name for word in ("personal", "cfpb", "literacy")):
        return "personal_finance"
    if any(word in name for word in (
        "technical analysis", "fundamental analysis", "futures", "option",
        "trading", "currency", "stock markets",
    )):
        return "trading"
    return "investing"
