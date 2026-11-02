"""Holding-level sector labels; no automatic ticker lookup or fund look-through."""
SECTORS = frozenset({
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials", "Information Technology",
    "Materials", "Real Estate", "Utilities",
})
SECTOR_LABELS = SECTORS | {"Multi-sector", "Not applicable", "Unknown"}
SECTOR_ELIGIBLE_CLASSES = {"US equity", "International equity", "Global equity", "Real estate"}
