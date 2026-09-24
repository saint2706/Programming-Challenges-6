"""Receipt categorization by merchant keyword rules and fuzzy matching.

Combines keyword-based rules (fast, exact) with fuzzy merchant matching
(handles typos and variations) to categorize receipts into predefined
expense categories.
"""

from __future__ import annotations

from rapidfuzz import fuzz, process as fuzz_process


CATEGORIES = ["Groceries", "Dining", "Transport", "Utilities", "Shopping", "Entertainment", "Health", "Other"]

KEYWORD_RULES: dict[str, list[str]] = {
    "Groceries": [
        "safeway",
        "kroger",
        "whole foods",
        "trader joe",
        "costco",
        "walmart",
        "target",
        "grocery",
        "supermarket",
        "market",
        "instacart",
        "amazon fresh",
    ],
    "Dining": [
        "restaurant",
        "cafe",
        "coffee",
        "pizza",
        "burger",
        "taco",
        "sushi",
        "bar",
        "bistro",
        "diner",
        "fast food",
        "doordash",
        "grubhub",
        "uber eats",
        "postmates",
        "mcdonald",
        "subway",
        "chipotle",
        "starbucks",
        "dunkin",
        "domino",
    ],
    "Transport": [
        "uber",
        "lyft",
        "taxi",
        "gas",
        "shell",
        "chevron",
        "exxon",
        "bp",
        "mobility",
        "transit",
        "parking",
        "airline",
        "amtrak",
        "train",
    ],
    "Utilities": [
        "electric",
        "water",
        "gas company",
        "internet",
        "phone",
        "verizon",
        "at&t",
        "comcast",
        "spectrum",
        "utility",
    ],
    "Shopping": [
        "amazon",
        "ebay",
        "mall",
        "apparel",
        "clothing",
        "shoes",
        "fashion",
        "h&m",
        "gap",
        "zara",
        "uniqlo",
        "best buy",
    ],
    "Entertainment": [
        "netflix",
        "spotify",
        "hulu",
        "disney",
        "movie",
        "cinema",
        "theater",
        "concert",
        "live nation",
        "ticketmaster",
        "gaming",
        "steam",
        "playstation",
    ],
    "Health": [
        "pharmacy",
        "walgreens",
        "cvs",
        "doctor",
        "clinic",
        "hospital",
        "medical",
        "dental",
        "health",
        "wellness",
        "gym",
        "fitness",
    ],
}

MERCHANT_DATABASE = [
    "Safeway",
    "Kroger",
    "Whole Foods Market",
    "Trader Joe's",
    "Costco",
    "Walmart",
    "Target",
    "Instacart",
    "Amazon Fresh",
    "McDonald's",
    "Subway",
    "Chipotle",
    "Starbucks",
    "Dunkin",
    "Pizza Hut",
    "Domino's",
    "Uber",
    "Lyft",
    "Taxi Service",
    "Shell Gas Station",
    "Chevron",
    "Exxon Mobil",
    "BP",
    "Netflix",
    "Spotify",
    "Hulu",
    "Disney Plus",
    "Amazon Prime",
    "Apple Music",
    "Walgreens",
    "CVS Pharmacy",
    "Rite Aid",
    "Best Buy",
    "Home Depot",
    "Lowes",
    "IKEA",
    "Urban Outfitters",
    "Zara",
    "H&M",
    "Gap",
    "Forever 21",
    "ASOS",
    "Uniqlo",
    "GrubHub",
    "DoorDash",
    "Uber Eats",
    "American Airlines",
    "Delta Airlines",
    "United Airlines",
    "Southwest Airlines",
]


def categorize_by_keywords(merchant: str) -> str:
    """Categorize by exact keyword matching against the merchant name."""
    merchant_lower = merchant.lower()

    for category, keywords in KEYWORD_RULES.items():
        for keyword in keywords:
            if keyword in merchant_lower:
                return category

    return "Other"


def categorize_by_fuzzy_match(merchant: str, threshold: float = 70.0) -> str:
    """Categorize by fuzzy matching the merchant against a known merchant database.

    Uses rapidfuzz's ratio scorer (whole-string matching, not partial) to
    find the best match in the merchant database, then returns the category
    for that matched merchant.
    """
    if not merchant.strip():
        return "Other"

    matches = fuzz_process.extract(
        merchant,
        MERCHANT_DATABASE,
        scorer=fuzz.ratio,
        limit=1,
        score_cutoff=threshold,
    )

    if not matches:
        return "Other"

    matched_merchant = matches[0][0]
    return categorize_by_keywords(matched_merchant)


def categorize(merchant: str) -> str:
    """Categorize a receipt by merchant name.

    First tries keyword rules (fast), then falls back to fuzzy matching
    against known merchants, then returns "Other" if no match found.
    """
    if not merchant.strip():
        return "Other"

    keyword_category = categorize_by_keywords(merchant)
    if keyword_category != "Other":
        return keyword_category

    fuzzy_category = categorize_by_fuzzy_match(merchant)
    return fuzzy_category
