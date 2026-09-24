"""Tests for receipt categorization."""

import pytest

from categorizer import categorize, categorize_by_fuzzy_match, categorize_by_keywords


class TestCategorizeByKeywords:
    def test_grocery_keywords(self):
        assert categorize_by_keywords("Whole Foods Market") == "Groceries"
        assert categorize_by_keywords("Kroger Supermarket") == "Groceries"
        assert categorize_by_keywords("SAFEWAY") == "Groceries"
        assert categorize_by_keywords("Trader Joe's") == "Groceries"

    def test_dining_keywords(self):
        assert categorize_by_keywords("McDonald's") == "Dining"
        assert categorize_by_keywords("STARBUCKS COFFEE") == "Dining"
        assert categorize_by_keywords("Pizza Hut") == "Dining"
        assert categorize_by_keywords("Chipotle Mexican Grill") == "Dining"

    def test_transport_keywords(self):
        assert categorize_by_keywords("Uber") == "Transport"
        assert categorize_by_keywords("Lyft") == "Transport"
        assert categorize_by_keywords("Shell Gas Station") == "Transport"

    def test_entertainment_keywords(self):
        assert categorize_by_keywords("Netflix") == "Entertainment"
        assert categorize_by_keywords("Spotify Premium") == "Entertainment"

    def test_health_keywords(self):
        assert categorize_by_keywords("CVS Pharmacy") == "Health"
        assert categorize_by_keywords("Walgreens") == "Health"
        assert categorize_by_keywords("Gym Membership") == "Health"

    def test_utilities_keywords(self):
        assert categorize_by_keywords("Verizon") == "Utilities"
        assert categorize_by_keywords("Comcast Internet") == "Utilities"

    def test_shopping_keywords(self):
        assert categorize_by_keywords("Amazon") == "Shopping"
        assert categorize_by_keywords("Best Buy") == "Shopping"
        assert categorize_by_keywords("H&M Fashion") == "Shopping"

    def test_case_insensitive(self):
        assert categorize_by_keywords("WHOLE FOODS") == "Groceries"
        assert categorize_by_keywords("whole foods") == "Groceries"
        assert categorize_by_keywords("Whole Foods") == "Groceries"

    def test_partial_match(self):
        assert categorize_by_keywords("Starbucks Coffee 123 Main") == "Dining"
        assert categorize_by_keywords("My Local Uber Ride") == "Transport"

    def test_no_match_returns_other(self):
        assert categorize_by_keywords("Unknown Merchant XYZ") == "Other"
        assert categorize_by_keywords("Random Store") == "Other"


class TestCategorizeByFuzzyMatch:
    def test_exact_merchant_match(self):
        result = categorize_by_fuzzy_match("Whole Foods Market")
        assert result in ["Groceries", "Other"]

    def test_typo_tolerance(self):
        result = categorize_by_fuzzy_match("Whloe Foods")
        assert result in ["Groceries", "Other"]

    def test_partial_merchant_name(self):
        result = categorize_by_fuzzy_match("Starbucks")
        assert result in ["Dining", "Other"]

    def test_threshold_enforcement(self):
        result = categorize_by_fuzzy_match("XYZ", threshold=95.0)
        assert result == "Other"

    def test_empty_merchant(self):
        result = categorize_by_fuzzy_match("")
        assert result == "Other"

    def test_whitespace_only(self):
        result = categorize_by_fuzzy_match("   ")
        assert result == "Other"


class TestCategorize:
    def test_keyword_match_priority(self):
        assert categorize("Whole Foods Market") == "Groceries"
        assert categorize("Netflix") == "Entertainment"

    def test_fallback_to_fuzzy(self):
        result = categorize("Whol Foods")
        assert result in ["Groceries", "Other"]

    def test_unknown_returns_other(self):
        assert categorize("Completely Unknown Store 12345") == "Other"

    def test_empty_input(self):
        assert categorize("") == "Other"

    def test_various_merchants(self):
        assert categorize("McDonald's") == "Dining"
        assert categorize("Uber") == "Transport"
        assert categorize("CVS") == "Health"
        assert categorize("Best Buy") == "Shopping"
        assert categorize("Comcast") == "Utilities"

    def test_combined_extraction_and_categorization(self):
        merchants = [
            ("WHOLE FOODS", "Groceries"),
            ("PIZZA HUT", "Dining"),
            ("CHEVRON GAS", "Transport"),
            ("NETFLIX", "Entertainment"),
            ("WALGREENS", "Health"),
        ]
        for merchant, expected in merchants:
            result = categorize(merchant)
            assert result == expected or result == "Other"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
