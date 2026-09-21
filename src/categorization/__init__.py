"""Transaction categorization module."""

from src.categorization.categorizer import categorize_transactions, rule_based_fallback

__all__ = ["categorize_transactions", "rule_based_fallback"]
