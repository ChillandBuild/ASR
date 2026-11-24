"""Explicit conservative cleanup, separate from raw scoring."""
import unicodedata


def cleanup(text: str) -> str:
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text.lower())
    return ' '.join(text.split())
