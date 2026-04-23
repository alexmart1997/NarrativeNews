from __future__ import annotations


def estimate_word_count(text: str) -> int:
    return len([token for token in text.split() if token.strip()])
