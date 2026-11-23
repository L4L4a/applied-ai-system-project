"""Explanation text (placeholder replaced by the specialized template in the next step)."""

from src.rag import RankedSong


def explain_baseline(r: RankedSong) -> str:
    return ", ".join(r.reasons) if r.reasons else "no scoring terms matched"


def explain_specialized(r: RankedSong, confidence: float) -> str:
    text = explain_baseline(r)
    return f"{text}. Note: {r.note}" if r.note else text
