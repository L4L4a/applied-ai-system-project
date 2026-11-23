"""
Input validation / guardrails. `validate_request` either returns a cleaned dict or raises
ValidationError with a message meant for the end user.

Rules:
  * input must be a dict or a string, and contain at least one of text/genre/mood/energy
  * genre and mood must be known (notes + catalog); unknown values get 'did you mean' hints
  * energy must be a real number in [0, 1] (no bools, NaN, strings)
  * free text is trimmed, stripped of control characters and capped at MAX_TEXT_CHARS
  * unknown keys are rejected rather than silently ignored
"""

import difflib
import math
import re
from typing import Any, Dict

from src.retriever import KnowledgeBase

MAX_TEXT_CHARS = 300
ALLOWED_KEYS = {"text", "genre", "mood", "energy"}


class ValidationError(ValueError):
    """Raised with a user-readable explanation of why a request was rejected."""


def _check_name(kind: str, value: Any, kb: KnowledgeBase) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{kind} must be a non-empty string, got {value!r}")
    name = value.strip().lower()
    aliases = kb.alias_lookup(kind)
    if name in aliases:
        return aliases[name]
    vocab = kb.vocabulary(kind)
    if name in vocab:
        return name
    close = difflib.get_close_matches(name, vocab + list(aliases), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(sorted({aliases.get(c, c) for c in close}))}?" if close else ""
    raise ValidationError(f"Unknown {kind} '{value.strip()}'. Known {kind}s: {', '.join(vocab)}.{hint}")


def validate_request(raw: Any, kb: KnowledgeBase) -> Dict[str, Any]:
    if isinstance(raw, str):
        raw = {"text": raw}
    if not isinstance(raw, dict):
        raise ValidationError(f"Request must be text or a dict, got {type(raw).__name__}")
    extra = set(raw) - ALLOWED_KEYS
    if extra:
        raise ValidationError(f"Unsupported field(s): {', '.join(sorted(extra))}. Allowed: {', '.join(sorted(ALLOWED_KEYS))}")

    clean: Dict[str, Any] = {}
    text = raw.get("text")
    if text is not None:
        if not isinstance(text, str):
            raise ValidationError(f"text must be a string, got {type(text).__name__}")
        text = re.sub(r"[\x00-\x1f\x7f]+", " ", text).strip()
        if len(text) > MAX_TEXT_CHARS:
            raise ValidationError(f"text is too long ({len(text)} characters; the limit is {MAX_TEXT_CHARS})")
        if text:
            clean["text"] = text

    for kind in ("genre", "mood"):
        if raw.get(kind) is not None:
            clean[kind] = _check_name(kind, raw[kind], kb)

    if raw.get("energy") is not None:
        e = raw["energy"]
        if isinstance(e, bool) or not isinstance(e, (int, float)) or math.isnan(e) or math.isinf(e):
            raise ValidationError(f"energy must be a number between 0 and 1, got {e!r}")
        if not 0.0 <= e <= 1.0:
            raise ValidationError(f"energy must be between 0.0 and 1.0, got {e}")
        clean["energy"] = float(e)

    if not clean:
        raise ValidationError("Empty request: give a description (text) and/or a genre, mood or energy value.")
    return clean
