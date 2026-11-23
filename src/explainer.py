"""
Explanations: baseline vs specialized.

baseline     the raw list of scoring terms that fired ("genre match (+2.0), mood match (+1.0)...").
specialized  a constrained, few-shot style template. Every explanation must be exactly two
             sentences plus a confidence tag:
               1. "<Title> by <Artist> ranks here because <up to 3 matched facts>."
               2. the retrieved note sentence that applies (or a fixed 'no note' sentence)
               [<level> confidence <0.00>]
             Facts come only from the song row and the terms that fired, never free text, so the
             explanation cannot claim anything the ranking did not use. FEW_SHOT_EXAMPLES are the
             worked examples of the format; tests render them and require an exact match.
             `check_constraints` enforces the format, and the pipeline falls back to the
             baseline text if a rendered explanation ever violates it.
"""

import re
from typing import Dict, List

from src.rag import RankedSong

MAX_WORDS = 70
MAX_FACTS = 3
NO_NOTE = "No retrieved note applies to this song, so it is ranked on the scoring terms alone."
TAG_RE = re.compile(r"\s*\[(high|medium|low) confidence \d\.\d\d\]$")


def confidence_level(confidence: float) -> str:
    return "high" if confidence >= 0.75 else "medium" if confidence >= 0.5 else "low"


def explain_baseline(r: RankedSong) -> str:
    return ", ".join(r.reasons) if r.reasons else "no scoring terms matched"


def _facts(r: RankedSong) -> List[str]:
    s = r.song
    facts: List[str] = []
    for reason in r.reasons:
        if reason.startswith("genre match"):
            facts.append(f"its genre ({s['genre']}) matches")
        elif reason.startswith("mood match"):
            facts.append(f"its mood ({s['mood']}) matches")
        elif reason.startswith("energy similarity"):
            sim = float(re.search(r"\+(\d\.\d+)", reason).group(1))   # 1 - |song - target|, so the claim stays honest
            e = float(s["energy"])
            facts.append(f"its energy ({e:.2f}) is close to your target" if sim >= 0.85 else
                         f"its energy ({e:.2f}) is only partly near your target" if sim >= 0.6 else
                         f"its energy ({e:.2f}) is far from your target")
        elif "inferred from notes" in reason and reason.startswith("genre"):
            facts.append(f"{s['genre']} fits what you described")
        elif "inferred from notes" in reason and reason.startswith("mood"):
            facts.append(f"its {s['mood']} mood fits what you described")
        elif "related genre" in reason:
            facts.append(f"{s['genre']} is a neighbouring genre in the notes")
        elif "related mood" in reason:
            facts.append(f"{s['mood']} is a related mood in the notes")
        elif reason.startswith("energy near notes"):
            facts.append(f"its energy ({float(s['energy']):.2f}) suits the typical level in the notes")
        elif reason.startswith("energy inside"):
            facts.append(f"its energy ({float(s['energy']):.2f}) sits in the typical range for the mood")
        elif reason.startswith("catalog tag"):
            facts.append(f"its '{s.get('mood_tag', '')}' tag matches your words")
    return facts[:MAX_FACTS]


def _join(facts: List[str]) -> str:
    return facts[0] if len(facts) == 1 else ", ".join(facts[:-1]) + " and " + facts[-1]


def render(r: RankedSong, confidence: float) -> str:
    facts = _facts(r)
    if not facts:
        facts = ["it was the closest remaining song"]
    level = confidence_level(confidence)
    first = f"{r.song['title']} by {r.song['artist']} ranks here because {_join(facts)}."
    if level == "low":
        first = "Weaker match: " + first
    return f"{first} {r.note or NO_NOTE} [{level} confidence {confidence:.2f}]"


def check_constraints(text: str, r: RankedSong) -> List[str]:
    """Return a list of violated constraints (empty means the explanation is valid)."""
    problems = []
    if not TAG_RE.search(text):
        problems.append("missing confidence tag")
    body = TAG_RE.sub("", text)
    if len(re.split(r"(?<=[.!?])\s+", body.strip())) != 2:
        problems.append("must be exactly two sentences")
    if r.song["title"] not in body:
        problems.append("must name the song")
    if len(body.split()) > MAX_WORDS:
        problems.append(f"longer than {MAX_WORDS} words")
    return problems


def explain_specialized(r: RankedSong, confidence: float) -> str:
    text = render(r, confidence)
    return text if not check_constraints(text, r) else explain_baseline(r)


# Worked examples of the target format (inputs -> exact expected output).
FEW_SHOT_EXAMPLES: List[Dict] = [
    {
        "song": {"title": "Sunrise City", "artist": "Neon Echo", "genre": "pop", "mood": "happy", "energy": 0.82},
        "reasons": ["genre match (+2.0)", "mood match (+1.0)", "energy similarity (+0.92)"],
        "note": "Pop is built around catchy hooks and polished production, and it bends easily toward dance or indie sounds.",
        "confidence": 0.79,
        "expected": "Sunrise City by Neon Echo ranks here because its genre (pop) matches, its mood (happy) matches and "
                    "its energy (0.82) is close to your target. Pop is built around catchy hooks and polished production, "
                    "and it bends easily toward dance or indie sounds. [high confidence 0.79]",
    },
    {
        "song": {"title": "Library Rain", "artist": "Paper Lanterns", "genre": "lofi", "mood": "chill", "energy": 0.35},
        "reasons": ["'lofi' is a related genre in the notes (+0.75)", "energy similarity (+0.95)"],
        "note": "Lofi is low-tempo, mostly instrumental and warm, which is why it dominates study and coding playlists.",
        "confidence": 0.69,
        "expected": "Library Rain by Paper Lanterns ranks here because lofi is a neighbouring genre in the notes and "
                    "its energy (0.35) is close to your target. Lofi is low-tempo, mostly instrumental and warm, which is why "
                    "it dominates study and coding playlists. [medium confidence 0.69]",
    },
    {
        "song": {"title": "Storm Runner", "artist": "Voltline", "genre": "rock", "mood": "intense", "energy": 0.91},
        "reasons": ["energy similarity (+0.96)"],
        "note": None,
        "confidence": 0.31,
        "expected": "Weaker match: Storm Runner by Voltline ranks here because its energy (0.91) is close to your target. "
                    "No retrieved note applies to this song, so it is ranked on the scoring terms alone. [low confidence 0.31]",
    },
]
