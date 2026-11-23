"""
Retriever for the knowledge base in data/knowledge/.

Three sources are indexed into one list of `Entry` objects:
  * genre_notes.md  - one entry per genre   (kind="genre")
  * mood_notes.md   - one entry per mood    (kind="mood")
  * songs.csv       - one entry per song    (kind="song"; keywords come from mood_tag, genre, mood, title)

`retrieve(query)` scores entries by keyword overlap (no network, no embeddings) and
returns the best matches. The recommender uses what comes back to change its ranking
and to write its explanations (see src/rag.py).
"""

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

KNOWLEDGE_DIR = os.path.join("data", "knowledge")
SONGS_CSV = os.path.join("data", "songs.csv")

NAME_WEIGHT = 4.0      # query names the genre/mood itself
ALIAS_WEIGHT = 3.0     # query uses an alias ("lo-fi", "rap")
KEYWORD_WEIGHT = 1.0   # query shares a topical keyword ("study", "workout")


@dataclass
class Entry:
    kind: str                       # "genre" | "mood" | "song"
    name: str
    text: str = ""
    source: str = ""
    aliases: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    related_genres: List[str] = field(default_factory=list)
    related_moods: List[str] = field(default_factory=list)
    energy_range: Optional[tuple] = None
    song: Optional[Dict] = None

    @property
    def first_sentence(self) -> str:
        m = re.match(r"(.+?[.!?])(\s|$)", self.text.strip())
        return m.group(1) if m else self.text.strip()


@dataclass
class Hit:
    entry: Entry
    score: float
    matched: List[str]


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> List[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9&']+", text.lower())]


def _contains_phrase(query_tokens: Sequence[str], phrase: str) -> bool:
    p = tokenize(phrase)
    if not p:
        return False
    n = len(p)
    return any(list(query_tokens[i:i + n]) == p for i in range(len(query_tokens) - n + 1))


def _csv_list(value: str) -> List[str]:
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def parse_notes(path: str, kind: str) -> List[Entry]:
    """Parse a notes file: '## name' sections, '- key: value' metadata, then free text."""
    entries: List[Entry] = []
    current: Optional[Entry] = None
    text_lines: List[str] = []

    def flush() -> None:
        if current is not None:
            current.text = " ".join(text_lines).strip()
            entries.append(current)

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("## "):
                flush()
                current = Entry(kind=kind, name=line[3:].strip().lower(), source=os.path.basename(path))
                text_lines = []
            elif current is None or not line.strip():
                continue
            elif line.startswith("- ") and ":" in line:
                key, _, value = line[2:].partition(":")
                key, value = key.strip(), value.strip()
                if key == "aliases":
                    current.aliases = _csv_list(value)
                elif key == "keywords":
                    current.keywords = _csv_list(value)
                elif key == "related_genres":
                    current.related_genres = _csv_list(value)
                elif key == "related_moods":
                    current.related_moods = _csv_list(value)
                elif key == "energy":
                    lo, _, hi = value.partition("-")
                    current.energy_range = (float(lo), float(hi))
            else:
                text_lines.append(line.strip())
    flush()
    return entries


def song_entries(songs_csv: str = SONGS_CSV) -> List[Entry]:
    entries: List[Entry] = []
    with open(songs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tag = row.get("mood_tag", "").strip().lower()
            entries.append(Entry(
                kind="song",
                name=row["title"],
                source=os.path.basename(songs_csv),
                keywords=[k for k in (tag, row["genre"].lower(), row["mood"].lower()) if k],
                text=f"{row['title']} by {row['artist']} is {row['genre']} with a {row['mood']} mood"
                     + (f" and a '{tag}' feel." if tag else "."),
                song=row,
            ))
    return entries


class KnowledgeBase:
    """All entries from all sources, plus helpers for vocabulary lookups."""

    def __init__(self, knowledge_dir: str = KNOWLEDGE_DIR, songs_csv: str = SONGS_CSV):
        self.entries: List[Entry] = []
        self.entries += parse_notes(os.path.join(knowledge_dir, "genre_notes.md"), "genre")
        self.entries += parse_notes(os.path.join(knowledge_dir, "mood_notes.md"), "mood")
        self.entries += song_entries(songs_csv)

    def of_kind(self, kind: str) -> List[Entry]:
        return [e for e in self.entries if e.kind == kind]

    def get(self, kind: str, name: str) -> Optional[Entry]:
        name = name.lower()
        for e in self.entries:
            if e.kind == kind and e.name == name:
                return e
        return None

    def vocabulary(self, kind: str) -> List[str]:
        """Known genre/mood names, taken from the notes AND the catalog."""
        names = {e.name for e in self.of_kind(kind)}
        names |= {e.song[kind].lower() for e in self.of_kind("song")}
        return sorted(names)

    def alias_lookup(self, kind: str) -> Dict[str, str]:
        out = {}
        for e in self.of_kind(kind):
            out[e.name] = e.name
            for a in e.aliases:
                out[a] = e.name
        return out


def retrieve(query: str, kb: KnowledgeBase, top_n: int = 6, kinds: Sequence[str] = ("genre", "mood", "song")) -> List[Hit]:
    """Return up to top_n entries with a positive keyword-overlap score, best first."""
    q = tokenize(query)
    hits: List[Hit] = []
    for e in kb.entries:
        if e.kind not in kinds:
            continue
        score, matched = 0.0, []
        if e.kind != "song" and _contains_phrase(q, e.name):
            score += NAME_WEIGHT
            matched.append(e.name)
        for alias in e.aliases:
            if _contains_phrase(q, alias):
                score += ALIAS_WEIGHT
                matched.append(alias)
        seen = set()
        for kw in e.keywords:
            key = tuple(tokenize(kw))
            if key in seen:
                continue
            seen.add(key)
            if _contains_phrase(q, kw):
                score += KEYWORD_WEIGHT
                matched.append(kw)
        if e.kind == "song":
            # song entries only count for their distinctive mood_tag / title, not generic genre words
            score = KEYWORD_WEIGHT * sum(1 for kw in e.keywords[:1] if _contains_phrase(q, kw))
            matched = [kw for kw in e.keywords[:1] if _contains_phrase(q, kw)]
        if score > 0:
            hits.append(Hit(e, score, matched))
    hits.sort(key=lambda h: (-h.score, h.entry.kind, h.entry.name))
    return hits[:top_n]
