"""
Retrieval-augmented ranking.

The retrieved knowledge entries are not decoration: they change both the score and the
explanation of each song.

  * Inference   - if the request leaves out genre / mood / energy, the top retrieved genre
                  and mood notes fill them in (at reduced weight).
  * Adjacency   - songs from a genre the notes list as `related_genres` of the requested
                  genre get +0.75; related moods get +0.5.
  * Energy fit  - songs inside the mood note's typical energy range get +0.25.
  * Catalog tag - a song whose `mood_tag` (from songs.csv) appears in the request gets +0.5.
  * Explanation - the matching note sentence is quoted next to the scoring terms.

`rank_baseline` is the original three-term rule with no retrieval, used for before/after
comparison and as the self-check fallback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.recommender import recommend_songs, score_song
from src.retriever import Entry, Hit, KnowledgeBase, _contains_phrase, retrieve, tokenize

INFERRED_GENRE_WEIGHT = 1.0
INFERRED_MOOD_WEIGHT = 0.5
INFERRED_ENERGY_WEIGHT = 0.5
ADJACENT_GENRE_BONUS = 0.75
ADJACENT_MOOD_BONUS = 0.5
ENERGY_RANGE_BONUS = 0.25
CATALOG_TAG_BONUS = 0.5

ENERGY_WORDS = {
    0.20: ["very calm", "low energy", "sleepy", "sleep", "quiet", "soft", "gentle"],
    0.35: ["mellow", "calm", "slow", "laid back", "unwind"],
    0.65: ["moderate", "medium energy"],
    0.85: ["upbeat", "energetic", "high energy", "pump", "workout", "gym", "party", "dance", "dancing"],
    0.95: ["very intense", "maximum energy", "max energy", "extreme"],
}


@dataclass
class ParsedRequest:
    genre: Optional[str] = None
    mood: Optional[str] = None
    energy: Optional[float] = None
    text: str = ""
    sources: Dict[str, str] = field(default_factory=dict)   # field -> "explicit" | "text"

    def stated_prefs(self) -> Dict:
        prefs: Dict = {}
        if self.genre:
            prefs["genre"] = self.genre
        if self.mood:
            prefs["mood"] = self.mood
        if self.energy is not None:
            prefs["energy"] = self.energy
        return prefs

    def query(self) -> str:
        return " ".join(p for p in (self.text, self.genre or "", self.mood or "") if p)


@dataclass
class Context:
    hits: List[Hit]
    genre: Optional[str]            # effective genre (stated or inferred)
    mood: Optional[str]
    energy: Optional[float]
    inferred: Dict[str, object]     # which of genre/mood/energy came from the notes
    genre_refs: List[Entry]
    mood_refs: List[Entry]
    tag_hits: Dict[str, str]        # song title -> matched tag


@dataclass
class RankedSong:
    song: Dict
    score: float
    reasons: List[str]
    note: Optional[str] = None
    evidence: List[str] = field(default_factory=list)   # names of knowledge entries that changed the score


def parse_request(raw: Dict, kb: KnowledgeBase) -> ParsedRequest:
    """Step 1: turn a validated request dict into genre / mood / energy / text."""
    req = ParsedRequest(text=(raw.get("text") or "").strip())
    for key in ("genre", "mood"):
        value = raw.get(key)
        if value:
            setattr(req, key, str(value).strip().lower())
            req.sources[key] = "explicit"
    if raw.get("energy") is not None:
        req.energy = float(raw["energy"])
        req.sources["energy"] = "explicit"

    q = tokenize(req.text)
    for key, kind in (("genre", "genre"), ("mood", "mood")):
        if getattr(req, key) is None:
            found = [(len(alias), name) for alias, name in kb.alias_lookup(kind).items() if _contains_phrase(q, alias)]
            if found:
                setattr(req, key, max(found)[1])
                req.sources[key] = "text"
    if req.energy is None:
        for level, words in ENERGY_WORDS.items():
            if any(_contains_phrase(q, w) for w in words):
                req.energy = level
                req.sources["energy"] = "text"
                break
    return req


def build_context(req: ParsedRequest, kb: KnowledgeBase) -> Context:
    """Step 2: retrieve notes for the request and work out what they imply."""
    hits = retrieve(req.query(), kb)
    named = [h for h in hits if h.score >= 3.0]
    genre_hits = [h.entry for h in named if h.entry.kind == "genre"]
    if not named and not req.stated_prefs():
        named = hits          # nothing named outright: fall back to weaker topical keyword matches
        genre_hits = [h.entry for h in named if h.entry.kind == "genre"]
    mood_hits = [h.entry for h in named if h.entry.kind == "mood"]
    inferred: Dict[str, object] = {}

    genre, mood, energy = req.genre, req.mood, req.energy
    if genre is None and genre_hits:
        genre = genre_hits[0].name
        inferred["genre"] = genre
    if mood is None and mood_hits:
        mood = mood_hits[0].name
        inferred["mood"] = mood
    if energy is None:
        ranges = [e.energy_range for e in ((kb.get("genre", genre) if genre else None),
                                             (kb.get("mood", mood) if mood else None)) if e and e.energy_range]
        if ranges and inferred:
            energy = round(sum((lo + hi) / 2 for lo, hi in ranges) / len(ranges), 2)
            inferred["energy"] = energy

    def refs(kind: str, name: Optional[str], extra: List[Entry]) -> List[Entry]:
        out: List[Entry] = []
        main = kb.get(kind, name) if name else None
        for e in ([main] if main else []) + extra:
            if e and e not in out:
                out.append(e)
        return out

    tag_hits = {}
    for h in hits:
        if h.entry.kind == "song" and h.matched:
            tag_hits[h.entry.name] = h.matched[0]
    return Context(hits, genre, mood, energy, inferred,
                   refs("genre", genre, genre_hits), refs("mood", mood, mood_hits), tag_hits)


def rank_baseline(req: ParsedRequest, songs: List[Dict], k: int = 5) -> List[RankedSong]:
    """The original rule (genre +2, mood +1, energy similarity) on the stated preferences only."""
    results = recommend_songs(req.stated_prefs(), songs, k=k)
    out = []
    for song, score, expl in results:
        out.append(RankedSong(song, score, [r for r in expl.split(", ") if r != "no scoring terms matched"]))
    return out


def rank_rag(req: ParsedRequest, ctx: Context, songs: List[Dict], k: int = 5) -> List[RankedSong]:
    """Step 3: score every song with the base rule plus retrieval-driven terms."""
    stated = req.stated_prefs()
    related_genres = {g for e in ctx.genre_refs for g in e.related_genres}
    related_moods = {m for e in ctx.mood_refs for m in e.related_moods}
    mood_entry = next((e for e in ctx.mood_refs if e.name == ctx.mood), None)
    ranked: List[RankedSong] = []

    for song in songs:
        score, reasons = score_song(stated, song)
        evidence: List[str] = []
        genre_note = mood_note = False

        if song["genre"] == ctx.genre:
            genre_note = True
        if "genre" in ctx.inferred and song["genre"] == ctx.genre:
            score += INFERRED_GENRE_WEIGHT
            reasons.append(f"genre '{ctx.genre}' inferred from notes (+{INFERRED_GENRE_WEIGHT:.2f})")
            evidence.append(f"genre:{ctx.genre}")
        elif song["genre"] in related_genres and song["genre"] != ctx.genre:
            score += ADJACENT_GENRE_BONUS
            reasons.append(f"'{song['genre']}' is a related genre in the notes (+{ADJACENT_GENRE_BONUS:.2f})")
            evidence.append(f"genre:{ctx.genre}")
            genre_note = True

        if "mood" in ctx.inferred and song["mood"] == ctx.mood:
            score += INFERRED_MOOD_WEIGHT
            reasons.append(f"mood '{ctx.mood}' inferred from notes (+{INFERRED_MOOD_WEIGHT:.2f})")
            evidence.append(f"mood:{ctx.mood}")
            mood_note = True
        elif song["mood"] in related_moods and song["mood"] != ctx.mood:
            score += ADJACENT_MOOD_BONUS
            reasons.append(f"'{song['mood']}' is a related mood in the notes (+{ADJACENT_MOOD_BONUS:.2f})")
            evidence.append(f"mood:{ctx.mood}")
            mood_note = True

        if "energy" in ctx.inferred and ctx.energy is not None:
            sim = INFERRED_ENERGY_WEIGHT * (1 - abs(float(song["energy"]) - ctx.energy))
            score += sim
            reasons.append(f"energy near notes' typical {ctx.energy:.2f} (+{sim:.2f})")
        if mood_entry and mood_entry.energy_range:
            lo, hi = mood_entry.energy_range
            if lo <= float(song["energy"]) <= hi:
                score += ENERGY_RANGE_BONUS
                reasons.append(f"energy inside '{mood_entry.name}' range {lo:.2f}-{hi:.2f} (+{ENERGY_RANGE_BONUS:.2f})")
                evidence.append(f"mood:{mood_entry.name}")

        tag = ctx.tag_hits.get(song["title"])
        if tag:
            score += CATALOG_TAG_BONUS
            reasons.append(f"catalog tag '{tag}' matches request (+{CATALOG_TAG_BONUS:.2f})")
            evidence.append(f"song:{song['title']}")

        marker = "__genre__" if genre_note else ("__mood__" if mood_note else None)
        ranked.append(RankedSong(song, score, reasons, marker, sorted(set(evidence))))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:k]


def attach_notes(ranked: List[RankedSong], kb: KnowledgeBase) -> None:
    """Replace the genre/mood markers with the quoted note sentence."""
    for r in ranked:
        kind = {"__genre__": "genre", "__mood__": "mood"}.get(r.note or "")
        entry = kb.get(kind, r.song[kind]) if kind else None
        r.note = entry.first_sentence if entry else None
