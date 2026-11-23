"""
Multi-step pipeline: validate -> parse -> retrieve -> rank -> self-check -> explain.

Every step appends a record to `PipelineResult.steps` (this is the agentic trace saved by
src/trace.py). Errors never escape: unexpected exceptions are logged and turned into a
safe `ok=False` result.
"""

import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.explainer import explain_baseline, explain_specialized
from src.logging_setup import get_logger
from src.rag import (Context, ParsedRequest, RankedSong, attach_notes, build_context, parse_request,
                     rank_baseline, rank_rag)
from src.recommender import load_songs
from src.retriever import KNOWLEDGE_DIR, SONGS_CSV, KnowledgeBase
from src.validation import ValidationError, validate_request

LOW_CONFIDENCE = 0.5
log = get_logger()


@dataclass
class Recommendation:
    song: Dict
    score: float
    confidence: float
    explanation: str            # specialized (few-shot template) explanation
    baseline_explanation: str   # plain list of scoring terms
    evidence: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    ok: bool
    request: Any = None
    mode: str = ""
    error: Optional[str] = None
    recommendations: List[Recommendation] = field(default_factory=list)
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)


def confidence_for(r: RankedSong, ctx: Context, use_rag: bool) -> float:
    """
    0-1 score = 0.55 * normalised score + 0.30 * preference coverage + 0.15 * retrieval support.
      normalised score : score / best achievable score for this request
      coverage         : how many of the (stated or inferred) genre/mood/energy dimensions the song
                         satisfies (exact = 1, related per notes = 0.5, energy within 0.15 = 1, 0.30 = 0.5)
      support          : 1 if retrieved notes contributed to this song's score, else 0
    """
    inferred = ctx.inferred if use_rag else {}
    ceiling, cover = 0.0, []
    if ctx.genre:
        ceiling += INFERRED["genre"] if "genre" in inferred else 2.0
        cover.append(1.0 if r.song["genre"] == ctx.genre else 0.5 if any(
            r.song["genre"] in e.related_genres for e in ctx.genre_refs) and use_rag else 0.0)
    if ctx.mood:
        ceiling += INFERRED["mood"] if "mood" in inferred else 1.0
        cover.append(1.0 if r.song["mood"] == ctx.mood else 0.5 if any(
            r.song["mood"] in e.related_moods for e in ctx.mood_refs) and use_rag else 0.0)
    if ctx.energy is not None:
        ceiling += INFERRED["energy"] if "energy" in inferred else 1.0
        gap = abs(float(r.song["energy"]) - ctx.energy)
        cover.append(1.0 if gap <= 0.15 else 0.5 if gap <= 0.30 else 0.0)
    norm = min(1.0, r.score / ceiling) if ceiling else 0.0
    coverage = sum(cover) / len(cover) if cover else 0.0
    support = 1.0 if (use_rag and r.evidence) else 0.0
    return round(0.55 * norm + 0.30 * coverage + 0.15 * support, 2)


INFERRED = {"genre": 1.0, "mood": 0.5, "energy": 0.5}


def self_check(ranked: List[RankedSong], ctx: Context, k: int, n_songs: int) -> List[Dict[str, Any]]:
    """Cheap sanity checks on a ranking. Returns [{check, passed, detail}]."""
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})

    add("result count", len(ranked) == min(k, n_songs), f"{len(ranked)} results for k={k}")
    titles = [r.song["title"] for r in ranked]
    add("no duplicates", len(set(titles)) == len(titles), "all titles unique" if len(set(titles)) == len(titles) else "duplicate title")
    scores = [r.score for r in ranked]
    add("sorted by score", all(a >= b for a, b in zip(scores, scores[1:])), "scores non-increasing")
    if ranked:
        top = ranked[0].song
        ok = ((ctx.genre is not None and top["genre"] == ctx.genre)
              or (ctx.mood is not None and top["mood"] == ctx.mood)
              or (ctx.energy is not None and abs(float(top["energy"]) - ctx.energy) <= 0.25)
              or any(top["genre"] in e.related_genres for e in ctx.genre_refs))
        add("top result relevant", ok, f"top='{top['title']}' ({top['genre']}/{top['mood']}/{top['energy']})")
    return checks


class Pipeline:
    def __init__(self, songs_csv: str = SONGS_CSV, knowledge_dir: str = KNOWLEDGE_DIR):
        self.songs = load_songs(songs_csv)
        self.kb: Optional[KnowledgeBase] = None
        self.kb_error: Optional[str] = None
        try:
            self.kb = KnowledgeBase(knowledge_dir, songs_csv)
        except (OSError, ValueError, KeyError) as exc:
            self.kb_error = f"knowledge base unavailable ({exc})"
            log.error(self.kb_error)

    # -- helpers ---------------------------------------------------------
    def _step(self, result: PipelineResult, name: str, **detail: Any) -> None:
        result.steps.append({"step": name, **detail})

    def _fail(self, result: PipelineResult, message: str, level: str = "warning") -> PipelineResult:
        result.ok, result.error = False, message
        getattr(log, level)("REJECTED %s | request=%r", message, result.request)
        return result

    # -- main entry ------------------------------------------------------
    def run(self, raw: Any, k: int = 5, use_rag: bool = True) -> PipelineResult:
        result = PipelineResult(ok=False, request=raw)
        try:
            return self._run(result, raw, k, use_rag)
        except Exception as exc:  # last-resort guardrail: never crash the caller
            log.error("UNEXPECTED %s: %s\n%s", type(exc).__name__, exc, traceback.format_exc())
            result.ok, result.recommendations = False, []
            result.error = "Internal error while building recommendations; nothing was returned. See logs/system.log."
            return result

    def _run(self, result: PipelineResult, raw: Any, k: int, use_rag: bool) -> PipelineResult:
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 20:
            return self._fail(result, f"k must be an integer from 1 to 20, got {k!r}")
        if use_rag and self.kb is None:
            result.warnings.append(f"Retrieval disabled: {self.kb_error}. Falling back to baseline ranking.")
            use_rag = False
        vocab_kb = self.kb
        if vocab_kb is None:
            return self._fail(result, "knowledge base unavailable, cannot validate input", "error")

        # 1. validate
        try:
            clean = validate_request(raw, vocab_kb)
        except ValidationError as exc:
            self._step(result, "validate", passed=False, reason=str(exc))
            return self._fail(result, str(exc))
        self._step(result, "validate", passed=True, cleaned=clean)

        # 2. parse
        req = parse_request(clean, vocab_kb)
        self._step(result, "parse", genre=req.genre, mood=req.mood, energy=req.energy, text=req.text, sources=req.sources)
        if not use_rag and not req.stated_prefs():
            return self._fail(result, "Baseline mode needs a genre, mood or energy; free text can only be interpreted with retrieval on.")

        # 3. retrieve
        ctx = build_context(req, vocab_kb) if use_rag else Context([], req.genre, req.mood, req.energy, {}, [], [], {})
        if use_rag:
            self._step(result, "retrieve",
                       hits=[{"kind": h.entry.kind, "name": h.entry.name, "source": h.entry.source,
                              "score": h.score, "matched": h.matched} for h in ctx.hits],
                       inferred=ctx.inferred)
        else:
            self._step(result, "retrieve", skipped="retrieval disabled (baseline mode)")
        if ctx.genre is None and ctx.mood is None and ctx.energy is None:
            return self._fail(result, "Could not match the request to any known genre, mood or topic. "
                                      "Try naming a genre or mood (e.g. 'chill lofi for studying').")
        result.mode = "rag" if use_rag else "baseline"
        result.warnings += self._contradiction_warnings(ctx, vocab_kb)

        # 4. rank
        ranked = rank_rag(req, ctx, self.songs, k) if use_rag else rank_baseline(req, self.songs, k)
        self._step(result, "rank", mode=result.mode, top=[{"title": r.song["title"], "score": round(r.score, 2)} for r in ranked])

        # 5. self-check (with fallback to the plain rule if the retrieval ranking looks wrong)
        checks = self_check(ranked, ctx, k, len(self.songs))
        if not all(c["passed"] for c in checks) and use_rag:
            result.warnings.append("Self-check failed on the retrieval ranking; used baseline ranking instead.")
            log.warning("self-check failed %s", checks)
            use_rag, result.mode = False, "baseline (self-check fallback)"
            ranked = rank_baseline(req, self.songs, k)
            checks = self_check(ranked, ctx, k, len(self.songs))
        self._step(result, "self-check", checks=checks, passed=all(c["passed"] for c in checks), final_mode=result.mode)
        if not all(c["passed"] for c in checks):
            return self._fail(result, "Self-check failed; recommendations withheld.", "error")

        # 6. explain + confidence
        attach_notes(ranked, vocab_kb)
        for r in ranked:
            conf = confidence_for(r, ctx, use_rag)
            result.recommendations.append(Recommendation(
                r.song, r.score, conf, explain_specialized(r, conf), explain_baseline(r), r.evidence))
        result.confidence = round(sum(r.confidence for r in result.recommendations) / len(result.recommendations), 2)
        if result.confidence < LOW_CONFIDENCE:
            result.warnings.append(f"Low confidence ({result.confidence:.2f}): the catalog has few songs matching this request.")
        self._step(result, "explain", confidences=[r.confidence for r in result.recommendations],
                   overall_confidence=result.confidence, warnings=result.warnings)
        result.ok = True
        log.info("OK mode=%s request=%r top=%r confidence=%.2f warnings=%d", result.mode, result.request,
                 result.recommendations[0].song["title"], result.confidence, len(result.warnings))
        return result

    @staticmethod
    def _contradiction_warnings(ctx: Context, kb: KnowledgeBase) -> List[str]:
        warnings = []
        entry = kb.get("mood", ctx.mood) if ctx.mood else None
        if entry and entry.energy_range and ctx.energy is not None:
            lo, hi = entry.energy_range
            if not lo - 0.1 <= ctx.energy <= hi + 0.1:
                warnings.append(f"Energy {ctx.energy:.2f} is unusual for mood '{entry.name}' (typical {lo:.2f}-{hi:.2f}); "
                                f"expect a compromise between mood and energy.")
        return warnings
