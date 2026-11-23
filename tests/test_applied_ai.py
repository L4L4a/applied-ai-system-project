"""Unit tests for retrieval, RAG ranking, validation, confidence and the pipeline (run from the project root)."""

import pytest

from src.pipeline import Pipeline
from src.rag import build_context, parse_request, rank_baseline, rank_rag
from src.retriever import KnowledgeBase, retrieve
from src.validation import ValidationError, validate_request


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline()


# --- retriever -----------------------------------------------------------
def test_knowledge_base_has_three_sources(kb):
    assert {e.kind for e in kb.entries} == {"genre", "mood", "song"}
    assert {e.source for e in kb.entries} >= {"genre_notes.md", "mood_notes.md", "songs.csv"}


def test_retrieve_finds_lofi_for_study(kb):
    names = [h.entry.name for h in retrieve("study session for coding", kb)]
    assert "lofi" in names and "focused" in names


def test_retrieve_alias_and_no_match(kb):
    assert retrieve("some rap please", kb)[0].entry.name == "hip hop"
    assert retrieve("qwzx plorb", kb) == []


# --- RAG changes behaviour -------------------------------------------------
def test_rag_changes_ranking_and_adds_related_genre(kb):
    songs = [dict(e.song, energy=float(e.song["energy"])) for e in kb.of_kind("song")]
    req = parse_request({"genre": "jazz", "mood": "relaxed", "energy": 0.3}, kb)
    ctx = build_context(req, kb)
    base = [r.song["title"] for r in rank_baseline(req, songs)]
    rag = rank_rag(req, ctx, songs)
    assert base != [r.song["title"] for r in rag]
    assert any("related genre" in reason for r in rag for reason in r.reasons)


def test_rag_infers_missing_preferences(kb):
    req = parse_request({"text": "late night study session"}, kb)
    ctx = build_context(req, kb)
    assert ctx.inferred.get("genre") == "lofi"


# --- validation ------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", {}, {"text": ""}, None, 5, {"genre": "zydeco"},
                                 {"energy": 2}, {"energy": -0.1}, {"energy": "x"}, {"energy": True},
                                 {"genre": "pop", "bogus": 1}, "x" * 400])
def test_validation_rejects(bad, kb):
    with pytest.raises(ValidationError):
        validate_request(bad, kb)


def test_validation_unknown_genre_message_has_hint(kb):
    with pytest.raises(ValidationError, match="Did you mean"):
        validate_request({"genre": "rok"}, kb)


def test_validation_normalises(kb):
    assert validate_request({"genre": " Lo-Fi ", "energy": 1}, kb) == {"genre": "lofi", "energy": 1.0}
    assert validate_request("Chill\x00 vibes", kb) == {"text": "Chill vibes"}


# --- pipeline --------------------------------------------------------------
def test_pipeline_success_has_confidence_and_trace(pipeline):
    r = pipeline.run({"genre": "pop", "mood": "happy", "energy": 0.8})
    assert r.ok and len(r.recommendations) == 5
    assert all(0.0 <= rec.confidence <= 1.0 for rec in r.recommendations)
    assert [s["step"] for s in r.steps] == ["validate", "parse", "retrieve", "rank", "self-check", "explain"]
    assert r.recommendations[0].song["title"] == "Sunrise City"


def test_pipeline_rejects_instead_of_raising(pipeline):
    for bad in ["", {"genre": "zydeco"}, "qwzx plorb", 3.14]:
        r = pipeline.run(bad)
        assert not r.ok and r.error and not r.recommendations


def test_pipeline_bad_k(pipeline):
    assert not pipeline.run("jazz", k=0).ok


def test_baseline_mode_cannot_read_free_text_without_prefs(pipeline):
    assert not pipeline.run("qwzx", use_rag=False).ok
    assert pipeline.run({"genre": "rock"}, use_rag=False).mode == "baseline"


def test_confidence_higher_for_perfect_match_than_contradiction(pipeline):
    good = pipeline.run({"genre": "lofi", "mood": "chill", "energy": 0.35})
    bad = pipeline.run({"genre": "blues", "mood": "sad", "energy": 0.95})
    assert good.confidence > bad.confidence
    assert bad.warnings


def test_missing_knowledge_base_degrades_safely(tmp_path):
    p = Pipeline(knowledge_dir=str(tmp_path))
    r = p.run({"genre": "pop"})
    assert not r.ok and "knowledge base unavailable" in r.error


# --- specialized explanations ----------------------------------------------
from src.explainer import FEW_SHOT_EXAMPLES, check_constraints, explain_baseline, render  # noqa: E402
from src.rag import RankedSong  # noqa: E402


@pytest.mark.parametrize("ex", FEW_SHOT_EXAMPLES)
def test_few_shot_examples_render_exactly(ex):
    r = RankedSong(ex["song"], 0.0, ex["reasons"], ex["note"])
    assert render(r, ex["confidence"]) == ex["expected"]
    assert check_constraints(ex["expected"], r) == []


def test_constraints_catch_violations():
    ex = FEW_SHOT_EXAMPLES[0]
    r = RankedSong(ex["song"], 0.0, ex["reasons"], ex["note"])
    assert "missing confidence tag" in check_constraints("Sunrise City is nice.", r)
    assert check_constraints("Sunrise City. One. Two. [high confidence 0.90]", r)


def test_pipeline_uses_specialized_text(pipeline):
    rec = pipeline.run({"genre": "pop", "mood": "happy", "energy": 0.8}).recommendations[0]
    assert rec.explanation != rec.baseline_explanation
    assert "confidence" in rec.explanation and "genre match" in rec.baseline_explanation
