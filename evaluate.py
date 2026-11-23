"""
Evaluation harness: runs the full pipeline on predefined inputs, checks each against its
expectation, prints a pass/fail summary and the average confidence, and exits 0 when all pass.

  python evaluate.py
"""

import sys
from typing import Any, Callable, Dict, List, Optional

from src.pipeline import Pipeline, PipelineResult


def top_genres(r: PipelineResult, n: int = 3) -> List[str]:
    return [rec.song["genre"] for rec in r.recommendations[:n]]


# Each case: name, input, expectation (a callable returning (passed, detail)).
CASES: List[Dict[str, Any]] = [
    {"name": "Structured: pop / happy / 0.8", "input": {"genre": "pop", "mood": "happy", "energy": 0.8},
     "check": lambda r: (r.ok and r.recommendations[0].song["title"] == "Sunrise City", "top should be Sunrise City")},
    {"name": "Free text: late-night study", "input": "late night study session, something mellow",
     "check": lambda r: (r.ok and r.mode == "rag" and r.recommendations[0].song["genre"] == "lofi", "retrieval should infer lofi")},
    {"name": "Free text: gym workout", "input": "pumped up workout at the gym",
     "check": lambda r: (r.ok and r.recommendations[0].song["mood"] in ("intense", "euphoric", "angry"), "top should be a high-energy mood")},
    {"name": "Related genre surfaces (jazz)", "input": {"genre": "jazz", "mood": "relaxed", "energy": 0.3},
     "check": lambda r: (r.ok and "lofi" in top_genres(r), "notes should pull lofi into the top 3")},
    {"name": "Edge: empty string", "input": "",
     "check": lambda r: (not r.ok and "Empty" in (r.error or ""), "should be rejected as empty")},
    {"name": "Edge: whitespace only", "input": {"text": "   \n\t "},
     "check": lambda r: (not r.ok and "Empty" in (r.error or ""), "should be rejected as empty")},
    {"name": "Edge: unknown genre", "input": {"genre": "zydeco", "mood": "happy"},
     "check": lambda r: (not r.ok and "Unknown genre" in (r.error or ""), "should be rejected with a clear message")},
    {"name": "Edge: unknown mood with typo hint", "input": {"mood": "chil"},
     "check": lambda r: (not r.ok and "chill" in (r.error or ""), "should suggest 'chill'")},
    {"name": "Edge: energy out of range", "input": {"genre": "rock", "energy": 1.7},
     "check": lambda r: (not r.ok and "between 0.0 and 1.0" in (r.error or ""), "should be rejected")},
    {"name": "Edge: energy is not a number", "input": {"genre": "rock", "energy": "loud"},
     "check": lambda r: (not r.ok and "number" in (r.error or ""), "should be rejected")},
    {"name": "Edge: wrong type (int)", "input": 42,
     "check": lambda r: (not r.ok and "text or a dict" in (r.error or ""), "should be rejected")},
    {"name": "Edge: gibberish text", "input": "qwzx plorb vnnm",
     "check": lambda r: (not r.ok and "Could not match" in (r.error or ""), "should be rejected as not understood")},
    {"name": "Edge: very long text", "input": "chill " * 100,
     "check": lambda r: (not r.ok and "too long" in (r.error or ""), "should be rejected")},
    {"name": "Contradiction: sad blues at 0.95", "input": {"genre": "blues", "mood": "sad", "energy": 0.95},
     "check": lambda r: (r.ok and any("unusual" in w for w in r.warnings), "should succeed with a contradiction warning")},
    {"name": "Unknown key is rejected", "input": {"genre": "pop", "artist": "x"},
     "check": lambda r: (not r.ok and "Unsupported" in (r.error or ""), "should be rejected")},
]


def main() -> int:
    pipeline = Pipeline()
    passed, confidences = 0, []
    print(f"Running {len(CASES)} evaluation cases\n")
    for i, case in enumerate(CASES, start=1):
        result = pipeline.run(case["input"])
        ok, expectation = case["check"](result)
        passed += bool(ok)
        outcome = f"ok, confidence {result.confidence:.2f}" if result.ok else f"rejected: {(result.error or '')[:60]}"
        if result.ok:
            confidences.append(result.confidence)
        print(f"[{'PASS' if ok else 'FAIL'}] {i:2d}. {case['name']:<36} -> {outcome}")
        if not ok:
            print(f"        expected: {expectation}")
    avg = sum(confidences) / len(confidences) if confidences else 0.0
    print(f"\nSummary: {passed}/{len(CASES)} passed")
    print(f"Average confidence over {len(confidences)} successful recommendation runs: {avg:.2f}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
