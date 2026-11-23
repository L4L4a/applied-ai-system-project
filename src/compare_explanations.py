"""Print baseline vs specialized explanations for a few requests (used for model_card.md).

  python -m src.compare_explanations
"""

from src.pipeline import Pipeline

REQUESTS = [
    {"genre": "pop", "mood": "happy", "energy": 0.8},
    "late night study session, something mellow",
    {"genre": "blues", "mood": "sad", "energy": 0.95},
]


def main() -> None:
    pipeline = Pipeline()
    for raw in REQUESTS:
        result = pipeline.run(raw, k=3)
        print(f"=== Request: {raw!r} (overall confidence {result.confidence:.2f}) ===")
        for rank, rec in enumerate(result.recommendations[:2], start=1):
            print(f"#{rank} {rec.song['title']}")
            print(f"  BASELINE   : {rec.baseline_explanation}")
            print(f"  SPECIALIZED: {rec.explanation}")
        print()


if __name__ == "__main__":
    main()
