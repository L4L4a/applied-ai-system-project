"""
Command line runner for the Music Recommender Simulation.

Run from the project root, for example:
  python -m src.main                                  # all profiles, balanced mode, table
  python -m src.main --mode mood-first --diversity
  python -m src.main --profile "Chill Lofi" --mode all
  python -m src.main --basic --plain                  # baseline 3-term scoring, plain text

Applied-AI (retrieval) mode, used when any of --query/--genre/--mood/--energy is given:
  python -m src.main --query "late night study session, something mellow"
  python -m src.main --genre jazz --mood relaxed --energy 0.3 --compare   # before/after retrieval
  python -m src.main --query "" --plain                                   # rejected by the guardrails
"""

import argparse
from typing import Dict, List

from src.pipeline import Pipeline
from src.recommender import STRATEGIES, get_strategy, load_songs, recommend_songs

# Core preferences (genre / mood / energy) plus optional extra-attribute preferences.
PROFILES: Dict[str, Dict] = {
    "High-Energy Pop": {
        "genre": "pop", "mood": "happy", "energy": 0.90,
        "mood_tag": "uplifting", "target_decade": 2020, "target_popularity": 80,
        "target_danceability": 0.85,
    },
    "Chill Lofi": {
        "genre": "lofi", "mood": "chill", "energy": 0.35,
        "mood_tag": "cozy", "target_decade": 2020, "target_popularity": 55,
        "target_acousticness": 0.80, "target_instrumentalness": 0.60,
    },
    "Deep Intense Rock": {
        "genre": "rock", "mood": "intense", "energy": 0.92,
        "mood_tag": "aggressive", "target_decade": 2010, "target_popularity": 60,
        "target_liveness": 0.40,
    },
    # Adversarial: the saddest genre in the catalog paired with an energy level no sad song has,
    # and a "gritty 1970s" taste that contradicts the high-energy, low-acousticness targets.
    "Adversarial (sad blues at max energy)": {
        "genre": "blues", "mood": "sad", "energy": 0.95,
        "mood_tag": "gritty", "target_decade": 1970, "target_popularity": 90,
        "target_acousticness": 0.10,
    },
}

CORE_KEYS = ("genre", "mood", "energy")


def format_results(results, as_table: bool) -> str:
    if as_table:
        from tabulate import tabulate  # imported lazily: retrieval mode and tests need no extra packages
        rows = [
            [rank, song["title"], song["artist"], f"{score:.2f}", explanation]
            for rank, (song, score, explanation) in enumerate(results, start=1)
        ]
        return tabulate(rows, headers=["Rank", "Title", "Artist", "Score", "Reasons"], tablefmt="github")
    lines: List[str] = []
    for rank, (song, score, explanation) in enumerate(results, start=1):
        lines.append(f"{rank}. {song['title']} by {song['artist']} - Score: {score:.2f}")
        lines.append(f"   Because: {explanation}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Music Recommender Simulation")
    parser.add_argument("--mode", default="balanced", choices=sorted(STRATEGIES) + ["all"],
                        help="ranking strategy (default: balanced); 'all' runs every mode")
    parser.add_argument("--profile", default="all",
                        help="profile name (or unique substring); default: all profiles")
    parser.add_argument("--diversity", action="store_true",
                        help="re-rank the top k with artist/genre repetition penalties")
    parser.add_argument("--basic", action="store_true",
                        help="ignore extra-attribute preferences; use only genre/mood/energy")
    parser.add_argument("--plain", action="store_true", help="print plain text instead of a table")
    rag = parser.add_argument_group("retrieval (RAG) mode")
    rag.add_argument("--query", default=None, help="free-text request, e.g. 'music for a late night study session'")
    rag.add_argument("--genre", default=None, help="requested genre")
    rag.add_argument("--mood", default=None, help="requested mood")
    rag.add_argument("--energy", default=None, help="target energy, 0.0-1.0")
    rag.add_argument("--no-rag", action="store_true", help="turn retrieval off (original scoring rule only)")
    rag.add_argument("--compare", action="store_true", help="show baseline (no retrieval) and retrieval results side by side")
    rag.add_argument("--trace", default=None, metavar="FILE", help="save the pipeline's intermediate steps to FILE (markdown)")
    parser.add_argument("-k", type=int, default=5, help="number of recommendations (default: 5)")
    return parser.parse_args()


def run_rag_mode(args: argparse.Namespace) -> None:
    raw = {key: getattr(args, key) for key in ("genre", "mood", "energy") if getattr(args, key) is not None}
    if "energy" in raw:
        try:
            raw["energy"] = float(raw["energy"])
        except ValueError:
            pass  # left as a string so validation can explain the problem
    if args.query is not None:
        raw["text"] = args.query
    pipeline = Pipeline()
    modes = [False, True] if args.compare else [not args.no_rag]
    results = []
    for use_rag in modes:
        result = pipeline.run(raw, k=args.k, use_rag=use_rag)
        results.append(result)
        label = "WITH retrieval" if use_rag else "WITHOUT retrieval (baseline)"
        print(f"\n=== {label} ===")
        print(f"Request: {raw}")
        if not result.ok:
            print(f"REJECTED: {result.error}")
            continue
        print(f"Mode: {result.mode} | overall confidence: {result.confidence:.2f}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        print()
        for rank, rec in enumerate(result.recommendations, start=1):
            print(f"{rank}. {rec.song['title']} by {rec.song['artist']} - Score: {rec.score:.2f} - Confidence: {rec.confidence:.2f}")
            print(f"   Because: {rec.explanation if use_rag else rec.baseline_explanation}")
    if args.trace:
        from src.trace import save_trace
        save_trace(args.trace, raw, results[-1])
        print(f"\nTrace saved to {args.trace}")


def main() -> None:
    args = parse_args()
    if any(v is not None for v in (args.query, args.genre, args.mood, args.energy)):
        run_rag_mode(args)
        return
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    profiles = {n: p for n, p in PROFILES.items() if args.profile == "all" or args.profile.lower() in n.lower()}
    if not profiles:
        raise SystemExit(f"no profile matches '{args.profile}'; choose from {list(PROFILES)}")
    modes = sorted(STRATEGIES) if args.mode == "all" else [args.mode]

    for name, prefs in profiles.items():
        if args.basic:
            prefs = {key: prefs[key] for key in CORE_KEYS}
        for mode in modes:
            label = f"{name} | mode={mode} | diversity={'on' if args.diversity else 'off'}"
            print(f"\n=== {label} ===")
            print("Profile: " + ", ".join(f"{key}={value}" for key, value in prefs.items()) + "\n")
            results = recommend_songs(prefs, songs, k=args.k, strategy=get_strategy(mode), diversity=args.diversity)
            print(format_results(results, as_table=not args.plain))


if __name__ == "__main__":
    main()
