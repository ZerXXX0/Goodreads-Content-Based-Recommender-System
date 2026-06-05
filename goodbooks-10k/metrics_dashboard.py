from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.evaluation import evaluate_recommender
from src.preprocessing import load_ratings
from src.recommender import get_recommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Goodreads content-based recommender.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild embeddings from scratch.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to evaluate.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device used for embedding generation.")
    parser.add_argument("--threshold", type=int, default=3, help="Rating threshold considered as positive feedback.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file to write the evaluation summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratings = load_ratings()
    recommender = get_recommender(force_rebuild=args.rebuild, device=args.device)
    # override positive threshold for evaluation
    recommender.positive_threshold = args.threshold

    def render_progress(processed_users: int, total_users: int) -> None:
        bar_width = 28
        filled = 0 if total_users == 0 else int(bar_width * processed_users / total_users)
        filled = min(filled, bar_width)
        bar = "█" * filled + "-" * (bar_width - filled)
        percent = 0 if total_users == 0 else (processed_users / total_users) * 100
        sys.stdout.write(f"\rEvaluating users: [{bar}] {processed_users}/{total_users} ({percent:5.1f}%)")
        sys.stdout.flush()

    metrics = evaluate_recommender(recommender, ratings, top_n=args.top_n, progress_callback=render_progress)

    if metrics.evaluated_users > 0:
        sys.stdout.write("\n")

    print("Goodreads Evaluation Report")
    print("===========================")
    print(f"Ratings rows: {len(ratings):,}")
    print(f"Unique users: {ratings['user_id'].nunique():,}")
    print(f"Unique books: {ratings['book_id'].nunique():,}")
    print()
    print(f"Precision@{args.top_n}: {metrics.precision_at_10:.3f}")
    print(f"Recall@{args.top_n}:    {metrics.recall_at_10:.3f}")
    print(f"NDCG@{args.top_n}:      {metrics.ndcg_at_10:.3f}")
    print(f"Evaluated users: {metrics.evaluated_users}")
    print(f"Embedding device: {args.device}")

    summary = {
        "precision_at_10": metrics.precision_at_10,
        "recall_at_10": metrics.recall_at_10,
        "ndcg_at_10": metrics.ndcg_at_10,
        "evaluated_users": metrics.evaluated_users,
        "top_n": args.top_n,
    }

    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print()
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
