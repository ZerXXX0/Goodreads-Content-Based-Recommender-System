from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .recommender import ContentBasedRecommender


@dataclass
class EvaluationSummary:
    precision_at_10: float
    recall_at_10: float
    ndcg_at_10: float
    evaluated_users: int


def train_test_split_user_positives(
    ratings: pd.DataFrame,
    positive_threshold: int = 3,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = ratings.loc[ratings["rating"] >= positive_threshold].copy()
    train_rows = []
    test_rows = []

    for _, user_group in positives.groupby("user_id"):
        if len(user_group) < 2:
            train_rows.append(user_group)
            continue
        shuffled = user_group.sample(frac=1.0, random_state=42)
        test_size = max(1, int(round(len(shuffled) * test_ratio)))
        test_rows.append(shuffled.iloc[:test_size])
        train_rows.append(shuffled.iloc[test_size:])

    train = pd.concat(train_rows, ignore_index=True) if train_rows else positives.iloc[0:0].copy()
    test = pd.concat(test_rows, ignore_index=True) if test_rows else positives.iloc[0:0].copy()
    return train, test


def ndcg_at_k(recommended_book_ids: list[int], relevant_book_ids: set[int], k: int = 10) -> float:
    score = 0.0
    for index, book_id in enumerate(recommended_book_ids[:k], start=1):
        if book_id in relevant_book_ids:
            score += 1.0 / np.log2(index + 1)

    ideal_hits = min(len(relevant_book_ids), k)
    if ideal_hits == 0:
        return 0.0
    ideal_score = sum(1.0 / np.log2(index + 1) for index in range(1, ideal_hits + 1))
    return score / ideal_score if ideal_score > 0 else 0.0


def evaluate_recommender(
    recommender: ContentBasedRecommender,
    ratings: pd.DataFrame,
    top_n: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvaluationSummary:
    original_ratings = recommender.ratings
    train, test = train_test_split_user_positives(ratings, positive_threshold=recommender.positive_threshold)
    recommender.ratings = train.copy()

    try:
        precisions = []
        recalls = []
        ndcgs = []
        total_users = test["user_id"].nunique()
        processed_users = 0

        for user_id, user_test in test.groupby("user_id"):
            relevant_ids = set(user_test["book_id"].astype(int).tolist())
            recommendations = recommender.recommend(int(user_id), top_n=top_n).recommendations
            recommended_ids = recommendations["book_id"].astype(int).tolist() if not recommendations.empty else []

            hits = len(relevant_ids.intersection(recommended_ids))
            precisions.append(hits / top_n)
            recalls.append(hits / max(1, len(relevant_ids)))
            ndcgs.append(ndcg_at_k(recommended_ids, relevant_ids, k=top_n))

            processed_users += 1
            if progress_callback is not None:
                progress_callback(processed_users, total_users)

        if not precisions:
            return EvaluationSummary(0.0, 0.0, 0.0, 0)

        return EvaluationSummary(
            precision_at_10=float(np.mean(precisions)),
            recall_at_10=float(np.mean(recalls)),
            ndcg_at_10=float(np.mean(ndcgs)),
            evaluated_users=len(precisions),
        )
    finally:
        recommender.ratings = original_ratings
