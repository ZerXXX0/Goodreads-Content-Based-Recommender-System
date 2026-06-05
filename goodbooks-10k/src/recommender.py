from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import BookEmbedder, load_embeddings, save_embeddings
from .preprocessing import build_books_enriched, load_books, load_ratings


@dataclass
class RecommendationResult:
    recommendations: pd.DataFrame
    history: pd.DataFrame
    user_vector: np.ndarray | None


class ContentBasedRecommender:
    def __init__(self, positive_threshold: int = 3, device: str = "auto") -> None:
        self.positive_threshold = positive_threshold
        self.device = device
        self.books = pd.DataFrame()
        self.ratings = pd.DataFrame()
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.book_ids = np.array([], dtype=int)
        self.book_index = pd.Series(dtype=int)

    def fit(self, force_rebuild: bool = False) -> "ContentBasedRecommender":
        self.books = build_books_enriched(load_books())
        self.ratings = load_ratings()

        cached = None if force_rebuild else load_embeddings()
        if cached is None:
            embedder = BookEmbedder(device=self.device)
            texts = self.books["book_text"].fillna("").astype(str).tolist()
            self.embeddings = embedder.fit_transform(texts)
            self.book_ids = self.books["work_id"].to_numpy()
            save_embeddings(self.embeddings, self.book_ids)
        else:
            self.embeddings = cached.embeddings
            self.book_ids = cached.book_ids

        self.book_index = pd.Series(range(len(self.book_ids)), index=self.book_ids)
        return self

    def _get_book_row(self, book_id: int) -> pd.Series | None:
        if "work_id" not in self.books.columns:
            return None
        matches = self.books.loc[self.books["work_id"] == book_id]
        if matches.empty:
            return None
        return matches.iloc[0]

    def _book_vector(self, book_id: int) -> np.ndarray | None:
        if book_id not in self.book_index.index:
            return None
        return self.embeddings[int(self.book_index.loc[book_id])]

    def user_history(self, user_id: int) -> pd.DataFrame:
        positive = self.ratings.loc[(self.ratings["user_id"] == user_id) & (self.ratings["rating"] >= self.positive_threshold)]
        if positive.empty:
            return pd.DataFrame(columns=["book_id", "title", "authors", "rating"])

        history = positive.copy()
        book_lookup = self.books.set_index("work_id") if "work_id" in self.books.columns else pd.DataFrame()
        if not book_lookup.empty:
            history["title"] = history["book_id"].map(book_lookup["title"])
            history["authors"] = history["book_id"].map(book_lookup["authors"])
        else:
            history["title"] = ""
            history["authors"] = ""

        return history[["book_id", "title", "authors", "rating"]].sort_values(["rating", "title"], ascending=[False, True])

    def build_user_vector(self, user_id: int) -> np.ndarray | None:
        positive = self.ratings.loc[(self.ratings["user_id"] == user_id) & (self.ratings["rating"] >= self.positive_threshold)]
        if positive.empty:
            return None

        vectors = []
        weights = []
        for _, row in positive.iterrows():
            vector = self._book_vector(int(row["book_id"]))
            if vector is None:
                continue
            vectors.append(vector)
            weights.append(float(row["rating"]))

        if not vectors:
            return None

        stacked = np.vstack(vectors)
        weight_array = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        user_vector = (stacked * weight_array).sum(axis=0) / weight_array.sum()
        norm = np.linalg.norm(user_vector)
        return user_vector / norm if norm > 0 else user_vector

    def recommend(self, user_id: int, top_n: int = 10) -> RecommendationResult:
        user_vector = self.build_user_vector(user_id)
        history = self.user_history(user_id)

        if user_vector is None or self.embeddings.size == 0:
            empty = pd.DataFrame(columns=["rank", "book_id", "title", "authors", "similarity"])
            return RecommendationResult(recommendations=empty, history=history, user_vector=user_vector)

        seen_books = set(history["book_id"].dropna().astype(int).tolist())
        candidate_mask = np.array([book_id not in seen_books for book_id in self.book_ids], dtype=bool)
        candidate_embeddings = self.embeddings[candidate_mask]
        candidate_book_ids = self.book_ids[candidate_mask]

        similarities = cosine_similarity([user_vector], candidate_embeddings).ravel()
        ranking = np.argsort(-similarities)[:top_n]

        recommendation_rows = []
        for rank, index in enumerate(ranking, start=1):
            book_id = int(candidate_book_ids[index])
            row = self._get_book_row(book_id)
            recommendation_rows.append(
                {
                    "rank": rank,
                    "book_id": book_id,
                    "title": row["title"] if row is not None else "",
                    "authors": row["authors"] if row is not None else "",
                    "similarity": float(similarities[index]),
                }
            )

        recommendations = pd.DataFrame(recommendation_rows)
        return RecommendationResult(recommendations=recommendations, history=history, user_vector=user_vector)


@lru_cache(maxsize=1)
def get_recommender(force_rebuild: bool = False, device: str = "auto") -> ContentBasedRecommender:
    recommender = ContentBasedRecommender(device=device)
    return recommender.fit(force_rebuild=force_rebuild)
