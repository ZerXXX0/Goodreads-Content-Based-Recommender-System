from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .utils import default_data_path, project_path


@lru_cache(maxsize=1)
def load_books(books_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(books_path) if books_path is not None else default_data_path("books.csv")
    books = pd.read_csv(path, low_memory=False)
    columns = [
        "book_id",
        "goodreads_book_id",
        "work_id",
        "authors",
        "original_title",
        "title",
        "average_rating",
        "ratings_count",
    ]
    available = [column for column in columns if column in books.columns]
    return books[available].copy()


@lru_cache(maxsize=1)
def load_ratings(ratings_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(ratings_path) if ratings_path is not None else default_data_path("ratings.csv")
    return pd.read_csv(path, low_memory=False)


@lru_cache(maxsize=1)
def load_tags(book_tags_path: str | Path | None = None, tags_path: str | Path | None = None) -> pd.DataFrame:
    book_tags_file = Path(book_tags_path) if book_tags_path is not None else default_data_path("book_tags.csv")
    tags_file = Path(tags_path) if tags_path is not None else default_data_path("tags.csv")

    book_tags = pd.read_csv(book_tags_file, low_memory=False)
    tags = pd.read_csv(tags_file, low_memory=False)

    merged = book_tags.merge(tags, on="tag_id", how="left")
    merged["tag_name"] = merged["tag_name"].fillna("")
    return merged


def build_books_enriched(
    books: pd.DataFrame | None = None,
    book_tags: pd.DataFrame | None = None,
    top_tags: int = 5,
) -> pd.DataFrame:
    books_df = load_books() if books is None else books.copy()
    tags_df = load_tags() if book_tags is None else book_tags.copy()

    if "goodreads_book_id" not in books_df.columns:
        raise ValueError("books dataframe must contain goodreads_book_id")

    if not tags_df.empty:
        top_tag_strings = (
            tags_df.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
            .groupby("goodreads_book_id")["tag_name"]
            .apply(lambda values: " ".join([value for value in values.head(top_tags) if isinstance(value, str) and value.strip()]))
            .reset_index(name="tags")
        )
    else:
        top_tag_strings = pd.DataFrame(columns=["goodreads_book_id", "tags"])

    enriched = books_df.merge(top_tag_strings, on="goodreads_book_id", how="left")
    enriched["tags"] = enriched["tags"].fillna("")
    enriched["description"] = ""
    enriched["book_text"] = enriched.apply(
        lambda row: " ".join(
            part.strip()
            for part in [
                str(row.get("title", "")),
                str(row.get("authors", "")),
                str(row.get("description", "")),
                str(row.get("tags", "")),
            ]
            if part.strip()
        ),
        axis=1,
    )
    return enriched


def save_books_enriched(output_path: str | Path | None = None) -> Path:
    enriched = build_books_enriched()
    path = Path(output_path) if output_path is not None else project_path("books_enriched.csv")
    enriched.to_csv(path, index=False)
    return path
