from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import ensure_models_dir, project_path


@dataclass
class EmbeddingArtifacts:
    embeddings: np.ndarray
    book_ids: np.ndarray


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BookEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "auto") -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.model = None
        self.vectorizer = None

    def _load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        return SentenceTransformer(self.model_name)

    def fit(self, texts: list[str]) -> "BookEmbedder":
        self.model = self._load_sentence_transformer()
        if self.model is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            self.vectorizer.fit(texts)
        else:
            self.model.to(self.device)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )
            return np.asarray(embeddings, dtype=np.float32)

        if self.vectorizer is None:
            raise RuntimeError("Embedder must be fitted before encoding texts")

        matrix = self.vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float32)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return dense / norms

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.encode(texts)


def save_embeddings(
    embeddings: np.ndarray,
    book_ids: np.ndarray,
    output_dir: str | Path | None = None,
) -> EmbeddingArtifacts:
    models_dir = ensure_models_dir() if output_dir is None else Path(output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    np.save(models_dir / "embeddings.npy", embeddings)
    np.save(models_dir / "book_ids.npy", book_ids)
    return EmbeddingArtifacts(embeddings=embeddings, book_ids=book_ids)


def load_embeddings(output_dir: str | Path | None = None) -> EmbeddingArtifacts | None:
    models_dir = project_path("models") if output_dir is None else Path(output_dir)
    embeddings_path = models_dir / "embeddings.npy"
    book_ids_path = models_dir / "book_ids.npy"
    if not embeddings_path.exists() or not book_ids_path.exists():
        return None
    return EmbeddingArtifacts(embeddings=np.load(embeddings_path), book_ids=np.load(book_ids_path))
