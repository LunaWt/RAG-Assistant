"""Deterministic test doubles shared across test modules."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class FakeEmbeddingModel:
    """Small deterministic embedder with controllable semantic directions."""

    _terms = ("california", "neural", "apple", "weather")

    def encode(
        self, texts: Iterable[str], normalize_embeddings: bool = True
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.array(
                [text.lower().count(term) for term in self._terms], dtype=float
            )
            if not vector.any():
                vector[0] = 1.0
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.asarray(vectors, dtype=float)
