"""Shared FAISS vector store for all RAG strategies."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = embeddings.astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class VectorStore:
    """Small wrapper around a normalized FAISS inner-product index."""

    def __init__(
        self,
        embedding_model_name: str,
        model: SentenceTransformer,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        index: faiss.Index,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.model = model
        self.chunks = chunks
        self.embeddings = embeddings
        self.index = index

    @property
    def dimension(self) -> int:
        return self.index.d

    @property
    def size(self) -> int:
        return self.index.ntotal

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        if not query.strip() or self.size == 0:
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embedding = normalize_embeddings(query_embedding)
        limit = min(top_k, self.size)
        scores, indices = self.index.search(query_embedding, limit)

        results = []
        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0:
                continue
            chunk = self.chunks[int(index_id)]
            results.append(
                {
                    **chunk,
                    "score": float(score),
                    "rank": len(results) + 1,
                }
            )
        return results


def build_vector_store(chunks: list[dict[str, Any]], embedding_model_name: str) -> VectorStore:
    if not chunks:
        raise ValueError("Cannot build a vector store without chunks.")

    model = load_embedding_model(embedding_model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embeddings = normalize_embeddings(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return VectorStore(
        embedding_model_name=embedding_model_name,
        model=model,
        chunks=chunks,
        embeddings=embeddings,
        index=index,
    )
