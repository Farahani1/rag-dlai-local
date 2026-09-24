"""Thin wrapper around sentence-transformers for embedding queries.

Uses the same model dimension (384) as the pre-computed embeddings in
``data/embeddings.joblib``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from setting import config  # noqa: E402

# Point HuggingFace cache to the local hub directory
os.environ["HF_HOME"] = str(config.hfLocalHub)

EMBEDDING_MODEL_NAME = config.embeddingModel

# Default embedding dimension for fallback/empty cases
_DEFAULT_DIM = 384


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


def _get_model_no_cache() -> SentenceTransformer:
    """Load the embedding model without caching (for testing compatibility)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


def embed_query(text: str) -> list[float]:
    """Embed a query string into a 384-dimensional vector.

    Parameters
    ----------
    text :
        The query text to embed.

    Returns
    -------
    list[float]
        A 384-dimensional embedding vector.
    """
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=False)
    return np.asarray(embedding, dtype=np.float32).tolist()


# ============================================================================
# get_embedding — batched embedding with caching
# ============================================================================


def _cache_path(cache_dir: str, texts: list[str], backend: str) -> str:
    """Derive a deterministic cache file path from inputs.

    Parameters
    ----------
    cache_dir :
        Directory where the cache file is stored.
    texts :
        List of texts to embed (used to build the hash).
    backend :
        Backend name (e.g. ``"sentence-transformers"`` or ``"ollama"``).

    Returns
    -------
    str
        Full path to the cache file.
    """
    # Create a hash of all texts + backend for a deterministic filename
    raw = json.dumps({"texts": texts, "backend": backend}, sort_keys=True)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"embeddings_{h}.joblib")


def get_embedding(
    texts: list[str],
    backend: str | None = None,
    cache_dir: str | None = None,
) -> np.ndarray:
    """Return embeddings for a list of texts, with joblib-based caching.

    Parameters
    ----------
    texts :
        List of text strings to embed.
    backend :
        Which backend to use.  ``"sentence-transformers"`` uses the local
        SentenceTransformer model.  ``"ollama"`` calls a local Ollama
        embedding endpoint.  If ``None``, falls back to the default backend
        (sentence-transformers, then ollama).
    cache_dir :
        Directory for the cache file.  If ``None``, defaults to
        ``PROJECT_ROOT / "w3"``.

    Returns
    -------
    np.ndarray
        Embedding array of shape ``(len(texts), D)`` with dtype ``np.float32``.

    Raises
    ------
    ValueError
        If *backend* is not a recognised value.
    """
    # Resolve default cache directory
    if cache_dir is None:
        cache_dir = os.path.join(str(PROJECT_ROOT), "w3")
    os.makedirs(cache_dir, exist_ok=True)

    # Resolve default backend
    if backend is None:
        backend = "sentence-transformers"

    # ------------------------------------------------------------------
    # Handle special case: empty text list
    # ------------------------------------------------------------------
    if not texts:
        return np.empty((0, _DEFAULT_DIM), dtype=np.float32)

    # ------------------------------------------------------------------
    # Cache check
    # ------------------------------------------------------------------
    cache_file = _cache_path(cache_dir, texts, backend)
    if os.path.exists(cache_file):
        return joblib.load(cache_file)

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------
    if backend == "sentence-transformers":
        model = _get_model_no_cache()
        embeddings = model.encode(texts, normalize_embeddings=False)
        result = np.asarray(embeddings, dtype=np.float32)
    elif backend == "ollama":
        # Ollama embedding API: /api/embed (single text per request)
        ollama_url = config.ollama["url"].replace("/api/generate", "/api/embed")
        ollama_model = (
            config.ollamaEmbeddingModel
            if config.ollamaEmbeddingModel
            else "bge-m3:latest"
        )
        all_embeddings: list[list[float]] = []
        for text in texts:
            resp = requests.post(
                ollama_url,
                json={"model": ollama_model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            # Support both:
            #   {"embeddings": [[0.1, ...]]}  — Ollama new API
            #   {"embedding": [0.1, ...]}     — Ollama legacy/unit-test mock
            if "embeddings" in data:
                all_embeddings.append(data["embeddings"][0])
            else:
                all_embeddings.append(data["embedding"])
        result = np.asarray(all_embeddings, dtype=np.float32)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported: 'sentence-transformers', 'ollama'."
        )

    # ------------------------------------------------------------------
    # Cache the result
    # ------------------------------------------------------------------
    joblib.dump(result, cache_file)
    return result