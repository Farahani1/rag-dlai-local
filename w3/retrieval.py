"""Retrieval functions adapted from the original notebook.

These functions replace Weaviate-based calls with ChromaDB calls via the
``ChromaStore`` adapter.  Each function preserves the educational behavior
and signature style of the original notebook exercise.
"""

from __future__ import annotations

from typing import Any, Callable

from typing import TYPE_CHECKING

import bm25s
import numpy as np

if TYPE_CHECKING:
    from w3.chroma_store import ChromaStore


# ============================================================================
# Step 2: filter_by_metadata
# ============================================================================


def filter_by_metadata(
    metadata_property: str,
    values: list[str],
    store: ChromaStore,
    collection_name: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve objects from a Chroma collection based on metadata filtering.

    Parameters
    ----------
    metadata_property :
        The metadata field to filter on (e.g. ``"title"``, ``"chunk"``).
    values :
        Values to match against the property (uses ``$in`` — exact match).
    store :
        A ``ChromaStore`` instance.
    collection_name :
        Name of the Chroma collection.
    limit :
        Maximum number of objects to retrieve.  Defaults to 5.

    Returns
    -------
    list[dict]
        Each dict contains ``id``, ``title``, ``chunk``, ``pubDate``, ``link``.
    """
    # Use a dummy embedding (all zeros) since we only filter, not search by similarity
    dummy_embedding = np.zeros(384, dtype=np.float32).tolist()
    results = store.query_with_filter(
        collection_name,
        query_embedding=dummy_embedding,
        top_k=limit,
        metadata_filter={metadata_property: {"$in": values}},
    )
    return results


# ============================================================================
# Step 3: semantic_search_retrieve
# ============================================================================


def semantic_search_retrieve(
    query: str,
    store: ChromaStore,
    collection_name: str,
    embed_function: Callable[[str], list[float]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Perform a semantic (vector) search on a Chroma collection.

    Parameters
    ----------
    query :
        The search query.
    store :
        A ``ChromaStore`` instance.
    collection_name :
        Name of the Chroma collection.
    embed_function :
        A callable that takes a string and returns a list of floats (embedding).
    top_k :
        Number of top relevant objects to retrieve.  Defaults to 5.

    Returns
    -------
    list[dict]
        Each dict contains ``id``, ``title``, ``chunk``, ``pubDate``, ``link``.
    """
    query_embedding = embed_function(query)
    results = store.query(collection_name, query_embedding, top_k=top_k)
    return results


# ============================================================================
# Step 4: bm25_retrieve
# ============================================================================


# Global BM25 index cache: (collection_name,) -> (bm25s.BM25, list[dict])
_BM25_CACHE: dict[str, tuple[bm25s.BM25, list[dict]]] = {}


def _build_bm25_index(
    collection_name: str,
    store: ChromaStore,
) -> tuple[bm25s.BM25 | None, list[dict]]:
    """Build (or retrieve from cache) a BM25 index for a collection.

    Returns ``(None, [])`` for an empty collection.
    """
    cache_key = collection_name
    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key]

    documents = store.get_all_documents(collection_name)
    if not documents:
        _BM25_CACHE[cache_key] = (None, [])
        return None, []

    corpus = [doc.get("chunk", "") for doc in documents]

    bm25 = bm25s.BM25()
    tokenized_corpus = bm25s.tokenize(corpus)
    bm25.index(tokenized_corpus)

    _BM25_CACHE[cache_key] = (bm25, documents)
    return bm25, documents


def clear_bm25_cache() -> None:
    """Clear the BM25 index cache.  Call when the collection changes."""
    _BM25_CACHE.clear()


def bm25_retrieve(
    query: str,
    store: ChromaStore,
    collection_name: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Perform a BM25 keyword search on a Chroma collection.

    Uses the ``bm25s`` library for indexing and scoring.  The index is
    built from all documents in the collection and cached for reuse.

    Parameters
    ----------
    query :
        The search query.
    store :
        A ``ChromaStore`` instance.
    collection_name :
        Name of the Chroma collection.
    top_k :
        Number of top relevant objects to retrieve.  Defaults to 5.

    Returns
    -------
    list[dict]
        Each dict contains ``id``, ``title``, ``chunk``, ``pubDate``, ``link``.
    """
    bm25, documents = _build_bm25_index(collection_name, store)
    if bm25 is None:
        return []

    tokenized_query = bm25s.tokenize([query])
    results_indices, _ = bm25.retrieve(tokenized_query, k=top_k)
    # results_indices[0] is the list of indices for the first (only) query
    indices = results_indices[0].tolist()

    return [documents[i] for i in indices if i < len(documents)]


# ============================================================================
# Step 5: hybrid_retrieve
# ============================================================================


def _safe_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1-D array to [0, 1]."""
    if scores.size == 0:
        return scores
    s_min, s_max = scores.min(), scores.max()
    if s_max == s_min:
        return np.ones_like(scores) * 0.5
    return (scores - s_min) / (s_max - s_min)


def hybrid_retrieve(
    query: str,
    store: ChromaStore,
    collection_name: str,
    embed_function: Callable[[str], list[float]],
    alpha: float = 0.5,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Perform a hybrid search combining semantic and BM25 scores.

    Parameters
    ----------
    query :
        The search query.
    store :
        A ``ChromaStore`` instance.
    collection_name :
        Name of the Chroma collection.
    embed_function :
        A callable that takes a string and returns a list of floats (embedding).
    alpha :
        Weight between semantic (1.0) and BM25 (0.0).  Defaults to 0.5.
    top_k :
        Number of top relevant objects to retrieve.  Defaults to 5.

    Returns
    -------
    list[dict]
        Each dict contains ``id``, ``title``, ``chunk``, ``pubDate``, ``link``.
    """
    # Get all documents
    all_docs = store.get_all_documents(collection_name)
    if not all_docs:
        return []

    # BM25 scores for all docs
    bm25, _ = _build_bm25_index(collection_name, store)
    tokenized_query = bm25s.tokenize([query])
    # Retrieve all docs to get scores
    results_indices, scores = bm25.retrieve(tokenized_query, k=len(all_docs))
    indices = results_indices[0].tolist()
    bm25_scores_raw = scores[0] if scores is not None else np.ones(len(all_docs))

    # Build full-length BM25 score array
    bm25_full = np.zeros(len(all_docs), dtype=np.float32)
    for i, idx in enumerate(indices):
        if i < len(bm25_scores_raw) and idx < len(all_docs):
            bm25_full[idx] = bm25_scores_raw[i]

    # Semantic scores: compute cosine similarity via Chroma query for all docs
    query_embedding = embed_function(query)
    semantic_results = store.query(collection_name, query_embedding, top_k=len(all_docs))
    semantic_indices = {doc["id"]: i for i, doc in enumerate(all_docs)}

    semantic_scores = np.zeros(len(all_docs), dtype=np.float32)
    for rank, doc in enumerate(semantic_results):
        idx = semantic_indices.get(doc["id"])
        if idx is not None:
            semantic_scores[idx] = 1.0 / (1.0 + rank)  # rank-based score

    # Normalize both score arrays
    bm25_norm = _safe_normalize(bm25_full)
    semantic_norm = _safe_normalize(semantic_scores)

    # Combined score
    combined = alpha * semantic_norm + (1.0 - alpha) * bm25_norm

    # Sort by combined score descending
    sorted_indices = np.argsort(combined)[::-1][:top_k]
    return [all_docs[i] for i in sorted_indices]


# ============================================================================
# Step 6: semantic_search_with_reranking (simple local reranker)
# ============================================================================


def _local_rerank(
    query: str,
    documents: list[dict[str, Any]],
    rerank_property: str,
) -> list[dict[str, Any]]:
    """Re-rank documents by how many query tokens appear in ``rerank_property``.

    This is a simple local reranker that does NOT require an external model.
    It scores each document by the count of query tokens (lowercased) found
    in the target property field.
    """
    query_tokens = set(query.lower().split())
    scored = []
    for doc in documents:
        field_text = doc.get(rerank_property, "").lower()
        score = sum(1 for token in query_tokens if token in field_text)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]


def semantic_search_with_reranking(
    query: str,
    rerank_property: str,
    store: ChromaStore,
    collection_name: str,
    embed_function: Callable[[str], list[float]],
    rerank_query: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Perform a semantic search and re-rank results by a specified property.

    Parameters
    ----------
    query :
        The search query.
    rerank_property :
        The metadata field to use for re-ranking (e.g. ``"title"``, ``"chunk"``).
    store :
        A ``ChromaStore`` instance.
    collection_name :
        Name of the Chroma collection.
    embed_function :
        A callable that takes a string and returns a list of floats (embedding).
    rerank_query :
        The query to use for re-ranking.  If ``None``, uses the original query.
    top_k :
        Number of top results to return.  Defaults to 5.

    Returns
    -------
    list[dict]
        Each dict contains ``id``, ``title``, ``chunk``, ``pubDate``, ``link``.
    """
    if rerank_query is None:
        rerank_query = query

    # Fetch more candidates for re-ranking
    fetch_k = max(top_k * 2, 10)
    query_embedding = embed_function(query)
    candidates = store.query(collection_name, query_embedding, top_k=fetch_k)

    if not candidates:
        return []

    reranked = _local_rerank(rerank_query, candidates, rerank_property)
    return reranked[:top_k]


# ============================================================================
# Step 7: generate_final_prompt
# ============================================================================


def generate_final_prompt(
    query: str,
    top_k: int,
    retrieve_function: Callable[..., list[dict[str, Any]]],
    store: ChromaStore,
    collection_name: str,
    embed_function: Callable[[str], list[float]],
    rerank_query: str | None = None,
    rerank_property: str | None = None,
    use_rerank: bool = False,
    use_rag: bool = True,
) -> str:
    """Generate a final prompt by optionally retrieving and formatting documents.

    Parameters
    ----------
    query :
        The user query.
    top_k :
        Number of top documents to retrieve.
    retrieve_function :
        One of the retrieval functions (e.g. ``semantic_search_retrieve``).
    store :
        A ``ChromaStore`` instance.
    collection_name :
        Name of the Chroma collection.
    embed_function :
        A callable that takes a string and returns a list of floats (embedding).
    rerank_query :
        Query to use for re-ranking.  Defaults to ``None``.
    rerank_property :
        Property to re-rank by.  Required if ``use_rerank`` is ``True``.
    use_rerank :
        Whether to use re-ranking.  Defaults to ``False``.
    use_rag :
        Whether to use retrieval-augmented generation.  Defaults to ``True``.

    Returns
    -------
    str
        The constructed prompt.
    """
    # If no rag, return the query
    if not use_rag:
        return query

    if use_rerank:
        if rerank_property is None:
            raise ValueError("rerank_property must be set if use_rerank = True")
        top_k_documents = retrieve_function(
            query=query,
            top_k=top_k,
            store=store,
            collection_name=collection_name,
            embed_function=embed_function,
            rerank_property=rerank_property,
            rerank_query=rerank_query,
        )
    else:
        top_k_documents = retrieve_function(
            query=query,
            top_k=top_k,
            store=store,
            collection_name=collection_name,
            embed_function=embed_function,
        )

    # Format the documents
    formatted_data = ""
    for document in top_k_documents:
        document_layout = (
            f"Title: {document['title']}, Chunk: {document['chunk']}, "
            f"Published at: {document['pubDate']}\nURL: {document['link']}"
        )
        formatted_data += document_layout + "\n"

    retrieve_data_formatted = formatted_data
    prompt = (
        f"Answer the user query below. There will be provided additional information for you to compose your answer. "
        f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
        f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
        f"The news data is ordered by relevance."
        f"Query: {query}\n"
        f"2024 News: {retrieve_data_formatted}"
    )

    return prompt


# ============================================================================
# Step 8: llm_call
# ============================================================================


def llm_call(
    query: str,
    retrieve_function: Callable[..., list[dict[str, Any]]] | None = None,
    store: ChromaStore | None = None,
    collection_name: str | None = None,
    embed_function: Callable[[str], list[float]] | None = None,
    top_k: int = 5,
    use_rag: bool = True,
    use_rerank: bool = False,
    rerank_property: str | None = None,
    rerank_query: str | None = None,
    llm_backend: Callable[[str], str] | None = None,
) -> str:
    """Simulate a call to a language model with RAG.

    Parameters
    ----------
    query :
        The user query.
    retrieve_function :
        One of the retrieval functions.
    store :
        A ``ChromaStore`` instance.
    collection_name :
        Name of the Chroma collection.
    embed_function :
        A callable that takes a string and returns a list of floats (embedding).
    top_k :
        Number of top documents to retrieve.  Defaults to 5.
    use_rag :
        Whether to use retrieval-augmented generation.  Defaults to ``True``.
    use_rerank :
        Whether to apply re-ranking.  Defaults to ``False``.
    rerank_property :
        Property to re-rank by.  Required if ``use_rerank`` is ``True``.
    rerank_query :
        Query to use for re-ranking.  Defaults to ``None``.
    llm_backend :
        A callable that takes a prompt string and returns a response string.
        If ``None``, returns the generated prompt without calling an LLM.

    Returns
    -------
    str
        The generated response.
    """
    PROMPT = generate_final_prompt(
        query,
        top_k=top_k,
        retrieve_function=retrieve_function,
        store=store,
        collection_name=collection_name,
        embed_function=embed_function,
        use_rag=use_rag,
        use_rerank=use_rerank,
        rerank_property=rerank_property,
        rerank_query=rerank_query,
    )

    if llm_backend is not None:
        return llm_backend(PROMPT)

    return PROMPT