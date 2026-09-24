"""Tests for the adapted W4 retrieval functions.

All tests use an ephemeral in-memory Chroma instance with tiny synthetic data.
No network calls, no real data files, no model inference.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Load retrieval module with same importlib trick as test_chroma_store
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_spec_retrieval = importlib.util.spec_from_file_location(
    "w4.retrieval",
    _PROJECT_ROOT / "w4" / "retrieval.py",
)
_retrieval_module = importlib.util.module_from_spec(_spec_retrieval)  # type: ignore[arg-type]
sys.modules[_spec_retrieval.name] = _retrieval_module  # type: ignore[union-attr]
_spec_retrieval.loader.exec_module(_retrieval_module)  # type: ignore[union-attr]

_spec_store = importlib.util.spec_from_file_location(
    "w4.chroma_store",
    _PROJECT_ROOT / "w4" / "chroma_store.py",
)
_store_module = importlib.util.module_from_spec(_spec_store)  # type: ignore[arg-type]
sys.modules[_spec_store.name] = _store_module  # type: ignore[union-attr]
_spec_store.loader.exec_module(_store_module)  # type: ignore[union-attr]

ChromaStore = _store_module.ChromaStore

# Get all retrieval functions
filter_by_metadata = _retrieval_module.filter_by_metadata
semantic_search_retrieve = _retrieval_module.semantic_search_retrieve
bm25_retrieve = _retrieval_module.bm25_retrieve
clear_bm25_cache = _retrieval_module.clear_bm25_cache
hybrid_retrieve = _retrieval_module.hybrid_retrieve
semantic_search_with_reranking = _retrieval_module.semantic_search_with_reranking
generate_final_prompt = _retrieval_module.generate_final_prompt
llm_call = _retrieval_module.llm_call


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear BM25 cache before each test."""
    clear_bm25_cache()
    yield


@pytest.fixture
def store():
    return ChromaStore()


@pytest.fixture
def collection_name():
    import uuid

    return f"test_retrieval_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_documents():
    return [
        {
            "id": "doc1",
            "title": "Taylor Swift wins Grammy",
            "chunk": "Taylor Swift won the Grammy for Best Album at the 2024 awards ceremony.",
            "pubDate": "2024-02-05",
            "link": "https://example.com/taylor-grammy",
        },
        {
            "id": "doc2",
            "title": "US economic growth report",
            "chunk": "The US economy grew by 3.2 percent in the fourth quarter of 2024 according to new data.",
            "pubDate": "2024-12-15",
            "link": "https://example.com/us-economy",
        },
        {
            "id": "doc3",
            "title": "Brazil signs trade deal with France",
            "chunk": "Brazil and France signed a comprehensive new trade agreement covering agriculture and technology.",
            "pubDate": "2024-06-20",
            "link": "https://example.com/brazil-france-trade",
        },
    ]


@pytest.fixture
def sample_embeddings():
    rng = np.random.default_rng(seed=42)
    return rng.normal(size=(3, 384)).astype(np.float32)


def _populate(
    store: ChromaStore,
    name: str,
    docs: list[dict[str, Any]],
    embs: np.ndarray,
) -> None:
    store.create_collection(name)
    store.add_documents(name, docs, embs)


def _stub_embed(text: str) -> list[float]:
    """A stub embed function that returns the same vector for any text."""
    return np.zeros(384, dtype=np.float32).tolist()


# ============================================================================
# Step 2: filter_by_metadata
# ============================================================================


class TestFilterByMetadata:
    def test_filter_by_exact_title(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = filter_by_metadata(
            "title",
            ["Taylor Swift wins Grammy"],
            store,
            collection_name,
            limit=5,
        )
        assert len(results) >= 1
        assert results[0]["title"] == "Taylor Swift wins Grammy"

    def test_filter_returns_multiple(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = filter_by_metadata(
            "title",
            ["Taylor Swift wins Grammy", "US economic growth report"],
            store,
            collection_name,
            limit=5,
        )
        assert len(results) >= 2

    def test_filter_no_match(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = filter_by_metadata(
            "title",
            ["Nonexistent Title"],
            store,
            collection_name,
            limit=5,
        )
        assert results == []

    def test_filter_respects_limit(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = filter_by_metadata(
            "title",
            [
                "Taylor Swift wins Grammy",
                "US economic growth report",
                "Brazil signs trade deal with France",
            ],
            store,
            collection_name,
            limit=2,
        )
        assert len(results) <= 2

    def test_filter_returns_dicts_with_required_fields(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = filter_by_metadata(
            "title",
            ["Taylor Swift wins Grammy"],
            store,
            collection_name,
            limit=5,
        )
        assert len(results) >= 1
        for field in ("id", "title", "chunk", "pubDate", "link"):
            assert field in results[0]


# ============================================================================
# Step 3: semantic_search_retrieve
# ============================================================================


class TestSemanticSearchRetrieve:
    def test_returns_list_of_dicts(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = semantic_search_retrieve(
            "music",
            store,
            collection_name,
            _stub_embed,
            top_k=2,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_respects_top_k(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        for k in (1, 2, 3):
            results = semantic_search_retrieve(
                "test",
                store,
                collection_name,
                _stub_embed,
                top_k=k,
            )
            assert len(results) == k, f"Expected {k}, got {len(results)}"

    def test_required_fields_present(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = semantic_search_retrieve(
            "test",
            store,
            collection_name,
            _stub_embed,
            top_k=1,
        )
        for field in ("id", "title", "chunk", "pubDate", "link"):
            assert field in results[0]

    def test_empty_collection_returns_empty(self, store, collection_name):
        store.create_collection(collection_name)
        results = semantic_search_retrieve(
            "test",
            store,
            collection_name,
            _stub_embed,
            top_k=5,
        )
        assert results == []


# ============================================================================
# Step 4: bm25_retrieve
# ============================================================================


class TestBm25Retrieve:
    def test_returns_list_of_dicts(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = bm25_retrieve("economy", store, collection_name, top_k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_respects_top_k(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        for k in (1, 2, 3):
            results = bm25_retrieve("economy", store, collection_name, top_k=k)
            assert len(results) == k

    def test_keyword_relevance(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        # Query about "economy" should rank doc2 (economy) higher
        results = bm25_retrieve("economy", store, collection_name, top_k=3)
        titles = [r["title"] for r in results]
        assert "US economic growth report" in titles

    def test_required_fields_present(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = bm25_retrieve("test", store, collection_name, top_k=1)
        for field in ("id", "title", "chunk", "pubDate", "link"):
            assert field in results[0]

    def test_empty_collection(self, store, collection_name):
        store.create_collection(collection_name)
        results = bm25_retrieve("test", store, collection_name, top_k=5)
        assert results == []


# ============================================================================
# Step 5: hybrid_retrieve
# ============================================================================


class TestHybridRetrieve:
    def test_returns_list_of_dicts(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = hybrid_retrieve(
            "economy",
            store,
            collection_name,
            _stub_embed,
            alpha=0.5,
            top_k=2,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_respects_top_k(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        for k in (1, 2, 3):
            results = hybrid_retrieve(
                "test",
                store,
                collection_name,
                _stub_embed,
                alpha=0.5,
                top_k=k,
            )
            assert len(results) == k

    def test_empty_collection(self, store, collection_name):
        store.create_collection(collection_name)
        results = hybrid_retrieve(
            "test",
            store,
            collection_name,
            _stub_embed,
            alpha=0.5,
            top_k=5,
        )
        assert results == []

    def test_required_fields_present(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = hybrid_retrieve(
            "test",
            store,
            collection_name,
            _stub_embed,
            alpha=0.5,
            top_k=1,
        )
        for field in ("id", "title", "chunk", "pubDate", "link"):
            assert field in results[0]


# ============================================================================
# Step 6: semantic_search_with_reranking
# ============================================================================


class TestSemanticSearchWithReranking:
    def test_returns_list_of_dicts(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = semantic_search_with_reranking(
            "test", "title", store, collection_name, _stub_embed, top_k=2,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_respects_top_k(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        results = semantic_search_with_reranking(
            "test", "title", store, collection_name, _stub_embed, top_k=1,
        )
        assert len(results) == 1

    def test_rerank_by_chunk(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        # Query about "Grammy" should make doc1 rank higher when reranking by chunk
        results = semantic_search_with_reranking(
            "Grammy awards ceremony",
            "chunk",
            store,
            collection_name,
            _stub_embed,
            top_k=3,
        )
        assert len(results) >= 1
        assert results[0]["title"] == "Taylor Swift wins Grammy"

    def test_rerank_query_differs(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        # Use a different rerank query
        results = semantic_search_with_reranking(
            "music",
            "title",
            store,
            collection_name,
            _stub_embed,
            rerank_query="trade agreement",
            top_k=3,
        )
        assert len(results) >= 1

    def test_empty_collection(self, store, collection_name):
        store.create_collection(collection_name)
        results = semantic_search_with_reranking(
            "test", "title", store, collection_name, _stub_embed, top_k=5,
        )
        assert results == []


# ============================================================================
# Step 7: generate_final_prompt
# ============================================================================


class TestGenerateFinalPrompt:
    def test_no_rag_returns_query(self, store, collection_name):
        result = generate_final_prompt(
            "hello world",
            top_k=5,
            retrieve_function=semantic_search_retrieve,
            store=store,
            collection_name=collection_name,
            embed_function=_stub_embed,
            use_rag=False,
        )
        assert result == "hello world"

    def test_use_rerank_without_property_raises(self, store, collection_name):
        with pytest.raises(ValueError, match="rerank_property must be set"):
            generate_final_prompt(
                "test",
                top_k=5,
                retrieve_function=semantic_search_retrieve,
                store=store,
                collection_name=collection_name,
                embed_function=_stub_embed,
                use_rag=True,
                use_rerank=True,
                rerank_property=None,
            )

    def test_rag_formats_documents(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        result = generate_final_prompt(
            "what is the economic situation",
            top_k=2,
            retrieve_function=semantic_search_retrieve,
            store=store,
            collection_name=collection_name,
            embed_function=_stub_embed,
            use_rag=True,
        )
        assert "Title:" in result
        assert "Chunk:" in result
        assert "Published at:" in result
        assert "URL:" in result

    def test_prompt_contains_query(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        result = generate_final_prompt(
            "my specific query here",
            top_k=1,
            retrieve_function=semantic_search_retrieve,
            store=store,
            collection_name=collection_name,
            embed_function=_stub_embed,
            use_rag=True,
        )
        assert "my specific query here" in result


# ============================================================================
# Step 8: llm_call
# ============================================================================


class TestLlmCall:
    def test_returns_prompt_when_no_backend(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)
        result = llm_call(
            "test query",
            retrieve_function=semantic_search_retrieve,
            store=store,
            collection_name=collection_name,
            embed_function=_stub_embed,
            top_k=1,
            use_rag=True,
        )
        assert isinstance(result, str)
        assert "test query" in result

    def test_uses_llm_backend(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)

        def fake_backend(prompt: str) -> str:
            return f"LLM response for: {prompt[:50]}"

        result = llm_call(
            "test",
            retrieve_function=semantic_search_retrieve,
            store=store,
            collection_name=collection_name,
            embed_function=_stub_embed,
            top_k=1,
            use_rag=True,
            llm_backend=fake_backend,
        )
        assert "LLM response for:" in result

    def test_no_rag_passes_through(
        self, store, collection_name, sample_documents, sample_embeddings
    ):
        _populate(store, collection_name, sample_documents, sample_embeddings)

        def fake_backend(prompt: str) -> str:
            return f"Response: {prompt}"

        result = llm_call(
            "direct query",
            retrieve_function=semantic_search_retrieve,
            store=store,
            collection_name=collection_name,
            embed_function=_stub_embed,
            use_rag=False,
            llm_backend=fake_backend,
        )
        assert "Response: direct query" in result