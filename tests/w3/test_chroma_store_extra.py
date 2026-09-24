"""Tests for ChromaStore extra methods needed by retrieval functions.

All tests use an ephemeral in-memory Chroma instance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "w3.chroma_store",
    _PROJECT_ROOT / "w3" / "chroma_store.py",
)
_chroma_store_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _chroma_store_module
_spec.loader.exec_module(_chroma_store_module)
ChromaStore = _chroma_store_module.ChromaStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    return ChromaStore()


@pytest.fixture
def sample_documents():
    return [
        {
            "id": "doc1",
            "title": "Taylor Swift wins award",
            "chunk": "Taylor Swift won the Grammy for Best Album.",
            "pubDate": "2024-02-05",
            "link": "https://example.com/taylor",
        },
        {
            "id": "doc2",
            "title": "Economic growth in US",
            "chunk": "The US economy grew by 3.2% in Q4 2024.",
            "pubDate": "2024-12-15",
            "link": "https://example.com/economy",
        },
        {
            "id": "doc3",
            "title": "Brazil and France trade deal",
            "chunk": "Brazil and France signed a new trade agreement.",
            "pubDate": "2024-06-20",
            "link": "https://example.com/trade",
        },
    ]


@pytest.fixture
def sample_embeddings():
    rng = np.random.default_rng(seed=42)
    return rng.normal(size=(3, 384)).astype(np.float32)


def _collection_name() -> str:
    import uuid
    return f"test_collection_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Tests: get_all_documents
# ---------------------------------------------------------------------------


class TestGetAllDocuments:
    def test_get_all_returns_all_docs(self, store, sample_documents, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_documents, sample_embeddings)
        docs = store.get_all_documents(name)
        assert len(docs) == len(sample_documents)
        for doc in docs:
            assert isinstance(doc, dict)
            for field in ("id", "title", "chunk", "pubDate", "link"):
                assert field in doc

    def test_get_all_empty_collection(self, store):
        name = _collection_name()
        store.create_collection(name)
        docs = store.get_all_documents(name)
        assert docs == []

    def test_get_all_nonexistent_collection_raises(self, store):
        with pytest.raises(ValueError):
            store.get_all_documents("nonexistent")


# ---------------------------------------------------------------------------
# Tests: query_with_filter
# ---------------------------------------------------------------------------


class TestQueryWithFilter:
    def test_filter_by_title_exact_match(self, store, sample_documents, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_documents, sample_embeddings)
        # Filter by exact title match using $in
        query_emb = sample_embeddings[0].tolist()
        results = store.query_with_filter(
            name, query_emb, top_k=5,
            metadata_filter={"title": {"$in": ["Taylor Swift wins award"]}},
        )
        assert len(results) >= 1
        assert results[0]["title"] == "Taylor Swift wins award"

    def test_filter_by_title_set_match(self, store, sample_documents, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_documents, sample_embeddings)
        # Filter by multiple possible titles using $in
        query_emb = sample_embeddings[0].tolist()
        results = store.query_with_filter(
            name, query_emb, top_k=5,
            metadata_filter={"title": {"$in": ["Taylor Swift wins award", "Economic growth in US"]}},
        )
        assert len(results) >= 2
        titles = {r["title"] for r in results}
        assert "Taylor Swift wins award" in titles
        assert "Economic growth in US" in titles

    def test_filter_no_match_returns_empty(self, store, sample_documents, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_documents, sample_embeddings)
        query_emb = sample_embeddings[0].tolist()
        results = store.query_with_filter(
            name, query_emb, top_k=5,
            metadata_filter={"title": {"$in": ["NonexistentTitle"]}},
        )
        assert results == []
    def test_filter_nonexistent_collection_raises(self, store):
        query_emb = np.zeros(384, dtype=np.float32).tolist()
        with pytest.raises(ValueError):
            store.query_with_filter("nonexistent", query_emb, top_k=5, metadata_filter={})