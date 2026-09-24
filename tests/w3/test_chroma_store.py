"""Tests for the ChromaStore adapter.

All tests use an ephemeral in-memory Chroma instance.
No network calls, no real data files.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load ChromaStore via importlib to avoid pytest namespace shadowing of the root
# ``w3`` package by the ``tests.w3`` test package.
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
    """Return a ChromaStore backed by ephemeral in-memory storage."""
    return ChromaStore()  # persist_directory=None → ephemeral


@pytest.fixture
def sample_documents():
    """Three tiny documents with hardcoded 384-dim vectors."""
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
    """Three random 384-dim vectors matching sample_documents count."""
    rng = np.random.default_rng(seed=42)
    return rng.normal(size=(3, 384)).astype(np.float32)


def _collection_name() -> str:
    """Return a unique collection name per test invocation."""
    import uuid
    return f"test_collection_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Tests: create_collection
# ---------------------------------------------------------------------------


class TestCreateCollection:
    def test_create_and_list_collection(self, store):
        name = _collection_name()
        store.create_collection(name)
        assert name in store.list_collections()

    def test_create_duplicate_collection(self, store):
        name = _collection_name()
        store.create_collection(name)
        # Second create with same name should not raise
        store.create_collection(name)
        assert name in store.list_collections()

    def test_create_multiple_collections(self, store):
        names = [_collection_name() for _ in range(3)]
        for name in names:
            store.create_collection(name)
        for name in names:
            assert name in store.list_collections()


# ---------------------------------------------------------------------------
# Tests: add_documents
# ---------------------------------------------------------------------------


class TestAddDocuments:
    def test_add_and_count(self, store, sample_documents, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_documents, sample_embeddings)
        assert store.count(name) == len(sample_documents)

    def test_add_empty_list_rejected(self, store):
        """ChromaDB does not accept empty embeddings lists — expected."""
        name = _collection_name()
        store.create_collection(name)
        with pytest.raises(ValueError):
            store.add_documents(name, [], np.empty((0, 384)))

    def test_add_to_nonexistent_collection_raises(self, store, sample_documents, sample_embeddings):
        with pytest.raises(ValueError, match="not found|does not exist|not exist"):
            store.add_documents("nonexistent", sample_documents, sample_embeddings)


# ---------------------------------------------------------------------------
# Tests: query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_returns_list_of_dicts(self, store, sample_documents, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_documents, sample_embeddings)

        # Use the first document's embedding as query
        query_emb = sample_embeddings[0].tolist()
        results = store.query(name, query_emb, top_k=3)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_query_returns_required_fields(self, store, sample_documents, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_documents, sample_embeddings)

        query_emb = sample_embeddings[0].tolist()
        results = store.query(name, query_emb, top_k=1)
        result = results[0]

        for field in ("id", "title", "chunk", "pubDate", "link"):
            assert field in result, f"Missing field '{field}' in result"

    def test_query_respects_top_k(self, store, sample_documents, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_documents, sample_embeddings)

        query_emb = sample_embeddings[0].tolist()
        for k in (1, 2, 3):
            results = store.query(name, query_emb, top_k=k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"

    def test_query_returns_only_dicts(self, store, sample_documents, sample_embeddings):
        """Verify all returned items are plain dictionaries, not Chroma-native objects."""
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_documents, sample_embeddings)

        query_emb = sample_embeddings[0].tolist()
        results = store.query(name, query_emb, top_k=1)

        assert all(isinstance(r, dict) for r in results)

    def test_query_empty_collection_returns_empty_list(self, store):
        name = _collection_name()
        store.create_collection(name)

        query_emb = np.zeros(384, dtype=np.float32).tolist()
        results = store.query(name, query_emb, top_k=5)
        assert results == []

    def test_query_nonexistent_collection_raises(self, store):
        query_emb = np.zeros(384, dtype=np.float32).tolist()
        with pytest.raises(ValueError, match="not found|does not exist|not exist"):
            store.query("nonexistent", query_emb, top_k=5)


# ---------------------------------------------------------------------------
# Tests: utility methods
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    def test_delete_collection(self, store):
        name = _collection_name()
        store.create_collection(name)
        assert name in store.list_collections()
        store.delete_collection(name)
        assert name not in store.list_collections()

    def test_delete_nonexistent_collection_raises(self, store):
        with pytest.raises(ValueError, match="not found|does not exist|not exist"):
            store.delete_collection("nonexistent")