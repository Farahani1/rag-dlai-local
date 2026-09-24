"""Tests for the W5 ChromaStore adapter.

Unlike ``w4/chroma_store.py`` (fixed ``title/chunk/pubDate/link`` schema),
``w5/chroma_store.py`` stores arbitrary metadata dicts — these tests exercise
that generic behavior with both product-shaped and FAQ-shaped documents.

All tests use an ephemeral in-memory Chroma instance. No network calls, no
real data files.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "w5.chroma_store",
    _PROJECT_ROOT / "w5" / "chroma_store.py",
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
    return ChromaStore()  # persist_directory=None -> ephemeral


@pytest.fixture
def sample_products():
    """Three tiny products with the full W5 metadata schema."""
    return [
        {
            "id": "1",
            "product_id": 1,
            "productDisplayName": "Blue T-Shirt",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "T-Shirts",
            "baseColour": "Blue",
            "season": "Summer",
            "usage": "Casual",
            "gender": "Men",
            "year": 2020.0,
            "price": 50,
        },
        {
            "id": "2",
            "product_id": 2,
            "productDisplayName": "Red Dress",
            "masterCategory": "Apparel",
            "subCategory": "Dress",
            "articleType": "Dresses",
            "baseColour": "Red",
            "season": "Summer",
            "usage": "Formal",
            "gender": "Women",
            "year": 2021.0,
            "price": 120,
        },
        {
            "id": "3",
            "product_id": 3,
            "productDisplayName": "Black Sneakers",
            "masterCategory": "Footwear",
            "subCategory": "Shoes",
            "articleType": "Sneakers",
            "baseColour": "Black",
            "season": "All seasons",
            "usage": "Sports",
            "gender": "Unisex",
            "year": 2022.0,
            "price": 80,
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Three random 384-dim vectors matching sample_products count."""
    rng = np.random.default_rng(seed=42)
    return rng.normal(size=(3, 384)).astype(np.float32)


@pytest.fixture
def sample_faq():
    return [
        {"id": "0", "question": "What is your return policy?", "answer": "30 days.", "type": "returns"},
        {"id": "1", "question": "How can I contact support?", "answer": "Email us.", "type": "support"},
    ]


@pytest.fixture
def faq_embeddings():
    rng = np.random.default_rng(seed=7)
    return rng.normal(size=(2, 384)).astype(np.float32)


def _collection_name() -> str:
    import uuid

    return f"test_collection_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Tests: create_collection / list_collections
# ---------------------------------------------------------------------------


class TestCreateCollection:
    def test_create_and_list_collection(self, store):
        name = _collection_name()
        store.create_collection(name)
        assert name in store.list_collections()

    def test_create_duplicate_collection(self, store):
        name = _collection_name()
        store.create_collection(name)
        store.create_collection(name)
        assert name in store.list_collections()

    def test_two_collections_coexist(self, store):
        name_a, name_b = _collection_name(), _collection_name()
        store.create_collection(name_a)
        store.create_collection(name_b)
        assert name_a in store.list_collections()
        assert name_b in store.list_collections()


# ---------------------------------------------------------------------------
# Tests: add_documents with arbitrary metadata (products schema)
# ---------------------------------------------------------------------------


class TestAddDocumentsGenericMetadata:
    def test_add_and_count(self, store, sample_products, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_products, sample_embeddings)
        assert store.count(name) == len(sample_products)

    def test_query_returns_full_product_schema(self, store, sample_products, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_products, sample_embeddings)

        query_emb = sample_embeddings[0].tolist()
        results = store.query(name, query_emb, top_k=1)
        result = results[0]

        for field in (
            "id", "product_id", "productDisplayName", "masterCategory",
            "baseColour", "gender", "articleType", "subCategory", "season",
            "usage", "year", "price",
        ):
            assert field in result, f"Missing field '{field}' in result"

    def test_add_to_nonexistent_collection_raises(self, store, sample_products, sample_embeddings):
        with pytest.raises(ValueError, match="not found|does not exist|not exist"):
            store.add_documents("nonexistent", sample_products, sample_embeddings)


class TestAddDocumentsFaqSchema:
    """Different metadata shape (question/answer/type) — proves genericity."""

    def test_faq_documents_round_trip(self, store, sample_faq, faq_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_faq, faq_embeddings)

        results = store.get_all_documents(name)
        assert len(results) == 2
        questions = {r["question"] for r in results}
        assert questions == {"What is your return policy?", "How can I contact support?"}
        for r in results:
            assert "answer" in r
            assert "type" in r
            # Product-only fields must NOT leak into FAQ metadata.
            assert "productDisplayName" not in r


# ---------------------------------------------------------------------------
# Tests: query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_returns_list_of_dicts(self, store, sample_products, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_products, sample_embeddings)

        query_emb = sample_embeddings[0].tolist()
        results = store.query(name, query_emb, top_k=3)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_query_respects_top_k(self, store, sample_products, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_products, sample_embeddings)

        query_emb = sample_embeddings[0].tolist()
        for k in (1, 2, 3):
            results = store.query(name, query_emb, top_k=k)
            assert len(results) == k

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
# Tests: query_with_filter (metadata filtering — needed for W5's non-simplified branch)
# ---------------------------------------------------------------------------


class TestQueryWithFilter:
    def test_filter_by_base_colour(self, store, sample_products, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_products, sample_embeddings)

        query_emb = np.zeros(384, dtype=np.float32).tolist()
        results = store.query_with_filter(
            name, query_emb, top_k=5, metadata_filter={"baseColour": {"$in": ["Red"]}}
        )
        assert len(results) == 1
        assert results[0]["baseColour"] == "Red"

    def test_filter_price_range(self, store, sample_products, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_products, sample_embeddings)

        query_emb = np.zeros(384, dtype=np.float32).tolist()
        results = store.query_with_filter(
            name,
            query_emb,
            top_k=5,
            metadata_filter={"$and": [{"price": {"$gte": 60}}, {"price": {"$lte": 100}}]},
        )
        assert len(results) == 1
        assert results[0]["productDisplayName"] == "Black Sneakers"

    def test_filter_no_match(self, store, sample_products, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_products, sample_embeddings)

        query_emb = np.zeros(384, dtype=np.float32).tolist()
        results = store.query_with_filter(
            name, query_emb, top_k=5, metadata_filter={"baseColour": {"$in": ["Purple"]}}
        )
        assert results == []

    def test_filter_empty_collection(self, store):
        name = _collection_name()
        store.create_collection(name)

        query_emb = np.zeros(384, dtype=np.float32).tolist()
        results = store.query_with_filter(name, query_emb, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: get_all_documents
# ---------------------------------------------------------------------------


class TestGetAllDocuments:
    def test_returns_all_docs(self, store, sample_products, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        store.add_documents(name, sample_products, sample_embeddings)

        docs = store.get_all_documents(name)
        assert len(docs) == 3
        assert all(isinstance(d, dict) for d in docs)

    def test_empty_collection_returns_empty(self, store):
        name = _collection_name()
        store.create_collection(name)
        assert store.get_all_documents(name) == []

    def test_nonexistent_collection_raises(self, store):
        with pytest.raises(ValueError, match="not found|does not exist|not exist"):
            store.get_all_documents("nonexistent")


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

    def test_count(self, store, sample_products, sample_embeddings):
        name = _collection_name()
        store.create_collection(name)
        assert store.count(name) == 0
        store.add_documents(name, sample_products, sample_embeddings)
        assert store.count(name) == 3
