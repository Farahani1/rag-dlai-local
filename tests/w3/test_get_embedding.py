"""Stub-only tests for the ``get_embedding`` function.

All tests use mocks/fixtures — no real model inference, no network calls,
no actual data files. These tests define the expected contract.

All tests will **skip** until ``get_embedding`` is implemented in
``w3/embedding.py``. This is by design — TDD red phase.

Run:
    python -m pytest tests/w3/test_get_embedding.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load the ``w3.embedding`` module via importlib to avoid pytest namespace
# shadowing of the root ``w3`` package by the ``tests.w3`` test package.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_W3_DIR = _PROJECT_ROOT / "w3"

_spec = importlib.util.spec_from_file_location(
    "w3.embedding",
    _W3_DIR / "embedding.py",
)
_embedding_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _embedding_module
try:
    _spec.loader.exec_module(_embedding_module)
except Exception:
    _embedding_module = None  # TDD red phase — expected


def _get_get_embedding():
    """Return ``get_embedding`` or skip if not yet implemented."""
    if _embedding_module is None or not hasattr(_embedding_module, "get_embedding"):
        pytest.skip(
            "get_embedding not yet implemented in w3/embedding.py "
            "(TDD red phase — expected)"
        )
    return _embedding_module.get_embedding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_texts() -> list[str]:
    """Two short sample texts for embedding."""
    return [
        "Taylor Swift wins Grammy for Best Album",
        "US economy grew by 3.2 percent in Q4 2024",
    ]


@pytest.fixture
def dummy_embeddings() -> np.ndarray:
    """Fixed (2, 384) float32 array matching ``dummy_texts`` count."""
    rng = np.random.default_rng(seed=42)
    return rng.normal(size=(2, 384)).astype(np.float32)


@pytest.fixture
def single_text() -> list[str]:
    """Single-element list for testing edge cases."""
    return ["Hello world"]


# ============================================================================
# 3.1 Cache hit
# ============================================================================


class TestGetEmbeddingCacheHit:
    """Cache file exists → load from ``joblib``, skip backend."""

    def test_cache_hit_loads_from_joblib(
        self, dummy_texts, dummy_embeddings,
    ):
        """When a valid cache file exists, ``get_embedding`` loads it."""
        get_embedding = _get_get_embedding()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "SentenceTransformer") as mock_st,
            patch.object(_embedding_module, "requests") as mock_requests,
        ):
            mock_os.path.exists.return_value = True
            mock_joblib.load.return_value = dummy_embeddings

            result = get_embedding(dummy_texts)

            np.testing.assert_array_equal(result, dummy_embeddings)
            mock_joblib.load.assert_called_once()
            # Backends must NOT be called on cache hit
            mock_st.assert_not_called()
            mock_requests.post.assert_not_called()

    def test_cache_hit_returns_correct_shape(
        self, dummy_texts, dummy_embeddings,
    ):
        """Loaded embedding array has shape ``(N, 384)``."""
        get_embedding = _get_get_embedding()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "SentenceTransformer"),
            patch.object(_embedding_module, "requests"),
        ):
            mock_os.path.exists.return_value = True
            mock_joblib.load.return_value = dummy_embeddings

            result = get_embedding(dummy_texts)
            assert result.shape == (2, 384)

    def test_cache_hit_returns_correct_dtype(
        self, dummy_texts, dummy_embeddings,
    ):
        """Loaded embedding array is ``np.float32``."""
        get_embedding = _get_get_embedding()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "SentenceTransformer"),
            patch.object(_embedding_module, "requests"),
        ):
            mock_os.path.exists.return_value = True
            mock_joblib.load.return_value = dummy_embeddings

            result = get_embedding(dummy_texts)
            assert result.dtype == np.float32


# ============================================================================
# 3.2 Cache miss — sentence-transformers backend
# ============================================================================


class TestGetEmbeddingCacheMissSentenceTransformers:
    """Cache miss → compute with sentence-transformers, save, return."""

    def test_cache_miss_computes_with_st(
        self, dummy_texts, dummy_embeddings,
    ):
        """When cache is missing, calls ``SentenceTransformer.encode``."""
        get_embedding = _get_get_embedding()

        fake_model = MagicMock()
        fake_model.encode.return_value = dummy_embeddings

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "SentenceTransformer", return_value=fake_model) as mock_st_cls,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "requests"),
        ):
            mock_os.path.exists.return_value = False

            result = get_embedding(dummy_texts, backend="sentence-transformers")

            np.testing.assert_array_equal(result, dummy_embeddings)
            mock_st_cls.assert_called_once()
            fake_model.encode.assert_called_once()
            mock_joblib.dump.assert_called_once()

    def test_cache_miss_saves_joblib(
        self, dummy_texts, dummy_embeddings,
    ):
        """Result is saved via ``joblib.dump`` after computing."""
        get_embedding = _get_get_embedding()

        fake_model = MagicMock()
        fake_model.encode.return_value = dummy_embeddings

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "SentenceTransformer", return_value=fake_model),
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "requests"),
        ):
            mock_os.path.exists.return_value = False

            get_embedding(dummy_texts, backend="sentence-transformers")
            mock_joblib.dump.assert_called_once()

    def test_second_call_uses_cache(
        self, dummy_texts, dummy_embeddings,
    ):
        """Second call with same texts loads from cache, does not re-compute."""
        get_embedding = _get_get_embedding()

        fake_model = MagicMock()
        fake_model.encode.return_value = dummy_embeddings

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "SentenceTransformer", return_value=fake_model) as mock_st_cls,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "requests"),
        ):
            # First call: cache miss (exists=False) → compute + save
            mock_os.path.exists.return_value = False
            result1 = get_embedding(dummy_texts, backend="sentence-transformers")
            np.testing.assert_array_equal(result1, dummy_embeddings)

            # Now simulate cache hit for second call
            mock_os.path.exists.return_value = True
            mock_joblib.load.return_value = dummy_embeddings

            # Second call: cache exists → load
            result2 = get_embedding(dummy_texts, backend="sentence-transformers")

            np.testing.assert_array_equal(result1, result2)
            # SentenceTransformer should have been called only once (first call)
            assert mock_st_cls.call_count <= 1


# ============================================================================
# 3.3 Cache miss — ollama backend
# ============================================================================


class TestGetEmbeddingCacheMissOllama:
    """Cache miss → compute with ollama API, save, return."""

    def test_cache_miss_computes_with_ollama(
        self, dummy_texts,
    ):
        """When backend is ``\"ollama\"``, calls ollama embedding API."""
        get_embedding = _get_get_embedding()
        fake_response = MagicMock()
        fake_response.json.return_value = {"embedding": [0.1] * 384}
        fake_response.raise_for_status = MagicMock()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "requests") as mock_requests,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "SentenceTransformer"),
        ):
            mock_os.path.exists.return_value = False
            mock_requests.post.return_value = fake_response

            result = get_embedding(dummy_texts, backend="ollama")

            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 384)
            assert result.dtype == np.float32
            # A batch of 2 texts → 2 API calls (or one batch call)
            assert mock_requests.post.call_count >= 2
            mock_joblib.dump.assert_called_once()

    def test_ollama_saves_to_cache(
        self, dummy_texts,
    ):
        """Ollama result is saved via ``joblib.dump`` after computing."""
        get_embedding = _get_get_embedding()
        fake_response = MagicMock()
        fake_response.json.return_value = {"embedding": [0.1] * 384}
        fake_response.raise_for_status = MagicMock()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "requests") as mock_requests,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "SentenceTransformer"),
        ):
            mock_os.path.exists.return_value = False
            mock_requests.post.return_value = fake_response

            get_embedding(dummy_texts, backend="ollama")
            mock_joblib.dump.assert_called_once()


# ============================================================================
# 3.4 Default cache directory
# ============================================================================


class TestGetEmbeddingDefaultPath:
    """Cache directory defaults to ``w3/``."""

    def test_default_cache_dir_is_w3(self, tmp_path, monkeypatch, dummy_texts, dummy_embeddings):
        """When ``cache_dir=None``, uses ``PROJECT_ROOT / \"w3\"``."""
        get_embedding = _get_get_embedding()

        # Point PROJECT_ROOT to tmp_path via monkeypatch
        monkeypatch.setattr(_embedding_module, "PROJECT_ROOT", tmp_path)

        # ``os`` stays real so the directory is actually created under
        # tmp_path; the cache misses and the (mocked) model is used.
        fake_model = MagicMock()
        fake_model.encode.return_value = dummy_embeddings
        with (
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "_get_model_no_cache", return_value=fake_model),
            patch.object(_embedding_module, "requests"),
        ):
            result = get_embedding(dummy_texts)

            np.testing.assert_array_equal(result, dummy_embeddings)
            expected_cache_parent = tmp_path / "w3"
            assert expected_cache_parent.is_dir(), (
                f"Expected cache dir {expected_cache_parent} to have been created"
            )
            cache_file = Path(mock_joblib.dump.call_args.args[1])
            assert cache_file.parent == expected_cache_parent

    def test_custom_cache_dir_respected(self, tmp_path, monkeypatch, dummy_texts, dummy_embeddings):
        """When ``cache_dir`` is provided, uses that instead of default."""
        get_embedding = _get_get_embedding()

        custom_dir = tmp_path / "my_cache"
        monkeypatch.setattr(_embedding_module, "PROJECT_ROOT", tmp_path)

        fake_model = MagicMock()
        fake_model.encode.return_value = dummy_embeddings
        with (
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "_get_model_no_cache", return_value=fake_model),
            patch.object(_embedding_module, "requests"),
        ):
            result = get_embedding(dummy_texts, cache_dir=str(custom_dir))

            np.testing.assert_array_equal(result, dummy_embeddings)
            assert custom_dir.is_dir(), (
                f"Expected custom cache dir {custom_dir} to have been created"
            )
            cache_file = Path(mock_joblib.dump.call_args.args[1])
            assert cache_file.parent == custom_dir
            assert not (tmp_path / "w3").exists()


# ============================================================================
# 3.5 Backend selection
# ============================================================================


class TestGetEmbeddingBackendSelection:
    """Backend selection logic."""

    def test_explicit_backend_overrides_config(self, dummy_texts):
        """``backend=\"ollama\"`` uses ollama even if config says ST."""
        get_embedding = _get_get_embedding()
        fake_response = MagicMock()
        fake_response.json.return_value = {"embedding": [0.1] * 384}
        fake_response.raise_for_status = MagicMock()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "requests") as mock_requests,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "SentenceTransformer"),
        ):
            mock_os.path.exists.return_value = False
            mock_requests.post.return_value = fake_response

            get_embedding(dummy_texts, backend="ollama")
            # Must have called ollama (requests.post), not ST
            assert mock_requests.post.call_count >= 1

    def test_invalid_backend_raises(self, dummy_texts):
        """Unknown backend string raises ``ValueError``."""
        get_embedding = _get_get_embedding()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "SentenceTransformer"),
            patch.object(_embedding_module, "requests"),
            patch.object(_embedding_module, "joblib"),
        ):
            mock_os.path.exists.return_value = False
            with pytest.raises(ValueError, match="backend|unknown|not supported"):
                get_embedding(dummy_texts, backend="invalid_backend_name")


# ============================================================================
# 3.6 Cache key determinism
# ============================================================================


class TestGetEmbeddingCacheKey:
    """Cache filename is deterministic and depends on inputs."""

    def test_same_texts_same_model_produce_same_key(self, tmp_path, monkeypatch, dummy_texts):
        """Identical inputs → identical cache filename."""
        get_embedding = _get_get_embedding()
        monkeypatch.setattr(_embedding_module, "PROJECT_ROOT", tmp_path)

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "SentenceTransformer") as mock_st,
            patch.object(_embedding_module, "requests"),
        ):
            fake_model = MagicMock()
            fake_model.encode.return_value = np.zeros((2, 384), dtype=np.float32)
            mock_st.return_value = fake_model

            # First call: cache miss (exists=False) → compute
            mock_os.path.exists.return_value = False
            get_embedding(dummy_texts, backend="sentence-transformers")

            # Second call: cache hit (exists=True) → load
            mock_os.path.exists.return_value = True
            mock_joblib.load.return_value = np.zeros((2, 384), dtype=np.float32)
            get_embedding(dummy_texts, backend="sentence-transformers")

            # Both joblib.load and joblib.dump should have been called once
            mock_joblib.dump.assert_called_once()
            mock_joblib.load.assert_called_once()

    def test_different_texts_produce_different_key(self, tmp_path, monkeypatch):
        """Different texts → different cache file (second call is a miss)."""
        get_embedding = _get_get_embedding()
        texts_a = ["Hello world"]
        texts_b = ["Different text"]
        monkeypatch.setattr(_embedding_module, "PROJECT_ROOT", tmp_path)

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "joblib") as mock_joblib,
            patch.object(_embedding_module, "SentenceTransformer") as mock_st,
            patch.object(_embedding_module, "requests"),
        ):
            fake_model = MagicMock()
            fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
            mock_st.return_value = fake_model
            mock_os.path.exists.return_value = False

            get_embedding(texts_a, backend="sentence-transformers")
            get_embedding(texts_b, backend="sentence-transformers")

            # Two different calls → two joblib.dump calls (two different cache files)
            assert mock_joblib.dump.call_count == 2


# ============================================================================
# 3.7 Edge cases
# ============================================================================


class TestGetEmbeddingEdgeCases:
    """Edge-case inputs."""

    def test_empty_texts_list(self):
        """``texts=[]`` returns empty ``(0, 384)`` array."""
        get_embedding = _get_get_embedding()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "joblib"),
            patch.object(_embedding_module, "SentenceTransformer"),
            patch.object(_embedding_module, "requests"),
        ):
            mock_os.path.exists.return_value = False

            result = get_embedding([], backend="sentence-transformers")
            assert result.shape == (0, 384)
            assert result.dtype == np.float32

    def test_single_text(self, single_text):
        """Single text returns ``(1, 384)`` array."""
        get_embedding = _get_get_embedding()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "SentenceTransformer") as mock_st,
            patch.object(_embedding_module, "joblib"),
            patch.object(_embedding_module, "requests"),
        ):
            mock_os.path.exists.return_value = False
            fake_model = MagicMock()
            fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
            mock_st.return_value = fake_model

            result = get_embedding(single_text, backend="sentence-transformers")
            assert result.shape == (1, 384)
            assert result.dtype == np.float32

    def test_invalid_backend_raises(self, dummy_texts):
        """Unknown backend raises ``ValueError``."""
        get_embedding = _get_get_embedding()

        with (
            patch.object(_embedding_module, "os") as mock_os,
            patch.object(_embedding_module, "SentenceTransformer"),
            patch.object(_embedding_module, "requests"),
            patch.object(_embedding_module, "joblib"),
        ):
            mock_os.path.exists.return_value = False
            with pytest.raises(ValueError, match="backend|unknown|not supported"):
                get_embedding(dummy_texts, backend="not_a_real_backend")