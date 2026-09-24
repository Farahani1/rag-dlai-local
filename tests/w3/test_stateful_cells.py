"""Smoke tests for the Chroma data-loading pipeline with ``adapted=True``.

These tests verify that the notebook cell sequence can load real data from
``news_data_dedup.csv`` and populate a Chroma collection.  They intentionally
**skip** cells that trigger model inference (embedding, BM25, LLM) — those
functions are tested in-depth in ``test_retrieval.py`` with synthetic fixtures.

All tests in this module skip when ``data/news_data_dedup.csv`` is missing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "C1M3_Assignment_stateful",
    _PROJECT_ROOT / "w3" / "C1M3_Assignment_stateful.py",
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

NotebookState = _module.NotebookState
run_until = _module.run_until

DATA_DIR = _PROJECT_ROOT / "data"
REQUIRES_DATA = pytest.mark.skipif(
    not (DATA_DIR / "news_data_dedup.csv").exists(),
    reason="Requires data/news_data_dedup.csv for integration test",
)


@pytest.fixture(autouse=True)
def _isolated_chroma_path(tmp_path, monkeypatch):
    """Point cell 8's ``ChromaStore(config.chromaPath)`` at a temp directory,
    so the tests never write to the project's ``data/chroma_db``."""
    import setting

    monkeypatch.setattr(setting.config, "chromaPath", tmp_path / "chroma_db")


def _make_state() -> NotebookState:
    """Create a fresh NotebookState with adapted mode enabled."""
    state = NotebookState()
    state.namespace["adapted"] = True
    state.namespace["namespace"] = state.namespace
    return state


@REQUIRES_DATA
class TestChromaDataPipeline:
    """Verify the data-loading cells work correctly with Chroma."""

    def test_setup_loads_and_populates_collection(self):
        """Cells 04→08→10→13: imports, creates store, loads CSV+embeddings,
        transforms rows, populates Chroma collection."""
        state = _make_state()
        # Skip cell 5 (flask_app/weaviate_server imports)
        run_until(13, state=state, skip_cells={5}, adapted=True)

        assert state.store is not None
        assert state.collection is not None
        count = state.store.count(state.collection)
        assert count == 870, f"Expected 870 documents, got {count}"

    def test_cell_14_prints_count(self):
        """Cell 14 prints the collection count without error."""
        state = _make_state()
        run_until(14, state=state, skip_cells={5}, adapted=True)
        assert state.collection is not None

    def test_cell_16_inspects_first_document(self):
        """Cell 16 fetches and prints the first document's properties."""
        state = _make_state()
        run_until(16, state=state, skip_cells={5}, adapted=True)
        assert state.collection is not None