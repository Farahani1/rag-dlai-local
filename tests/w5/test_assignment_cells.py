"""Tests for extracted pure cell functions, plus a stateful-runner regression test.

The pure-function tests exercise ``tests/w5/assignment_functions.py`` (hand
extracted copies of the corresponding cell bodies in
``w5/C1M5_Assignment.py``).

The regression test at the bottom runs the full notebook cell sequence
(``C1M5_Assignment_stateful.run_until``) with ``adapted=True`` and every
Ollama/Chroma/embedding call mocked, asserting the whole pipeline executes
without error and produces sane state. It lets the compatibility of future changes be assessed quickly.
Nothing equivalent exists for W4 (its stateful module has no test
coverage anywhere in the repo).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.w5.assignment_functions import (
    generate_faq_layout,
    generate_items_context,
    get_filter_by_metadata,
    parse_json_output,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Tests: generate_faq_layout (cell_42)
# ============================================================================


class TestGenerateFaqLayout:
    def test_single_faq(self):
        faqs = [{"question": "How to return?", "answer": "Visit our returns page.", "type": "returns"}]
        result = generate_faq_layout(faqs)
        assert "Question: How to return?" in result
        assert "Answer: Visit our returns page." in result
        assert "Type: returns" in result

    def test_multiple_faqs(self):
        faqs = [
            {"question": "Q1", "answer": "A1", "type": "t1"},
            {"question": "Q2", "answer": "A2", "type": "t2"},
        ]
        result = generate_faq_layout(faqs)
        assert "Q1" in result
        assert "Q2" in result

    def test_empty_list_returns_empty_string(self):
        assert generate_faq_layout([]) == ""

    def test_each_entry_ends_with_newline(self):
        faqs = [{"question": "Q", "answer": "A", "type": "t"}]
        assert generate_faq_layout(faqs).endswith("\n")


# ============================================================================
# Tests: parse_json_output (cell_91 adapted)
# ============================================================================


class TestParseJsonOutput:
    def test_parse_valid_json(self):
        assert parse_json_output('{"key": "value"}') == {"key": "value"}

    def test_parse_with_newlines(self):
        assert parse_json_output('{\n"key": "value"\n}') == {"key": "value"}

    def test_parse_double_braces_cleaned(self):
        assert parse_json_output('{{"key": "value"}}') == {"key": "value"}

    def test_parse_invalid_json_returns_none(self):
        assert parse_json_output("not json at all") is None

    def test_parse_empty_string_returns_none(self):
        assert parse_json_output("") is None


# ============================================================================
# Tests: get_filter_by_metadata (cell_91 adapted — list shape)
# ============================================================================


class TestGetFilterByMetadata:
    def test_none_input_returns_none(self):
        assert get_filter_by_metadata(None) is None

    def test_empty_dict_returns_none(self):
        assert get_filter_by_metadata({}) is None

    def test_returns_a_list(self):
        result = get_filter_by_metadata({"gender": ["Men"]})
        assert isinstance(result, list)
        assert result == [{"gender": {"$in": ["Men"]}}]

    def test_scalar_value_wrapped_in_list(self):
        result = get_filter_by_metadata({"usage": "Casual"})
        assert result == [{"usage": {"$in": ["Casual"]}}]

    def test_price_range_produces_two_conditions(self):
        result = get_filter_by_metadata({"price": {"min": 50, "max": 200}})
        assert result == [{"price": {"$gte": 50}}, {"price": {"$lte": 200}}]

    def test_price_with_min_zero_skipped(self):
        assert get_filter_by_metadata({"price": {"min": 0, "max": 200}}) is None

    def test_price_with_max_inf_skipped(self):
        assert get_filter_by_metadata({"price": {"min": 50, "max": "inf"}}) is None

    def test_multiple_keys_each_own_condition(self):
        result = get_filter_by_metadata({"gender": ["Men"], "baseColour": ["Blue"]})
        assert len(result) == 2
        keys = {next(iter(c)) for c in result}
        assert keys == {"gender", "baseColour"}

    def test_unknown_keys_skipped(self):
        assert get_filter_by_metadata({"unknown_field": ["value"]}) is None

    def test_mixed_valid_and_invalid_keys(self):
        result = get_filter_by_metadata({"gender": ["Men"], "unknown_field": ["value"]})
        assert result == [{"gender": {"$in": ["Men"]}}]

    def test_all_valid_keys_recognized(self):
        for key in ("gender", "masterCategory", "articleType", "baseColour", "usage", "season"):
            result = get_filter_by_metadata({key: ["Test"]})
            assert result is not None, f"Key '{key}' should produce a valid filter"


# ============================================================================
# Tests: generate_items_context (cell_104 adapted)
# ============================================================================


class TestGenerateItemsContext:
    def _product(self, **overrides):
        base = {
            "product_id": 123,
            "productDisplayName": "Blue T-Shirt",
            "masterCategory": "Apparel",
            "usage": "Casual",
            "gender": "Men",
            "articleType": "T-Shirts",
            "subCategory": "Topwear",
            "baseColour": "Blue",
            "season": "Summer",
            "year": 2024,
        }
        base.update(overrides)
        return base

    def test_single_product(self):
        result = generate_items_context([self._product()])
        assert "Product ID: 123" in result
        assert "Product name: Blue T-Shirt" in result
        assert "Apparel" in result
        assert "Blue" in result

    def test_multiple_products(self):
        products = [self._product(product_id=1), self._product(product_id=2)]
        result = generate_items_context(products)
        assert result.count("Product ID:") == 2

    def test_empty_list_returns_empty_string(self):
        assert generate_items_context([]) == ""


# ============================================================================
# Regression test: full cell-sequence smoke run via the stateful runner
# ============================================================================


def _fake_llm_response(prompt: str, **kwargs):
    """A context-sensitive fake LLM so the routing cells produce coherent labels."""
    text = prompt if isinstance(prompt, str) else str(prompt)
    if "FAQ or Product" in text or "FAQ or a product related" in text:
        content = "Product"
    elif "creative" in text.lower() and "technical" in text.lower():
        content = "technical"
    elif "vector database" in text.lower() and "JSON" in text:
        content = '{"gender": ["Men"], "price": {"min": 0, "max": "inf"}}'
    else:
        content = "Here is a response. Product ID: 1."
    return {
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def w5_stateful_module(monkeypatch):
    """Import C1M5_Assignment_stateful with setting/embedding fully mocked.

    Mirrors the ``sys.modules`` stubbing technique from
    ``tests/w4/test_utils.py``, extended to also stub ``embedding`` (avoids
    loading a real SentenceTransformer model) and to patch
    ``utils.generate_with_single_input`` before any cell imports it.
    """
    mock_config = SimpleNamespace(
        ollama={"modelName": "test-model", "url": "http://localhost:11434/api/generate"},
        completionOptions={"temperature": 0.7, "maxTokens": 500},
        clothesData=_PROJECT_ROOT / "data" / "clothes_json.joblib",
        faqData=_PROJECT_ROOT / "data" / "faq.joblib",
        productsChromaPath=_PROJECT_ROOT / "data" / "chroma_db_products",
        embeddingModel="sentence-transformers/all-MiniLM-L6-v2",
    )
    setting_mod = ModuleType("setting")
    setting_mod.config = mock_config

    embedding_mod = ModuleType("embedding")
    embedding_mod.embed_query = MagicMock(return_value=[0.1] * 384)

    # These bare names ("utils", "chroma_store", ...) collide with every
    # other week's identically-named module. Save whatever is currently
    # cached under each so it can be restored verbatim afterward --
    # otherwise this fixture would permanently redirect e.g. `import utils`
    # in *other* test files (w1-w4) to w5's utils.py for the rest of the
    # pytest session.
    _BARE_NAMES = ("setting", "embedding", "utils", "chroma_store", "C1M5_Assignment", "C1M5_Assignment_stateful")
    _originals = {name: sys.modules.get(name) for name in _BARE_NAMES}
    original_sys_path = list(sys.path)

    sys.modules["setting"] = setting_mod
    sys.modules["embedding"] = embedding_mod
    w5_dir = str(_PROJECT_ROOT / "w5")
    if w5_dir not in sys.path:
        sys.path.insert(0, w5_dir)

    for name in ("utils", "chroma_store", "C1M5_Assignment", "C1M5_Assignment_stateful"):
        sys.modules.pop(name, None)

    import utils as utils_mod

    utils_mod.generate_with_single_input = MagicMock(side_effect=_fake_llm_response)

    spec = importlib.util.spec_from_file_location(
        "C1M5_Assignment_stateful", _PROJECT_ROOT / "w5" / "C1M5_Assignment_stateful.py"
    )
    stateful_mod = importlib.util.module_from_spec(spec)
    sys.modules["C1M5_Assignment_stateful"] = stateful_mod
    spec.loader.exec_module(stateful_mod)

    try:
        yield stateful_mod, utils_mod, embedding_mod
    finally:
        for name, original in _originals.items():
            if original is not None:
                sys.modules[name] = original
            else:
                sys.modules.pop(name, None)
        sys.path[:] = original_sys_path


class TestStatefulRegression:
    """Full cell-sequence regression run — catches breakage from future edits."""

    def _mock_store(self):
        store = MagicMock()
        store.list_collections.return_value = ["faq_w5", "products_w5"]
        store.count.return_value = 25
        product = {
            "id": "0",
            "product_id": 1,
            "productDisplayName": "Blue Tee",
            "masterCategory": "Apparel",
            "usage": "Casual",
            "gender": "Men",
            "articleType": "T-Shirts",
            "subCategory": "Topwear",
            "baseColour": "Blue",
            "season": "Summer",
            "year": 2020,
        }
        faq_doc = {"id": "0", "question": "What is your return policy?", "answer": "30 days.", "type": "general"}
        store.query.side_effect = lambda collection, *a, **k: (
            [faq_doc] if collection == "faq_w5" else [product]
        )
        store.query_with_filter.return_value = [product]
        return store

    def test_full_sequence_runs_without_error(self, w5_stateful_module):
        stateful_mod, utils_mod, embedding_mod = w5_stateful_module

        state = stateful_mod.NotebookState()
        state.store = self._mock_store()

        # cell_06 is skipped: it would try a real ChromaStore(persist_directory=...)
        # existence check, which is exactly the kind of "requires local setup"
        # integration path that unit tests skip rather than fail on.
        state = stateful_mod.run_until(133, state=state, skip_cells={6})

        assert state.check_if_faq_or_product is not None
        assert state.query_on_faq is not None
        assert state.decide_task_nature is not None
        assert state.get_relevant_products_from_query is not None
        assert state.answer_query is not None
        assert state.products_data is not None
        assert state.faq is not None
        assert state.values is not None

    def test_graded_functions_produce_sane_output(self, w5_stateful_module):
        stateful_mod, utils_mod, embedding_mod = w5_stateful_module

        state = stateful_mod.NotebookState()
        state.store = self._mock_store()
        state = stateful_mod.run_until(94, state=state, skip_cells={6})

        label, tokens = state.check_if_faq_or_product("What is your return policy?", simplified=True)
        assert label in ("FAQ", "Product", "undefined")
        assert isinstance(tokens, int)

        results, tokens = state.get_relevant_products_from_query("blue shirts", simplified=True)
        assert isinstance(results, list)
        assert tokens == 0
