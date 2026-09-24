"""Tests for the 4 graded cell functions from w5/C1M5_Assignment.py.

These tests verify the completed (solved) adapted versions of the W5 graded
functions. Functions that call the LLM or the vector store are tested with
mocked ``generate_with_single_input``/``store``/``embed_query`` to avoid
requiring a live Ollama server or a populated Chroma collection during
testing (tests never download models or modify project data
automatically).

Both the ``simplified=True`` and ``simplified=False`` branches are covered
for each graded function, since W5's whole point is comparing the two modes.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Module-level loading with mocked setting/embedding (cleaned up at session end)
# ============================================================================

_mock_config = SimpleNamespace(
    ollama={"modelName": "test-model", "url": "http://localhost:11434/api/generate"},
    completionOptions={"temperature": 0.7, "maxTokens": 500},
    clothesData=_PROJECT_ROOT / "data" / "clothes_json.joblib",
    faqData=_PROJECT_ROOT / "data" / "faq.joblib",
    productsChromaPath=_PROJECT_ROOT / "data" / "chroma_db_products",
    embeddingModel="sentence-transformers/all-MiniLM-L6-v2",
)


class _MockSetting:
    config = _mock_config


_original_setting = sys.modules.get("setting")
_original_embedding = sys.modules.get("embedding")
_original_sys_path = list(sys.path)
sys.modules["setting"] = _MockSetting()
sys.modules["embedding"] = ModuleType("embedding")
sys.modules["embedding"].embed_query = MagicMock(return_value=[0.1] * 384)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_UTILS = _load_module("w5.utils", _PROJECT_ROOT / "w5" / "utils.py")
# C1M5_Assignment.py's own module-level code does `sys.path.insert(0, w5_dir)`
# (same convention as w4/C1M4_Assignment.py) so its cell bodies can import
# sibling modules by bare name. That insert has no teardown of its own and
# would otherwise permanently shadow every other week's identically-named
# utils.py/chroma_store.py for the rest of the pytest session (this is what
# broke tests/w3 the first time this file was added) -- restore sys.path
# immediately after loading so only *this* module load benefits from it.
_ASSIGNMENT = _load_module("w5.C1M5_Assignment", _PROJECT_ROOT / "w5" / "C1M5_Assignment.py")
sys.path[:] = _original_sys_path

if _original_setting is not None:
    sys.modules["setting"] = _original_setting
else:
    sys.modules.pop("setting", None)
if _original_embedding is not None:
    sys.modules["embedding"] = _original_embedding
else:
    sys.modules.pop("embedding", None)


NullTracer = _UTILS.NullTracer
Status = _UTILS.Status
StatusCode = _UTILS.StatusCode
filters_to_where = _UTILS.filters_to_where
_condition_key = _UTILS._condition_key

cell_37 = _ASSIGNMENT.cell_37
cell_53 = _ASSIGNMENT.cell_53
cell_70 = _ASSIGNMENT.cell_70
cell_94 = _ASSIGNMENT.cell_94


FAUX_FAQ = [
    {"question": "What is your return policy?", "answer": "30 days, no questions asked.", "type": "returns"},
    {"question": "How can I contact support?", "answer": "Email support@example.com.", "type": "support"},
]


def _openai_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 2):
    return {
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _extract_inner_function(cell_func, func_name, extra_ns=None):
    """Extract an inner function definition from a cell function.

    Executes the inner function's source in a namespace pre-populated with
    the notebook-global names it relies on (``tracer``, ``generate_params_dict``,
    ``store``, etc.) — mirroring the technique in
    ``tests/w4/test_graded_cells.py``, extended for W5's tracer/store/embed
    dependencies.
    """
    ns = {
        "adapted": True,
        "json": __import__("json"),
        "tracer": NullTracer(),
        "Status": Status,
        "StatusCode": StatusCode,
        "faq": FAUX_FAQ,
        "generate_faq_layout": lambda faq_list: "".join(
            f"Question: {f['question']} Answer: {f['answer']} Type: {f['type']}\n" for f in faq_list
        ),
        "generate_params_dict": MagicMock(
            side_effect=lambda prompt, **kw: {"prompt": prompt, "role": kw.get("role", "user"), **kw}
        ),
        "generate_with_single_input": MagicMock(return_value=_openai_response("FAQ")),
        "store": MagicMock(),
        "embed_query": MagicMock(return_value=[0.1] * 384),
        "filters_to_where": filters_to_where,
        "_condition_key": _condition_key,
        "generate_filters_from_query": MagicMock(return_value=(None, 0)),
    }
    if extra_ns:
        ns.update(extra_ns)

    source_text = inspect.getsource(cell_func)
    inner_start = source_text.find(f"def {func_name}")
    if inner_start == -1:
        raise ValueError(f"Could not find inner function '{func_name}' in cell source")

    inner_code = source_text[inner_start:]
    exec(inner_code, ns)

    return ns[func_name], ns


# ============================================================================
# Graded cell 1: check_if_faq_or_product (cell_37)
# ============================================================================


class TestCheckIfFaqOrProduct:
    @pytest.fixture
    def setup(self):
        mock_gen = MagicMock(return_value=_openai_response("FAQ"))
        func, ns = _extract_inner_function(
            cell_37, "check_if_faq_or_product", extra_ns={"generate_with_single_input": mock_gen}
        )
        return func, ns, mock_gen

    @pytest.mark.parametrize("simplified", [False, True])
    def test_faq_query_returns_faq(self, setup, simplified):
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("FAQ")
        label, tokens = func("What is your return policy?", simplified=simplified)
        assert label == "FAQ"
        assert isinstance(tokens, int)

    @pytest.mark.parametrize("simplified", [False, True])
    def test_product_query_returns_product(self, setup, simplified):
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("Product")
        label, tokens = func("Do you have blue T-shirts?", simplified=simplified)
        assert label == "Product"

    def test_unknown_label_returns_undefined(self, setup):
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("Unknown")
        label, tokens = func("Some random query", simplified=False)
        assert label == "undefined"

    def test_label_case_and_whitespace_normalized(self, setup):
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("  faq  \n")
        label, tokens = func("Return policy?", simplified=False)
        assert label == "FAQ"

    def test_simplified_prompt_shorter_than_full(self, setup):
        """The whole point of `simplified` is fewer tokens -> a shorter prompt."""
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("FAQ")
        gen_params = ns["generate_params_dict"]
        func("What is your return policy?", simplified=False)
        full_prompt = gen_params.call_args[0][0]
        func("What is your return policy?", simplified=True)
        simplified_prompt = gen_params.call_args[0][0]
        assert len(simplified_prompt) < len(full_prompt)

    def test_query_present_in_both_prompt_variants(self, setup):
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("FAQ")
        query = "What is your return policy?"
        for simplified in (False, True):
            func(query, simplified=simplified)
            prompt_arg = ns["generate_params_dict"].call_args[0][0]
            assert query in prompt_arg

    def test_returns_total_tokens_from_usage(self, setup):
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("FAQ", prompt_tokens=20, completion_tokens=3)
        _, tokens = func("Return policy?", simplified=False)
        assert tokens == 23


# ============================================================================
# Graded cell 2: query_on_faq (cell_53)
# ============================================================================


class TestQueryOnFaq:
    @pytest.fixture
    def setup(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {"id": "0", "question": "What is your return policy?", "answer": "30 days.", "type": "returns"},
        ]
        func, ns = _extract_inner_function(cell_53, "query_on_faq", extra_ns={"store": mock_store})
        return func, ns, mock_store

    def test_non_simplified_returns_kwargs_with_full_faq(self, setup):
        func, ns, mock_store = setup
        result = func("How can I return an item?", simplified=False)
        assert isinstance(result, dict)
        assert "PROVIDED FAQ:" in result["prompt"]
        assert "What is your return policy?" in result["prompt"]
        mock_store.query.assert_not_called()

    def test_simplified_uses_semantic_search_top5(self, setup):
        func, ns, mock_store = setup
        result = func("How can I return an item?", simplified=True)
        assert isinstance(result, dict)
        mock_store.query.assert_called_once()
        args, kwargs = mock_store.query.call_args
        assert args[0] == "faq_w5"
        assert kwargs.get("top_k", args[2] if len(args) > 2 else None) == 5

    def test_simplified_prompt_includes_retrieved_faq(self, setup):
        func, ns, mock_store = setup
        result = func("How can I return an item?", simplified=True)
        assert "What is your return policy?" in result["prompt"]

    def test_prompt_includes_query_both_modes(self, setup):
        func, ns, mock_store = setup
        query = "How can I return an item?"
        for simplified in (False, True):
            result = func(query, simplified=simplified)
            assert query in result["prompt"]

    def test_forwards_extra_kwargs(self, setup):
        func, ns, mock_store = setup
        result = func("Test query", simplified=False, temperature=0.5, role="assistant")
        assert result.get("temperature") == 0.5
        assert result.get("role") == "assistant"


# ============================================================================
# Graded cell 3: decide_task_nature (cell_70)
# ============================================================================


class TestDecideTaskNature:
    @pytest.fixture
    def setup(self):
        mock_gen = MagicMock(return_value=_openai_response("technical"))
        func, ns = _extract_inner_function(
            cell_70, "decide_task_nature", extra_ns={"generate_with_single_input": mock_gen}
        )
        return func, ns, mock_gen

    @pytest.mark.parametrize("simplified", [False, True])
    def test_technical_query_returns_technical(self, setup, simplified):
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("technical")
        label, tokens = func("What are the blue dresses you have available?", simplified=simplified)
        assert label == "technical"

    @pytest.mark.parametrize("simplified", [False, True])
    def test_creative_query_returns_creative(self, setup, simplified):
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("creative")
        label, tokens = func("Give me suggestions on a nice look for a nightclub.", simplified=simplified)
        assert label == "creative"

    def test_default_is_simplified(self, setup):
        """decide_task_nature defaults to simplified=True, per the notebook signature."""
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("technical")
        gen_params = ns["generate_params_dict"]
        func("What are the prices?")
        default_prompt = gen_params.call_args[0][0]
        func("What are the prices?", simplified=False)
        full_prompt = gen_params.call_args[0][0]
        assert len(default_prompt) < len(full_prompt)

    def test_uses_zero_temperature_and_max_tokens_1(self, setup):
        func, ns, mock_gen = setup
        mock_gen.return_value = _openai_response("technical")
        func("What are the prices?", simplified=False)
        call_kwargs = ns["generate_params_dict"].call_args[1]
        assert call_kwargs.get("temperature") == 0
        assert call_kwargs.get("max_tokens") == 1


# ============================================================================
# Graded cell 4: get_relevant_products_from_query (cell_94)
# ============================================================================


class TestGetRelevantProductsFromQuery:
    def _product(self, **overrides):
        base = {"id": "1", "product_id": 1, "baseColour": "Blue", "gender": "Men"}
        base.update(overrides)
        return base

    @pytest.fixture
    def setup(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [self._product()]
        mock_store.query_with_filter.return_value = [self._product()]
        func, ns = _extract_inner_function(
            cell_94, "get_relevant_products_from_query", extra_ns={"store": mock_store}
        )
        return func, ns, mock_store

    def test_simplified_returns_zero_tokens_no_llm(self, setup):
        func, ns, mock_store = setup
        results, tokens = func("blue shirts", simplified=True)
        assert tokens == 0
        mock_store.query.assert_called_once_with("products_w5", [0.1] * 384, top_k=20)
        ns["generate_filters_from_query"].assert_not_called()

    def test_non_simplified_no_filters_falls_back_to_plain_search(self, setup):
        func, ns, mock_store = setup
        ns["generate_filters_from_query"].return_value = (None, 12)
        results, tokens = func("anything", simplified=False)
        assert tokens == 12
        mock_store.query.assert_called_once()
        mock_store.query_with_filter.assert_not_called()

    def test_non_simplified_with_filters_uses_filtered_search(self, setup):
        func, ns, mock_store = setup
        filters = [{"baseColour": {"$in": ["Blue"]}}]
        ns["generate_filters_from_query"].return_value = (filters, 12)
        mock_store.query_with_filter.return_value = [self._product() for _ in range(15)]
        results, tokens = func("blue shirts", simplified=False)
        assert tokens == 12
        mock_store.query_with_filter.assert_called_once()
        _, call_kwargs = mock_store.query_with_filter.call_args
        assert call_kwargs["metadata_filter"] == {"baseColour": {"$in": ["Blue"]}}
        assert len(results) == 15

    def test_progressive_relaxation_drops_least_important_filters_first(self, setup):
        """Fewer than 10 results triggers relaxation; baseColour (index 0 in
        importance_order) should be dropped before gender (last)."""
        func, ns, mock_store = setup
        filters = [
            {"baseColour": {"$in": ["Blue"]}},
            {"gender": {"$in": ["Men"]}},
        ]
        ns["generate_filters_from_query"].return_value = (filters, 5)

        call_log = []

        def fake_query_with_filter(collection, emb, top_k, metadata_filter):
            call_log.append(metadata_filter)
            # First call (both filters): too few results -> triggers relaxation.
            if len(call_log) == 1:
                return [self._product() for _ in range(2)]
            # After dropping baseColour: enough results -> stop here.
            return [self._product() for _ in range(6)]

        mock_store.query_with_filter.side_effect = fake_query_with_filter

        results, tokens = func("blue shirts for men", simplified=False)

        assert len(results) == 6
        # Second call's where-clause should no longer reference baseColour.
        assert "baseColour" not in str(call_log[1])
        assert "gender" in str(call_log[1])

    def test_returns_a_list(self, setup):
        func, ns, mock_store = setup
        results, tokens = func("blue shirts", simplified=True)
        assert isinstance(results, list)
