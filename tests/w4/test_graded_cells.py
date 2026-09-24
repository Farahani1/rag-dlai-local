"""Tests for the 5 graded cell functions from w4/C1M4_Assignment.py.

These tests verify the completed (solved) versions of the assignment functions.
Functions that call Ollama are tested with mocked ``generate_with_single_input``
and ``generate_params_dict`` to avoid requiring a live LLM during testing.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Path setup so tests can import w4 modules ─────────────────
_w4_dir = Path(__file__).resolve().parent.parent.parent / "w4"
if str(_w4_dir) not in sys.path:
    sys.path.insert(0, str(_w4_dir))

from C1M4_Assignment import (  # noqa: E402
    cell_22,
    cell_32,
    cell_40,
    cell_46,
    cell_55,
)


# ============================================================================
# Helper: extract the inner graded function from a cell function by executing
# its source in a controlled namespace.
# ============================================================================


FAUX_FAQ_LAYOUT = (
    "Question: What is your return policy? "
    "Answer: We accept returns within 30 days. Type: returns\n"
    "Question: How can I contact support? "
    "Answer: Email support@example.com. Type: support\n"
)

FAUX_VALUES = {
    "gender": {"Men", "Women", "Unisex"},
    "masterCategory": {"Apparel", "Footwear", "Accessories"},
    "articleType": {"T-Shirts", "Dresses", "Shoes"},
    "baseColour": {"Blue", "Red", "Green", "Black"},
    "usage": {"Casual", "Formal", "Sports"},
    "season": {"Summer", "Winter", "Spring", "Fall"},
}


def _extract_inner_function(cell_func, func_name, extra_ns=None):
    """Extract an inner function definition from a cell function.

    Executes the inner function's source code in a namespace that
    provides mocked versions of ``generate_params_dict`` and
    ``generate_with_single_input``, plus the globals the notebook
    cells normally provide (``faq_layout``, ``values``, etc.).

    Any keys in *extra_ns* override or extend the base namespace.
    """
    ns = {
        "adapted": True,
        "json": __import__("json"),
        "joblib": MagicMock(),
        "np": MagicMock(),
        "pd": MagicMock(),
        "faq_layout": FAUX_FAQ_LAYOUT,
        "values": FAUX_VALUES,
        "generate_params_dict": MagicMock(return_value={"prompt": "test", "role": "user"}),
        "generate_with_single_input": MagicMock(
            return_value={"role": "assistant", "content": "Test response"}
        ),
    }
    if extra_ns:
        ns.update(extra_ns)

    source_text = inspect.getsource(cell_func)

    # Locate the inner function definition
    inner_start = source_text.find(f"def {func_name}")
    if inner_start == -1:
        raise ValueError(f"Could not find inner function '{func_name}' in cell source")

    inner_code = source_text[inner_start:]
    exec(inner_code, ns)

    return ns[func_name], ns["generate_params_dict"], ns["generate_with_single_input"]


# ============================================================================
# Exercise 1: check_if_faq_or_product (cell_22)
# ============================================================================


class TestCheckIfFaqOrProduct:
    """Tests for check_if_faq_or_product (Exercise 1, cell_22)."""

    @pytest.fixture
    def setup(self):
        """Create the function with mocked dependencies."""
        mock_gen = MagicMock(
            return_value={"role": "assistant", "content": "FAQ"}
        )
        mock_params = MagicMock(return_value={"prompt": "test", "role": "user"})
        func, _, _ = _extract_inner_function(
            cell_22, "check_if_faq_or_product",
            extra_ns={
                "generate_with_single_input": mock_gen,
                "generate_params_dict": mock_params,
            },
        )
        return func, mock_params, mock_gen

    def test_faq_query_returns_faq(self, setup):
        func, _, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "FAQ"}
        result = func("What is your return policy?")
        assert result == "FAQ"

    def test_product_query_returns_product(self, setup):
        func, _, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "Product"}
        result = func("Do you have blue T-shirts?")
        assert result == "Product"

    def test_unknown_label_returns_none(self, setup):
        func, _, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "Unknown"}
        result = func("Some random query")
        assert result is None

    def test_label_with_whitespace_normalized(self, setup):
        func, _, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "FAQ  \n"}
        result = func("Return policy?")
        assert result == "FAQ"

    def test_empty_content_returns_none(self, setup):
        func, _, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": ""}
        result = func("What?")
        assert result is None

    def test_calls_generate_params_dict_with_low_temperature(self, setup):
        func, mock_params, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "FAQ"}
        func("Return policy?")
        call_kwargs = mock_params.call_args[1]
        assert call_kwargs.get("temperature") == 0.3
        assert call_kwargs.get("max_tokens") == 1

    def test_prompt_contains_query(self, setup):
        func, mock_params, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "FAQ"}
        test_query = "What is your return policy?"
        func(test_query)
        prompt_arg = mock_params.call_args[0][0]
        assert test_query in prompt_arg


# ============================================================================
# Exercise 2: query_on_faq (cell_32)
# ============================================================================


class TestQueryOnFaq:
    """Tests for query_on_faq (Exercise 2, cell_32)."""

    @pytest.fixture
    def setup(self):
        """Create the function with mocked dependencies."""
        mock_params = MagicMock(return_value={"prompt": "test", "role": "user"})
        func, _, _ = _extract_inner_function(
            cell_32, "query_on_faq",
            extra_ns={
                "generate_params_dict": mock_params,
            },
        )
        return func, mock_params

    def test_returns_kwargs_dict(self, setup):
        func, mock_params = setup
        result = func("How can I return an item?")
        assert isinstance(result, dict)
        assert "prompt" in result

    def test_prompt_includes_faq_layout(self, setup):
        func, mock_params = setup
        func("How can I return an item?")
        prompt_arg = mock_params.call_args[0][0]
        assert "PROVIDED FAQ:" in prompt_arg
        assert "What is your return policy?" in prompt_arg

    def test_prompt_includes_query(self, setup):
        func, mock_params = setup
        test_query = "How can I return an item?"
        func(test_query)
        prompt_arg = mock_params.call_args[0][0]
        assert test_query in prompt_arg

    def test_prompt_includes_faq_tags(self, setup):
        func, mock_params = setup
        func("Test query")
        prompt_arg = mock_params.call_args[0][0]
        assert "<FAQ>" in prompt_arg
        assert "</FAQ>" in prompt_arg

    def test_forwards_extra_kwargs(self, setup):
        func, mock_params = setup
        func("Test query", temperature=0.5, role="assistant")
        call_kwargs = mock_params.call_args[1]
        assert call_kwargs.get("temperature") == 0.5
        assert call_kwargs.get("role") == "assistant"


# ============================================================================
# Exercise 3: decide_task_nature (cell_40)
# ============================================================================


class TestDecideTaskNature:
    """Tests for decide_task_nature (Exercise 3, cell_40)."""

    @pytest.fixture
    def setup(self):
        """Create the function with mocked dependencies."""
        mock_gen = MagicMock(
            return_value={"role": "assistant", "content": "technical"}
        )
        mock_params = MagicMock(return_value={"prompt": "test", "role": "user"})
        func, _, _ = _extract_inner_function(
            cell_40, "decide_task_nature",
            extra_ns={
                "generate_with_single_input": mock_gen,
                "generate_params_dict": mock_params,
            },
        )
        return func, mock_params, mock_gen

    def test_technical_query_returns_technical(self, setup):
        func, _, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "technical"}
        result = func("What are the blue dresses you have available?")
        assert result == "technical"

    def test_creative_query_returns_creative(self, setup):
        func, _, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "creative"}
        result = func("Give me suggestions on a nice look for a nightclub.")
        assert result == "creative"

    def test_normalizes_case(self, setup):
        func, _, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "Creative"}
        result = func("Give me a look for a wedding.")
        assert result == "creative"

    def test_normalizes_whitespace(self, setup):
        func, _, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "  technical  "}
        result = func("Give me three T-shirts for summer.")
        assert result == "technical"

    def test_uses_zero_temperature(self, setup):
        func, mock_params, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "technical"}
        func("What are the prices?")
        call_kwargs = mock_params.call_args[1]
        assert call_kwargs.get("temperature") == 0
        assert call_kwargs.get("max_tokens") == 1

    def test_prompt_contains_query(self, setup):
        func, mock_params, mock_gen = setup
        mock_gen.return_value = {"role": "assistant", "content": "technical"}
        test_query = "Give me three T-shirts for summer."
        func(test_query)
        prompt_arg = mock_params.call_args[0][0]
        assert test_query in prompt_arg


# ============================================================================
# Exercise 4: get_params_for_task (cell_46)
# ============================================================================


class TestGetParamsForTask:
    """Tests for get_params_for_task (Exercise 4, cell_46)."""

    @pytest.fixture
    def func(self):
        """Return the get_params_for_task function (no mocks needed)."""
        f, _, _ = _extract_inner_function(cell_46, "get_params_for_task")
        return f

    def test_technical_returns_low_temperature(self, func):
        result = func("technical")
        assert result["temperature"] < 0.5

    def test_technical_returns_top_p_below_one(self, func):
        result = func("technical")
        assert result["top_p"] < 1.0

    def test_creative_returns_high_temperature(self, func):
        result = func("creative")
        assert result["temperature"] >= 0.8

    def test_creative_returns_top_p_below_one(self, func):
        result = func("creative")
        assert result["top_p"] < 1.0

    def test_creative_temperature_below_threshold(self, func):
        result = func("creative")
        assert result["temperature"] < 1.3

    def test_unrecognized_task_returns_fallback(self, func):
        result = func("unknown_type")
        assert isinstance(result, dict)
        assert "top_p" in result
        assert "temperature" in result

    def test_output_contains_required_keys(self, func):
        for task in ("technical", "creative", "unknown"):
            result = func(task)
            assert "top_p" in result
            assert "temperature" in result

    def test_technical_is_more_deterministic_than_creative(self, func):
        technical = func("technical")
        creative = func("creative")
        assert technical["temperature"] < creative["temperature"]


# ============================================================================
# Exercise 5: generate_metadata_from_query (cell_55)
# ============================================================================


class TestGenerateMetadataFromQuery:
    """Tests for generate_metadata_from_query (Exercise 5, cell_55)."""

    @pytest.fixture
    def setup(self):
        """Create the function with mocked dependencies."""
        mock_gen = MagicMock(
            return_value={
                "role": "assistant",
                "content": '{"gender": ["Men"], "price": {"min": 0, "max": "inf"}}',
            }
        )
        mock_params = MagicMock(return_value={"prompt": "test", "role": "user"})
        func, _, _ = _extract_inner_function(
            cell_55, "generate_metadata_from_query",
            extra_ns={
                "generate_with_single_input": mock_gen,
                "generate_params_dict": mock_params,
            },
        )
        return func, mock_params, mock_gen

    def test_returns_content_from_response(self, setup):
        func, _, mock_gen = setup
        expected = '{"gender": ["Men"]}'
        mock_gen.return_value = {"role": "assistant", "content": expected}
        result = func("Give me men's clothes")
        assert result == expected

    def test_prompt_includes_query(self, setup):
        func, mock_params, _ = setup
        test_query = "Create a look for a man that suits a sunny day"
        func(test_query)
        prompt_arg = mock_params.call_args[0][0]
        assert test_query in prompt_arg

    def test_prompt_includes_values_content(self, setup):
        func, mock_params, _ = setup
        func("Give me summer clothes")
        prompt_arg = mock_params.call_args[0][0]
        # The values dict's string representation should be in the prompt
        assert "gender" in prompt_arg.lower()
        assert "usage" in prompt_arg.lower()
        assert "season" in prompt_arg.lower()

    def test_uses_max_tokens_1500(self, setup):
        func, mock_params, _ = setup
        func("Give me clothes")
        call_kwargs = mock_params.call_args[1]
        assert call_kwargs.get("max_tokens") == 1500

    def test_uses_zero_temperature(self, setup):
        func, mock_params, _ = setup
        func("Give me clothes")
        call_kwargs = mock_params.call_args[1]
        assert call_kwargs.get("temperature") == 0

    def test_prompt_includes_json_structure_keys(self, setup):
        func, mock_params, _ = setup
        func("Give me blue dresses")
        prompt_arg = mock_params.call_args[0][0]
        for key in ("gender", "masterCategory", "articleType",
                      "baseColour", "price", "usage", "season"):
            assert key in prompt_arg