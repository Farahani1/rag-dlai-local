"""Tests for w5/utils.py adapted utility functions.

Covers what differs from w4/utils.py: the OpenAI-shaped LLM response
(``generate_with_single_input``), the local no-op tracer, and the
Chroma list-shaped metadata filters. No Ollama calls are made — the HTTP
layer is mocked via ``requests.post``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_original_setting = sys.modules.get("setting")

_mock_config = SimpleNamespace()
_mock_config.ollama = {"modelName": "test-model", "url": "http://localhost:11434/api/generate"}
_mock_config.completionOptions = {"temperature": 0.7, "maxTokens": 500}
_mock_config.clothesData = _PROJECT_ROOT / "data" / "clothes_json.joblib"
_mock_config.faqData = _PROJECT_ROOT / "data" / "faq.joblib"
_mock_config.productsChromaPath = _PROJECT_ROOT / "data" / "chroma_db_products"


class _MockSetting:
    config = _mock_config


sys.modules["setting"] = _MockSetting()


def _load_utils():
    spec = importlib.util.spec_from_file_location("w5.utils", _PROJECT_ROOT / "w5" / "utils.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["w5.utils"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_UTILS = _load_utils()

if _original_setting is not None:
    sys.modules["setting"] = _original_setting
else:
    sys.modules.pop("setting", None)


# ============================================================================
# Tests: generate_with_single_input (OpenAI-compatible shape)
# ============================================================================


class TestGenerateWithSingleInput:
    def _fake_ollama_response(self, response_text="Hello", prompt_eval_count=7, eval_count=3):
        payload = {"response": response_text}
        if prompt_eval_count is not None:
            payload["prompt_eval_count"] = prompt_eval_count
        if eval_count is not None:
            payload["eval_count"] = eval_count
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = payload
        return mock_resp

    def test_returns_openai_shape(self):
        with patch.object(_UTILS.requests, "post", return_value=self._fake_ollama_response()) as mock_post:
            result = _UTILS.generate_with_single_input("Hi there")
        assert "choices" in result
        assert "usage" in result
        assert result["choices"][0]["message"]["content"] == "Hello"
        assert result["choices"][0]["message"]["role"] == "assistant"
        mock_post.assert_called_once()

    def test_token_counts_from_ollama_fields(self):
        with patch.object(
            _UTILS.requests, "post",
            return_value=self._fake_ollama_response(prompt_eval_count=15, eval_count=5),
        ):
            result = _UTILS.generate_with_single_input("Hi there")
        assert result["usage"]["prompt_tokens"] == 15
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 20

    def test_falls_back_to_word_count_when_fields_missing(self):
        """If Ollama omits prompt_eval_count/eval_count, don't crash — estimate instead."""
        with patch.object(
            _UTILS.requests, "post",
            return_value=self._fake_ollama_response(
                response_text="one two three", prompt_eval_count=None, eval_count=None
            ),
        ):
            result = _UTILS.generate_with_single_input("a b c d")
        assert result["usage"]["prompt_tokens"] == 4
        assert result["usage"]["completion_tokens"] == 3
        assert result["usage"]["total_tokens"] == 7

    def test_model_defaults_to_config(self):
        with patch.object(_UTILS.requests, "post", return_value=self._fake_ollama_response()):
            result = _UTILS.generate_with_single_input("Hi")
        assert result["model"] == "test-model"

    def test_explicit_model_overrides_default(self):
        with patch.object(_UTILS.requests, "post", return_value=self._fake_ollama_response()):
            result = _UTILS.generate_with_single_input("Hi", model="custom-model")
        assert result["model"] == "custom-model"

    def test_raises_on_request_failure(self):
        with patch.object(_UTILS.requests, "post", side_effect=Exception("connection refused")):
            with pytest.raises(Exception, match="Ollama call failed"):
                _UTILS.generate_with_single_input("Hi")


# ============================================================================
# Tests: generate_params_dict
# ============================================================================


class TestGenerateParamsDict:
    def test_has_required_keys(self):
        result = _UTILS.generate_params_dict("Hello", temperature=0.5)
        for key in ("prompt", "role", "temperature", "top_p", "max_tokens", "model"):
            assert key in result

    def test_default_model_from_config(self):
        result = _UTILS.generate_params_dict("Test")
        assert result["model"] == "test-model"

    def test_explicit_model_overrides_default(self):
        result = _UTILS.generate_params_dict("Test", model="custom-model")
        assert result["model"] == "custom-model"


# ============================================================================
# Tests: NullTracer / NullSpan (no-op tracing stand-ins)
# ============================================================================


class TestNullTracer:
    def test_tool_decorator_is_identity(self):
        tracer = _UTILS.NullTracer()

        @tracer.tool
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_start_as_current_span_is_a_context_manager(self):
        tracer = _UTILS.NullTracer()
        with tracer.start_as_current_span("my_span", openinference_span_kind="tool") as span:
            span.set_input("in")
            span.set_output("out")
            span.set_attribute("key", "value")
            span.set_status(_UTILS.Status(_UTILS.StatusCode.OK))

    def test_span_swallows_record_exception_without_raising(self):
        tracer = _UTILS.NullTracer()
        with tracer.start_as_current_span("my_span") as span:
            try:
                raise ValueError("boom")
            except ValueError as e:
                span.record_exception(e)
        # No exception propagated past this point.


# ============================================================================
# Tests: get_filter_by_metadata / filters_to_where / _condition_key
# ============================================================================


class TestGetFilterByMetadata:
    def test_none_input_returns_none(self):
        assert _UTILS.get_filter_by_metadata(None) is None

    def test_returns_list_not_merged_dict(self):
        result = _UTILS.get_filter_by_metadata({"gender": ["Men"], "baseColour": ["Blue"]})
        assert isinstance(result, list)
        assert len(result) == 2

    def test_price_range_two_conditions(self):
        result = _UTILS.get_filter_by_metadata({"price": {"min": 10, "max": 50}})
        assert result == [{"price": {"$gte": 10}}, {"price": {"$lte": 50}}]


class TestFiltersToWhere:
    def test_none_or_empty_returns_none(self):
        assert _UTILS.filters_to_where(None) is None
        assert _UTILS.filters_to_where([]) is None

    def test_single_condition_returned_as_is(self):
        cond = {"gender": {"$in": ["Men"]}}
        assert _UTILS.filters_to_where([cond]) == cond

    def test_multiple_conditions_wrapped_in_and(self):
        conds = [{"gender": {"$in": ["Men"]}}, {"baseColour": {"$in": ["Blue"]}}]
        result = _UTILS.filters_to_where(conds)
        assert result == {"$and": conds}


class TestConditionKey:
    def test_extracts_the_single_key(self):
        assert _UTILS._condition_key({"baseColour": {"$in": ["Blue"]}}) == "baseColour"


# ============================================================================
# Tests: parse_json_output
# ============================================================================


class TestParseJsonOutput:
    def test_parse_valid_json(self):
        assert _UTILS.parse_json_output('{"key": "value"}') == {"key": "value"}

    def test_parse_invalid_returns_none(self):
        assert _UTILS.parse_json_output("not json") is None


# ============================================================================
# Tests: print_properties / make_url (smoke — no crash)
# ============================================================================


class TestDisplayHelpers:
    def test_print_properties_handles_plain_dict(self, capsys):
        _UTILS.print_properties({"id": "1", "question": "Q?"})
        captured = capsys.readouterr()
        assert "question" in captured.out

    def test_make_url_does_not_raise(self, capsys):
        _UTILS.make_url()
        _UTILS.make_url("/settings/models")
        captured = capsys.readouterr()
        assert "adapted mode" in captured.out
