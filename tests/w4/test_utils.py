"""Tests for w4/utils.py adapted utility functions.

These tests verify the structure and behavior of key utility functions
introduced or modified during W4 adaptation. No Ollama calls are made.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Module-level loading with mock config (cleaned up at session end)
# ============================================================================

# Save original setting module so we can restore it
_original_setting = sys.modules.get("setting")

_mock_config = SimpleNamespace()
_mock_config.ollama = {
    "modelName": "test-model",
    "url": "http://localhost:11434/api/generate",
}
_mock_config.completionOptions = {"temperature": 0.7, "maxTokens": 500}


class _MockSetting:
    config = _mock_config


sys.modules["setting"] = _MockSetting()


def _load_utils():
    """Load w4/utils.py via importlib, returning the module object."""
    spec = importlib.util.spec_from_file_location(
        "w4.utils",
        _PROJECT_ROOT / "w4" / "utils.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["w4.utils"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_UTILS_MODULE = _load_utils()

# Restore original setting module so other test files are not affected
if _original_setting is not None:
    sys.modules["setting"] = _original_setting
else:
    sys.modules.pop("setting", None)


# ============================================================================
# Tests: generate_params_dict
# ============================================================================


class TestGenerateParamsDict:
    """Tests for generate_params_dict — a pure dict-builder function."""

    def test_has_required_keys(self):
        result = _UTILS_MODULE.generate_params_dict("Hello", temperature=0.5)
        for key in ("prompt", "role", "temperature", "top_p", "max_tokens", "model"):
            assert key in result, f"Missing key '{key}' in result"

    def test_preserves_prompt(self):
        result = _UTILS_MODULE.generate_params_dict("Hello world")
        assert result["prompt"] == "Hello world"

    def test_default_role_is_user(self):
        result = _UTILS_MODULE.generate_params_dict("Test")
        assert result["role"] == "user"

    def test_custom_role(self):
        result = _UTILS_MODULE.generate_params_dict("Test", role="assistant")
        assert result["role"] == "assistant"

    def test_default_model_from_config(self):
        result = _UTILS_MODULE.generate_params_dict("Test")
        assert result["model"] == "test-model"

    def test_explicit_model_overrides_default(self):
        result = _UTILS_MODULE.generate_params_dict("Test", model="custom-model")
        assert result["model"] == "custom-model"

    def test_temperature_passed_through(self):
        result = _UTILS_MODULE.generate_params_dict("Test", temperature=0.3)
        assert result["temperature"] == 0.3

    def test_top_p_passed_through(self):
        result = _UTILS_MODULE.generate_params_dict("Test", top_p=0.9)
        assert result["top_p"] == 0.9


# ============================================================================
# Tests: print_object_properties
# ============================================================================


class TestPrintObjectProperties:
    """Smoke tests for print_object_properties (just verify no crash)."""

    def test_prints_dict_without_crash(self):
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            _UTILS_MODULE.print_object_properties({"key": "value"})
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "key:" in output
        assert "value" in output

    def test_prints_list_without_crash(self):
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            _UTILS_MODULE.print_object_properties([{"a": 1}, {"b": 2}])
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "a:" in output or "b:" in output