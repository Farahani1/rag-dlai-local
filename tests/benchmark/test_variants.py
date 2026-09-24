"""Unit tests for ``benchmark/variants.py`` (the before/after experiment switches)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "benchmark_variants", _PROJECT_ROOT / "benchmark" / "variants.py"
)
variants = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = variants
_spec.loader.exec_module(variants)

JSON_TEXT = '{\n    "gender": ["Men"],\n    "baseColour": ["Black"]\n}'


class TestStripCodeFence:
    @pytest.mark.parametrize("wrapped", [
        f"```json\n{JSON_TEXT}\n```",
        f"```\n{JSON_TEXT}\n```",
        f"  ```json\n{JSON_TEXT}\n```  \n",
    ])
    def test_fenced_json_parses_after_stripping(self, wrapped):
        assert json.loads(variants.strip_code_fence(wrapped)) == json.loads(JSON_TEXT)

    def test_plain_json_is_untouched(self):
        assert variants.strip_code_fence(JSON_TEXT) == JSON_TEXT

    def test_backticks_inside_are_kept(self):
        text = '{"note": "use ``` here"}'
        assert variants.strip_code_fence(text) == text


def test_drop_non_string_values_removes_nan():
    values = {"baseColour": {"Black", math.nan, float("nan"), "Red"}, "season": {"Summer"}}
    cleaned = variants.drop_non_string_values(values)
    assert cleaned == {"baseColour": {"Black", "Red"}, "season": {"Summer"}}
    assert "nan" not in str(cleaned)
    assert any(isinstance(v, float) for v in values["baseColour"])  # input not modified


class TestRouter:
    def test_max_similarity_is_cosine(self):
        matrix = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        assert variants.max_similarity([3.0, 0.0], matrix) == pytest.approx(1.0)
        assert variants.max_similarity([1.0, 1.0], matrix) == pytest.approx(1 / math.sqrt(2))

    @pytest.mark.parametrize("similarity, expected", [(0.9, "FAQ"), (0.4025, "FAQ"), (0.40, "Product")])
    def test_route_by_similarity(self, similarity, expected):
        assert variants.route_by_similarity(similarity, 0.4025) == expected

    def test_best_threshold_separates_and_takes_widest_margin(self):
        threshold, accuracy = variants.best_threshold([0.6, 0.7, 0.8], [0.2, 0.3])
        assert accuracy == 1.0
        assert threshold == pytest.approx(0.45)

    def test_best_threshold_with_overlap(self):
        threshold, accuracy = variants.best_threshold([0.3, 0.7], [0.4, 0.2])
        assert accuracy == pytest.approx(0.75)

    def test_patch_router_uses_fake_embeddings(self, monkeypatch, tmp_path):
        dev = tmp_path / "router_dev.yaml"
        dev.write_text("threshold: 0.5\nfaq: []\nproduct: []\n", encoding="utf-8")
        monkeypatch.setattr(variants, "ROUTER_DEV_PATH", dev)
        monkeypatch.setattr(variants, "load_router_threshold", lambda path=dev: 0.5)
        vectors = {"How do I return an item?": [1.0, 0.0], "return it": [0.9, 0.1], "blue shirts": [0.0, 1.0]}
        ns = {"faq": [{"question": "How do I return an item?"}], "embed_query": lambda q: vectors[q]}
        info = variants.apply_variant("router", ns)
        assert info["router_threshold"] == 0.5
        assert ns["check_if_faq_or_product"]("return it") == ("FAQ", 0)
        assert ns["check_if_faq_or_product"]("blue shirts") == ("Product", 0)


def test_patch_parse_wraps_parser_and_cleans_values():
    seen = []
    ns = {
        "parse_json_output": lambda text: seen.append(text) or json.loads(text),
        "values": {"baseColour": {"Black", float("nan")}},
    }
    variants.apply_variant("parse", ns)
    assert ns["parse_json_output"](f"```json\n{JSON_TEXT}\n```") == json.loads(JSON_TEXT)
    assert seen == [JSON_TEXT]
    assert ns["values"] == {"baseColour": {"Black"}}


def test_every_variant_has_a_call_sequence():
    assert set(variants.VARIANTS) == set(variants.LLM_CALLS)
    with pytest.raises(SystemExit):
        variants.apply_variant("nonsense", {})


def test_committed_router_threshold_is_set():
    assert 0 < variants.load_router_threshold() < 1
