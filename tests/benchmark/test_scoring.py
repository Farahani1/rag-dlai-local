"""Unit tests for ``benchmark/scoring.py`` and the question set's shape."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "benchmark_scoring", _PROJECT_ROOT / "benchmark" / "scoring.py"
)
scoring = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = scoring
_spec.loader.exec_module(scoring)

CATALOGUE = {
    15970: {"articleType": "Tshirts", "baseColour": "Blue", "gender": "Men", "price": 40},
    15971: {"articleType": "Tshirts", "baseColour": "Blue", "gender": "Men", "price": 90},
    15972: {"articleType": "Tshirts", "baseColour": "Red", "gender": "Men", "price": 20},
    2012: {"articleType": "Tshirts", "baseColour": "Blue", "gender": "Men", "price": 30},
}
BLUE_MEN_TSHIRTS = {"match": {"articleType": "Tshirts", "baseColour": "Blue", "gender": "Men"}}


class TestFacts:
    def test_all_facts_present(self):
        expected = {"facts": [["8:00 am", "8 am"], ["8 pm", "20:00"]]}
        assert scoring.score(expected, "Support is open 8 AM to 8 PM.", CATALOGUE)

    def test_one_fact_missing(self):
        expected = {"facts": [["8:00 am", "8 am"], ["8 pm", "20:00"]]}
        assert not scoring.score(expected, "Support opens at 8 AM.", CATALOGUE)


class TestProducts:
    def test_enough_matching_products(self):
        spec = {"products": {**BLUE_MEN_TSHIRTS, "min": 2}}
        answer = "Try Product ID: 15970 and Product ID: 15971."
        assert scoring.score(spec, answer, CATALOGUE)

    def test_too_few_products(self):
        spec = {"products": {**BLUE_MEN_TSHIRTS, "min": 3}}
        assert not scoring.score(spec, "Try 15970 and 15971.", CATALOGUE)

    def test_any_wrong_product_fails(self):
        spec = {"products": {**BLUE_MEN_TSHIRTS, "min": 1}}
        assert not scoring.score(spec, "Try 15970 or the red 15972.", CATALOGUE)

    def test_exclude(self):
        spec = {"products": {"match": {"articleType": "Tshirts"}, "exclude": {"baseColour": "Red"}, "min": 1}}
        assert scoring.score(spec, "ID 15970", CATALOGUE)
        assert not scoring.score(spec, "ID 15972", CATALOGUE)

    def test_price_max(self):
        spec = {"products": {**BLUE_MEN_TSHIRTS, "price_max": 50, "min": 1}}
        assert scoring.score(spec, "ID 15970", CATALOGUE)
        assert not scoring.score(spec, "ID 15971", CATALOGUE)

    def test_unknown_numbers_are_ignored(self):
        spec = {"products": {**BLUE_MEN_TSHIRTS, "min": 1}}
        assert scoring.score(spec, "Order 99999 and ID 15970", CATALOGUE)

    def test_bare_year_is_not_an_id_but_labelled_one_is(self):
        assert scoring.mentioned_product_ids("Product Year: 2012.0", set(CATALOGUE)) == []
        assert scoring.mentioned_product_ids("Product ID: 2012", set(CATALOGUE)) == [2012]


class TestAbstain:
    @pytest.mark.parametrize("answer", [
        "Sorry, we don't sell bicycles.",
        "There is no information about a student discount.",
    ])
    def test_abstains(self, answer):
        assert scoring.score({"abstain": True}, answer, CATALOGUE)

    def test_does_not_abstain(self):
        assert not scoring.score({"abstain": True}, "Yes! Students get 15% off.", CATALOGUE)


def test_unknown_spec_raises():
    with pytest.raises(ValueError):
        scoring.score({"nonsense": 1}, "x", CATALOGUE)


@pytest.fixture(scope="module")
def questions():
    path = _PROJECT_ROOT / "benchmark" / "questions.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


class TestQuestionSet:
    """The committed question set keeps the shape the plan fixed."""

    def test_twelve_unique_ids(self, questions):
        ids = [q["id"] for q in questions]
        assert len(ids) == 12 and len(set(ids)) == 12

    def test_category_counts(self, questions):
        from collections import Counter

        assert Counter(q["category"] for q in questions) == {
            "direct": 3, "indirect": 2, "filter": 3, "limitation": 2, "insufficient": 2,
        }

    def test_every_entry_is_scoreable(self, questions):
        for q in questions:
            assert {"id", "category", "question", "expects_filter", "expected"} <= set(q)
            # Must not raise: every expected spec is one the scorer understands.
            scoring.score(q["expected"], "", CATALOGUE)
