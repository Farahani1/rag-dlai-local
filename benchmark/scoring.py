"""Score one benchmark answer against its ``expected`` entry in questions.yaml.

Deliberately simple and deterministic: substring facts, catalogue-checked
product IDs, and a fixed list of abstain phrases. Anything the rules get
wrong is corrected in the results CSV's ``manual_override`` column.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

ABSTAIN_PHRASES = (
    "we don't",
    "we do not",
    "don't sell",
    "do not sell",
    "don't offer",
    "do not offer",
    "don't carry",
    "do not carry",
    "don't have",
    "do not have",
    "not available",
    "no information",
    "not mentioned",
    "isn't mentioned",
    "is not mentioned",
    "not covered",
    "not able to find",
    "unable to find",
    "couldn't find",
    "could not find",
    "i don't know",
    "i do not know",
    "not sure",
    "no mention",
)

# "Product ID: 15970", "ID 15970", "(ID: 15970)", "#15970"
_ID_WITH_LABEL = re.compile(r"(?:\bID\b|#)\s*[:#]?\s*(\d{4,5})\b", re.IGNORECASE)
_BARE_NUMBER = re.compile(r"\b(\d{4,5})\b")
# Catalogue years (2007-2019) are echoed as "Product Year: 2012.0"; a bare
# number in this range is only counted as an ID when it carries a label.
_YEAR_RANGE = range(1990, 2031)


def mentioned_product_ids(answer: str, catalogue_ids: set[int]) -> list[int]:
    """Return catalogue product IDs named in *answer*, in order, without repeats."""
    found: list[int] = []
    labelled = {int(m) for m in _ID_WITH_LABEL.findall(answer)}
    for match in _BARE_NUMBER.finditer(answer):
        value = int(match.group(1))
        if value not in catalogue_ids or value in found:
            continue
        if value in _YEAR_RANGE and value not in labelled:
            continue
        found.append(value)
    return found


def _product_matches(product: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    for key, value in (spec.get("match") or {}).items():
        if product.get(key) != value:
            return False
    for key, value in (spec.get("exclude") or {}).items():
        if product.get(key) == value:
            return False
    price_max = spec.get("price_max")
    if price_max is not None and not product.get("price", float("inf")) <= price_max:
        return False
    return True


def score(
    expected: Mapping[str, Any],
    answer: str,
    catalogue: Mapping[int, Mapping[str, Any]],
) -> bool:
    """Return True if *answer* satisfies *expected*.

    *catalogue* maps product_id to the product's attribute dict.
    """
    text = (answer or "").lower()

    if "facts" in expected:
        return all(
            any(spelling.lower() in text for spelling in fact)
            for fact in expected["facts"]
        )

    if "products" in expected:
        spec = expected["products"]
        ids = mentioned_product_ids(answer or "", set(catalogue))
        # Every product named must fit the request (a black bag in answer to
        # "not black" is wrong even next to correct ones), and enough of them.
        return len(ids) >= spec.get("min", 1) and all(
            _product_matches(catalogue[i], spec) for i in ids
        )

    if expected.get("abstain"):
        return any(phrase in text for phrase in ABSTAIN_PHRASES)

    raise ValueError(f"Unknown expected spec: {expected!r}")
