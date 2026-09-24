"""Pipeline variants for the before/after experiments.

Each variant patches the loaded W5 notebook namespace; the notebook and its
mirror stay unchanged. Variants are a registry (``VARIANTS``), so a later
variant, such as a trained classifier replacing a step, is one new entry.

- ``baseline``: the course pipeline as written.
- ``parse``: glue fixes for the filter step. Strip a Markdown code fence
  from the model's JSON before the notebook's parser sees it, and remove
  non-string values (``nan`` for products without a colour) from the
  allowed-values list the filter prompt shows the model.
- ``router``: replace the LLM routing call with an embedding router: FAQ if
  the query is close enough to one of the FAQ questions, else Product.
- ``both``: ``parse`` + ``router``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROUTER_DEV_PATH = Path(__file__).resolve().parent / "router_dev.yaml"

_FENCE_OPEN = re.compile(r"^\s*```[A-Za-z]*\s*\n?")
_FENCE_CLOSE = re.compile(r"\n?\s*```\s*$")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def strip_code_fence(text: str) -> str:
    """Remove a leading ```/```json fence line and a trailing ``` fence, if present."""
    if not text.lstrip().startswith("```"):
        return text
    return _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text, count=1), count=1)


def drop_non_string_values(values: dict[str, set]) -> dict[str, set]:
    """Return a copy of the allowed-values dict keeping only string values."""
    return {key: {v for v in vals if isinstance(v, str)} for key, vals in values.items()}


def max_similarity(query_vec: Sequence[float], faq_matrix: np.ndarray) -> float:
    """Highest cosine similarity between a query vector and the rows of *faq_matrix*."""
    q = np.asarray(query_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    m = faq_matrix / np.clip(np.linalg.norm(faq_matrix, axis=1, keepdims=True), 1e-12, None)
    return float(np.max(m @ q))


def route_by_similarity(similarity: float, threshold: float) -> str:
    return "FAQ" if similarity >= threshold else "Product"


def best_threshold(faq_scores: Sequence[float], product_scores: Sequence[float]) -> tuple[float, float]:
    """Pick the threshold with the highest accuracy on labelled similarity scores.

    Candidates are midpoints between consecutive sorted scores; ties go to the
    widest-margin candidate. Returns (threshold, accuracy).
    """
    labelled = sorted([(s, "FAQ") for s in faq_scores] + [(s, "Product") for s in product_scores])
    scores = [s for s, _ in labelled]
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])] or scores
    best = None
    for t in candidates:
        correct = sum(route_by_similarity(s, t) == label for s, label in labelled)
        gaps = [abs(s - t) for s in scores]
        key = (correct, min(gaps))
        if best is None or key > best[0]:
            best = (key, t)
    (correct, _), threshold = best
    return round(threshold, 4), correct / len(labelled)


def load_router_threshold(path: Path = ROUTER_DEV_PATH) -> float:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    threshold = data.get("threshold")
    if threshold is None:
        raise SystemExit(f"No threshold in {path}; run: python benchmark/tune_router.py --write")
    return float(threshold)


# ---------------------------------------------------------------------------
# Patches applied to the loaded notebook namespace
# ---------------------------------------------------------------------------


def patch_parse(ns: dict[str, Any], info: dict[str, Any]) -> None:
    inner_parse = ns["parse_json_output"]

    def parse_without_fence(llm_output):
        return inner_parse(strip_code_fence(llm_output))

    ns["parse_json_output"] = parse_without_fence
    # The filter prompt reads the global `values` at call time.
    ns["values"] = drop_non_string_values(ns["values"])


def faq_matrix(ns: dict[str, Any]) -> np.ndarray:
    return np.asarray([ns["embed_query"](item["question"]) for item in ns["faq"]], dtype=np.float32)


def patch_router(ns: dict[str, Any], info: dict[str, Any]) -> None:
    threshold = load_router_threshold()
    matrix = faq_matrix(ns)
    embed = ns["embed_query"]

    def check_if_faq_or_product(query, simplified=False):
        return route_by_similarity(max_similarity(embed(query), matrix), threshold), 0

    ns["check_if_faq_or_product"] = check_if_faq_or_product
    info["router_threshold"] = threshold


Patch = Callable[[dict[str, Any], dict[str, Any]], None]

VARIANTS: dict[str, list[Patch]] = {
    "baseline": [],
    "parse": [patch_parse],
    "router": [patch_router],
    "both": [patch_parse, patch_router],
}

# LLM calls per question, in order, by route; used by step_times.py.
LLM_CALLS: dict[str, dict[str, list[str]]] = {
    "baseline": {"FAQ": ["t_route", "t_answer"], "Product": ["t_route", "t_task", "t_filter", "t_answer"]},
    "parse": {"FAQ": ["t_route", "t_answer"], "Product": ["t_route", "t_task", "t_filter", "t_answer"]},
    "router": {"FAQ": ["t_answer"], "Product": ["t_task", "t_filter", "t_answer"]},
    "both": {"FAQ": ["t_answer"], "Product": ["t_task", "t_filter", "t_answer"]},
}


def apply_variant(name: str, ns: dict[str, Any]) -> dict[str, Any]:
    """Apply a variant's patches; return extra facts to record in the CSV header."""
    if name not in VARIANTS:
        raise SystemExit(f"Unknown variant '{name}'. Known: {', '.join(VARIANTS)}")
    info: dict[str, Any] = {}
    for patch in VARIANTS[name]:
        patch(ns, info)
    return info
