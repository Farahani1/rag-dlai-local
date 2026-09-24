"""Extracted pure cell functions from w5/C1M5_Assignment.py for standalone testing.

These are hand-copied (not dynamically extracted) so they can be unit tested
without needing the notebook cell-sequencing machinery. Keep them in sync with
the corresponding cell bodies in ``w5/C1M5_Assignment.py`` if those change.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# From cell_42: generate_faq_layout
# ---------------------------------------------------------------------------


def generate_faq_layout(faq_dict: list) -> str:
    """
    Generates a formatted string layout for a list of FAQs.

    Parameters:
    - faq_dict (list): A list of dicts with 'question', 'answer', 'type' keys.

    Returns:
    - str: One formatted line per FAQ entry.
    """
    t = ""
    for f in faq_dict:
        t += f"Question: {f['question']} Answer: {f['answer']} Type: {f['type']}\n"
    return t


# ---------------------------------------------------------------------------
# From cell_91 (adapted branch): parse_json_output
# ---------------------------------------------------------------------------


def parse_json_output(llm_output: str) -> dict | None:
    """
    Parses a string output from an LLM into a JSON object.

    Cleans minor formatting issues before parsing. Returns ``None`` on failure.
    """
    import json

    try:
        llm_output = (
            llm_output.replace("\n", "").replace("'", "").replace("}}", "}").replace("{{", "{")
        )
        parsed_json = json.loads(llm_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None


# ---------------------------------------------------------------------------
# From cell_91 (adapted branch): get_filter_by_metadata
# ---------------------------------------------------------------------------


def get_filter_by_metadata(json_output: dict | None = None) -> list[dict] | None:
    """
    Generate a list of Chroma single-key filter conditions from a metadata dict.

    Returns a **list** (not a merged ``$and`` dict) so callers can
    progressively drop the least-important conditions by key.
    """
    if json_output is None:
        return None

    valid_keys = (
        "gender",
        "masterCategory",
        "articleType",
        "baseColour",
        "price",
        "usage",
        "season",
    )

    conditions: list[dict] = []
    for key, value in json_output.items():
        if key not in valid_keys:
            continue

        if key == "price":
            if not isinstance(value, dict):
                continue
            min_price = value.get("min")
            max_price = value.get("max")
            if min_price is None or max_price is None:
                continue
            if min_price <= 0 or max_price == "inf":
                continue
            conditions.append({key: {"$gte": min_price}})
            conditions.append({key: {"$lte": max_price}})
        else:
            conditions.append(
                {key: {"$in": list(value) if isinstance(value, (list, set)) else [value]}}
            )

    return conditions if conditions else None


# ---------------------------------------------------------------------------
# From cell_104 (adapted branch): generate_items_context
# ---------------------------------------------------------------------------


def generate_items_context(results: list) -> str:
    """Compile product details from plain dicts into a formatted string.

    Parameters:
    - results (list): A list of product dicts with the full W5 schema
      (product_id, productDisplayName, masterCategory, usage, gender,
      articleType, subCategory, baseColour, season, year).

    Returns:
    - str: A multi-line string, one formatted line per product.
    """
    t = ""
    for item in results:
        t += (
            f"Product ID: {item['product_id']}. "
            f"Product name: {item['productDisplayName']}. "
            f"Product Category: {item['masterCategory']}. "
            f"Product usage: {item['usage']}. "
            f"Product gender: {item['gender']}. "
            f"Product Type: {item['articleType']}. "
            f"Product Category: {item['subCategory']} "
            f"Product Color: {item['baseColour']}. "
            f"Product Season: {item['season']}. "
            f"Product Year: {item['year']}.\n"
        )
    return t
