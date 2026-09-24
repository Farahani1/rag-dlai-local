"""Extracted cell functions from w4/C1M4_Assignment.py for standalone testing.

These functions are the adapted versions used when ``adapted=True``.
They are extracted here so tests can import them without needing to execute
the notebook cell runner machinery.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# From cell_29: generate_faq_layout
# ---------------------------------------------------------------------------


def generate_faq_layout(faq_dict: list) -> str:
    """
    Generates a formatted string layout for a list of FAQs.

    This function iterates through a dictionary of frequently asked questions (FAQs) and constructs
    a string where each question is followed by its corresponding answer and type.

    Parameters:
    - faq_dict (list): A list of dictionaries, each containing keys 'question', 'answer', and 'type'
      representing an FAQ entry.

    Returns:
    - str: A string representing the formatted layout of FAQs, with each entry on a separate line.
    """
    t = ""
    for f in faq_dict:
        t += f"Question: {f['question']} Answer: {f['answer']} Type: {f['type']}\n"
    return t


# ---------------------------------------------------------------------------
# From cell_59: parse_json_output
# ---------------------------------------------------------------------------


def parse_json_output(llm_output: str) -> dict | None:
    """
    Parses a string output from an LLM into a JSON object.

    This function attempts to clean and parse a JSON-formatted string produced by an LLM.
    The input string might contain minor formatting issues, such as unnecessary newlines or single quotes
    instead of double quotes. The function attempts to correct such issues before parsing.

    Parameters:
    - llm_output (str): The string output from the LLM that is expected to be in JSON format.

    Returns:
    - dict or None: A dictionary if parsing is successful, or None if the input string cannot be parsed into valid JSON.

    Exception Handling:
    - In case of a JSONDecodeError during parsing, an error message is printed, and the function returns None.
    """
    try:
        # Since the input might be improperly formatted, ensure any single quotes are removed
        llm_output = (
            llm_output.replace("\n", "").replace("'", "").replace("}}", "}").replace("{{", "{")
        )
        parsed_json = json.loads(llm_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None


# ---------------------------------------------------------------------------
# From cell_66 (adapted branch): get_filter_by_metadata
# ---------------------------------------------------------------------------


def get_filter_by_metadata(json_output: dict | None = None) -> dict | None:
    """
    Generate a Chroma where-filter dict based on a provided metadata dictionary.

    Parameters:
    - json_output (dict) or None: Dictionary containing metadata keys and their values.

    Returns:
    - dict or None: A Chroma-compatible where filter dict, or None if input is None.
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

    # Build a list of individual filter conditions
    conditions = []
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

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# From cell_78 (adapted branch): generate_items_context
# ---------------------------------------------------------------------------


def generate_items_context(results: list) -> str:
    """Compile product details from results into a formatted string.

    Adapted version that works on plain dicts (not Weaviate objects with .properties).

    Parameters:
    - results (list): A list of product dicts with keys like id, title, chunk, etc.

    Returns:
    - str: A multi-line string where each line contains the formatted details of a single product.
    """
    t = ""
    for item in results:
        t += (
            f"Product ID: {item['id']}. "
            f"Product name: {item['title']}. "
            f"Product Category: {item.get('masterCategory', '')}. "
            f"Product usage: {item.get('usage', '')}. "
            f"Product gender: {item.get('gender', '')}. "
            f"Product Type: {item.get('articleType', '')}. "
            f"Product Category: {item.get('subCategory', '')} "
            f"Product Color: {item.get('baseColour', '')}. "
            f"Product Season: {item.get('season', '')}. "
            f"Product Year: {item.get('year', '')}.\n"
        )
    return t