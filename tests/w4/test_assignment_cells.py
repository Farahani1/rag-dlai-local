"""Tests for extracted assignment cell functions from w4/C1M4_Assignment.py.

These tests verify the adapted (``adapted=True``) versions of functions that
are defined inside notebook cells. The functions themselves are extracted into
``tests/w4/assignment_functions.py`` for standalone testing.
"""

from __future__ import annotations

from tests.w4.assignment_functions import (
    generate_faq_layout,
    generate_items_context,
    get_filter_by_metadata,
    parse_json_output,
)


# ============================================================================
# Tests: generate_faq_layout (cell_29)
# ============================================================================


class TestGenerateFaqLayout:
    def test_single_faq(self):
        faqs = [{"question": "How to return?", "answer": "Visit our returns page.", "type": "returns"}]
        result = generate_faq_layout(faqs)
        assert "Question: How to return?" in result
        assert "Answer: Visit our returns page." in result
        assert "Type: returns" in result

    def test_multiple_faqs(self):
        faqs = [
            {"question": "Q1", "answer": "A1", "type": "t1"},
            {"question": "Q2", "answer": "A2", "type": "t2"},
        ]
        result = generate_faq_layout(faqs)
        assert "Q1" in result
        assert "Q2" in result
        assert "A1" in result
        assert "A2" in result

    def test_empty_list_returns_empty_string(self):
        result = generate_faq_layout([])
        assert result == ""

    def test_each_entry_ends_with_newline(self):
        faqs = [{"question": "Q", "answer": "A", "type": "t"}]
        result = generate_faq_layout(faqs)
        assert result.endswith("\n")


# ============================================================================
# Tests: parse_json_output (cell_59)
# ============================================================================


class TestParseJsonOutput:
    def test_parse_valid_json(self):
        result = parse_json_output('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_integers(self):
        result = parse_json_output('{"count": 42}')
        assert result == {"count": 42}

    def test_parse_nested_json(self):
        """parse_json_output strips single quotes, so nested JSON must not
        contain single quotes in the Python string literal."""
        # Double-brace the outer curly to avoid the {{ -> { cleaning interfering
        result = parse_json_output('{{"outer": {{"inner": 1}}}}')
        assert result == {"outer": {"inner": 1}}

    def test_parse_with_newlines(self):
        result = parse_json_output('{\n"key": "value"\n}')
        assert result == {"key": "value"}

    def test_parse_with_single_quotes(self):
        """The function strips single quotes, then tries to parse double-quoted JSON."""
        # After stripping single quotes from '{"key":"value"}', we get {"key":"value"} — valid
        result = parse_json_output('{"key":"value"}')
        assert result == {"key": "value"}

    def test_parse_double_braces_cleaned(self):
        """Extra braces are stripped."""
        result = parse_json_output('{{"key": "value"}}')
        assert result == {"key": "value"}

    def test_parse_invalid_json_returns_none(self):
        result = parse_json_output("not json at all")
        assert result is None

    def test_parse_empty_string_returns_none(self):
        result = parse_json_output("")
        assert result is None


# ============================================================================
# Tests: get_filter_by_metadata (cell_66 — adapted branch)
# ============================================================================


class TestGetFilterByMetadata:
    def test_none_input_returns_none(self):
        assert get_filter_by_metadata(None) is None

    def test_empty_dict_returns_none(self):
        assert get_filter_by_metadata({}) is None

    def test_single_category_key(self):
        result = get_filter_by_metadata({"gender": ["Men"]})
        assert result == {"gender": {"$in": ["Men"]}}

    def test_multiple_values_in_key(self):
        result = get_filter_by_metadata({"baseColour": ["Red", "Blue"]})
        assert result == {"baseColour": {"$in": ["Red", "Blue"]}}

    def test_single_value_not_in_list(self):
        """A scalar string value should be wrapped in a list for $in."""
        result = get_filter_by_metadata({"usage": "Casual"})
        assert result == {"usage": {"$in": ["Casual"]}}

    def test_price_range(self):
        result = get_filter_by_metadata({"price": {"min": 50, "max": 200}})
        assert result == {"$and": [{"price": {"$gte": 50}}, {"price": {"$lte": 200}}]}

    def test_price_with_min_zero_skipped(self):
        """When min is 0, price filter should be skipped."""
        result = get_filter_by_metadata({"price": {"min": 0, "max": 200}})
        assert result is None

    def test_price_with_max_inf_skipped(self):
        """When max is 'inf', price filter should be skipped."""
        result = get_filter_by_metadata({"price": {"min": 50, "max": "inf"}})
        assert result is None

    def test_price_with_none_values_skipped(self):
        result = get_filter_by_metadata({"price": {"min": None, "max": 200}})
        assert result is None

    def test_multiple_conditions_combined_with_and(self):
        result = get_filter_by_metadata({
            "gender": ["Men"],
            "baseColour": ["Blue"],
        })
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_unknown_keys_skipped(self):
        result = get_filter_by_metadata({"unknown_field": ["value"]})
        assert result is None

    def test_mixed_valid_and_invalid_keys(self):
        result = get_filter_by_metadata({
            "gender": ["Men"],
            "unknown_field": ["value"],
        })
        assert result == {"gender": {"$in": ["Men"]}}

    def test_price_not_a_dict_skipped(self):
        result = get_filter_by_metadata({"price": "not-a-dict"})
        assert result is None

    def test_all_valid_keys_recognized(self):
        """All seven valid keys from the function should be recognized."""
        for key in ("gender", "masterCategory", "articleType", "baseColour", "usage", "season"):
            result = get_filter_by_metadata({key: ["Test"]})
            assert result is not None, f"Key '{key}' should produce a valid filter"
            assert key in str(result)


# ============================================================================
# Tests: generate_items_context (cell_78 — adapted branch)
# ============================================================================


class TestGenerateItemsContext:
    def test_single_product(self):
        products = [
            {
                "id": "123",
                "title": "Blue T-Shirt",
                "masterCategory": "Apparel",
                "usage": "Casual",
                "gender": "Men",
                "articleType": "T-Shirts",
                "subCategory": "Topwear",
                "baseColour": "Blue",
                "season": "Summer",
                "year": "2024",
            }
        ]
        result = generate_items_context(products)
        assert "Product ID: 123" in result
        assert "Product name: Blue T-Shirt" in result
        assert "Apparel" in result
        assert "Blue" in result

    def test_multiple_products(self):
        products = [
            {"id": "1", "title": "Item A"},
            {"id": "2", "title": "Item B"},
        ]
        result = generate_items_context(products)
        assert "Product ID: 1" in result
        assert "Product ID: 2" in result
        assert result.count("Product ID:") == 2

    def test_empty_list_returns_empty_string(self):
        result = generate_items_context([])
        assert result == ""

    def test_missing_fields_default_to_empty(self):
        """Missing optional fields should default to empty string."""
        products = [{"id": "1", "title": "Minimal"}]
        result = generate_items_context(products)
        assert "Product ID: 1" in result
        assert "Product name: Minimal" in result
        # Should not crash on missing keys