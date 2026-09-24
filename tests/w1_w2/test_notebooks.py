from .notebook_execution_plans import NOTEBOOK_EXECUTION_PLANS, VALID_DECISIONS
from .support import NOTEBOOKS, classify_code_cell, code_cells, function_source_from_notebook


def test_notebook_code_cells_are_classified():
    expected = {
        "safe syntax/import cells",
        "exercise function definition cells",
        "cells requiring Ollama",
        "cells requiring embeddings/model files",
        "widget/display-only cells",
        "cells that should be skipped in automated tests",
    }

    for stage, path in NOTEBOOKS.items():
        found = set()
        for _, source, metadata in code_cells(path):
            found.update(classify_code_cell(source, metadata))

        assert expected <= found, f"{stage} is missing expected notebook cell categories"


def test_key_notebook_functions_are_present():
    expected_functions = {
        "w1": [
            "query_news",
            "get_relevant_data",
            "format_relevant_data",
            "generate_final_prompt",
            "llm_call",
        ],
        "w2": [
            "query_news",
            "bm25_retrieve",
            "semantic_search_retrieve",
            "reciprocal_rank_fusion",
            "generate_final_prompt",
            "llm_call",
        ],
    }

    for stage, path in NOTEBOOKS.items():
        for function_name in expected_functions[stage]:
            assert function_source_from_notebook(path, function_name)


def test_every_notebook_code_cell_has_one_explicit_execution_decision():
    for stage, path in NOTEBOOKS.items():
        code_cell_indices = {index for index, _, _ in code_cells(path)}
        plan = NOTEBOOK_EXECUTION_PLANS[stage]

        assert set(plan) == code_cell_indices
        for index, (decision, reason) in plan.items():
            assert decision in VALID_DECISIONS, f"{stage} cell {index}: {decision}"
            assert reason.strip(), f"{stage} cell {index} needs a reason"
