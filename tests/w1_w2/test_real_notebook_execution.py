from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from .notebook_execution_plans import EXECUTE_REAL, NOTEBOOK_EXECUTION_PLANS
from .support import NOTEBOOKS, code_cells


pytestmark = pytest.mark.local_integration


def selected_real_sources(stage):
    planned = NOTEBOOK_EXECUTION_PLANS[stage]
    return [
        source
        for index, source, _ in code_cells(NOTEBOOKS[stage])
        if planned[index][0] == EXECUTE_REAL
    ]


def execute_real_cells(stage, assertion_source, tmp_path):
    sources = [*selected_real_sources(stage), assertion_source]
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source) for source in sources],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    output_path = tmp_path / f"{stage}_real_selected_cells.ipynb"
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT / stage)}},
    )
    client.execute()
    nbformat.write(notebook, output_path)
    return output_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_w1_selected_real_notebook_cells_execute(tmp_path):
    output_path = execute_real_cells(
        "w1",
        """
documents = get_relevant_data("economic growth and GDP", top_k=2)
assert len(documents) == 2
formatted = format_relevant_data(documents)
assert "Title:" in formatted
prompt = generate_final_prompt("economic growth and GDP", top_k=2)
assert "2024 News:" in prompt
assert "economic growth and GDP" in prompt
""",
        tmp_path,
    )
    assert output_path.is_file()


def test_w2_selected_real_notebook_cells_execute(tmp_path):
    output_path = execute_real_cells(
        "w2",
        """
bm25_indices = bm25_retrieve("economic growth and GDP", top_k=2)
semantic_indices = semantic_search_retrieve("economic growth and GDP", top_k=2)
fused_indices = reciprocal_rank_fusion(bm25_indices, semantic_indices, top_k=2)
assert len(bm25_indices) == 2
assert len(semantic_indices) == 2
assert len(fused_indices) == 2
prompt = generate_final_prompt(
    "economic growth and GDP",
    top_k=2,
    retrieve_function=reciprocal_rank_fusion,
)
assert "2024 News:" in prompt
assert "economic growth and GDP" in prompt
""",
        tmp_path,
    )
    assert output_path.is_file()
