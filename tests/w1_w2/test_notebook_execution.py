from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from .support import W1_NOTEBOOK, W2_NOTEBOOK, code_cells


def cell_source_containing_function(path, function_name):
    needle = f"def {function_name}"
    for _, source, _ in code_cells(path):
        if needle in source:
            return source
    raise AssertionError(f"{function_name} not found in {path}")


def execute_selected_cells(sources, tmp_path):
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
    notebook_path = tmp_path / "selected_cells.ipynb"
    client = NotebookClient(
        notebook,
        timeout=30,
        kernel_name="python3",
        resources={"metadata": {"path": str(Path.cwd())}},
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    return notebook_path


@pytest.mark.notebook
def test_w1_selected_notebook_cells_execute_with_tiny_stubs(tmp_path):
    sources = [
        """
NEWS_DATA = [
    {
        "title": "Local retrieval",
        "description": "A tiny smoke document",
        "published_at": "2024-04-25",
        "url": "https://example.test/local",
    },
    {
        "title": "Second item",
        "description": "Another tiny smoke document",
        "published_at": "2024-04-26",
        "url": "https://example.test/second",
    },
]

def retrieve(query, top_k=5):
    return list(range(top_k))
""",
        cell_source_containing_function(W1_NOTEBOOK, "query_news"),
        cell_source_containing_function(W1_NOTEBOOK, "get_relevant_data"),
        cell_source_containing_function(W1_NOTEBOOK, "format_relevant_data"),
        cell_source_containing_function(W1_NOTEBOOK, "generate_final_prompt"),
        """
assert query_news([1]) == [NEWS_DATA[1]]
assert get_relevant_data("local", top_k=2) == NEWS_DATA

formatted = format_relevant_data([NEWS_DATA[0]])
assert "Local retrieval" in formatted
assert "description" in formatted.lower()
assert "url" in formatted.lower()

assert generate_final_prompt("What happened?", use_rag=False) == "What happened?"
prompt = generate_final_prompt(
    "What happened?",
    top_k=1,
    prompt="Question: {query}\\nDocs: {documents}",
)
assert "Question: What happened?" in prompt
assert "Local retrieval" in prompt
""",
    ]

    execute_selected_cells(sources, tmp_path)


@pytest.mark.notebook
def test_w2_selected_notebook_cells_execute_with_tiny_stubs(tmp_path):
    sources = [
        """
class TinyScores:
    def __init__(self, values):
        self.values = values

    def __neg__(self):
        return TinyScores([-value for value in self.values])

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return len(self.values)


class TinyVector:
    def reshape(self, *args):
        return self


class TinyMatrix:
    def __getitem__(self, item):
        return [[0.0, 0.1], [0.9, 0.1], [0.3, 0.2]][item]


class TinyBM25Retriever:
    def index(self, tokenized_data):
        self.indexed = True

    def retrieve(self, tokenized_query, k=5):
        return [corpus[:k]], [[1.0] * k]


class TinyBM25Module:
    @staticmethod
    def tokenize(texts):
        return texts


class TinyNumpy:
    @staticmethod
    def argsort(values):
        return sorted(range(len(values)), key=values.__getitem__)


def cosine_similarity(query, embeddings):
    return [TinyScores([0.2, 0.9, 0.4])]


NEWS_DATA = [
    {
        "title": "GDP growth report",
        "description": "A tiny economic smoke document",
        "published_at": "2024-04-25",
        "url": "https://example.test/gdp",
    },
    {
        "title": "Music tour revenue",
        "description": "A tiny music smoke document",
        "published_at": "2024-04-26",
        "url": "https://example.test/music",
    },
]
corpus = ["GDP growth report", "music tour revenue", "climate policy"]
BM25_RETRIEVER = TinyBM25Retriever()
TOKENIZED_DATA = ["GDP", "music", "climate"]
bm25s = TinyBM25Module()
model = type("Model", (), {"encode": lambda self, query: TinyVector()})()
EMBEDDINGS = TinyMatrix()
np = TinyNumpy()
""",
        cell_source_containing_function(W2_NOTEBOOK, "query_news"),
        cell_source_containing_function(W2_NOTEBOOK, "bm25_retrieve"),
        cell_source_containing_function(W2_NOTEBOOK, "semantic_search_retrieve"),
        cell_source_containing_function(W2_NOTEBOOK, "reciprocal_rank_fusion"),
        cell_source_containing_function(W2_NOTEBOOK, "generate_final_prompt"),
        """
assert query_news([1]) == [NEWS_DATA[1]]
assert bm25_retrieve("GDP", top_k=2) == [0, 1]
assert semantic_search_retrieve("GDP", top_k=2) == [1, 2]
assert reciprocal_rank_fusion([1, 2, 3], [2, 1, 4], top_k=3) == [1, 2, 3]

def fake_retrieve(query, top_k=5):
    return list(range(top_k))

assert generate_final_prompt(
    "What happened?",
    top_k=1,
    retrieve_function=fake_retrieve,
    use_rag=False,
) == "What happened?"

prompt = generate_final_prompt(
    "What happened?",
    top_k=2,
    retrieve_function=fake_retrieve,
    use_rag=True,
)
assert "Query: What happened?" in prompt
assert "GDP growth report" in prompt
assert "Music tour revenue" in prompt
""",
    ]

    execute_selected_cells(sources, tmp_path)
