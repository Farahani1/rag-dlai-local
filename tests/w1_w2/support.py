import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
W1_NOTEBOOK = PROJECT_ROOT / "w1" / "C1M1_Assignment.ipynb"
W2_NOTEBOOK = PROJECT_ROOT / "w2" / "C1M2_Assignment.ipynb"
NOTEBOOKS = {
    "w1": W1_NOTEBOOK,
    "w2": W2_NOTEBOOK,
}


class DummyPath:
    def __init__(self, value="dummy", exists=False):
        self.value = value
        self._exists = exists

    def exists(self):
        return self._exists

    def __str__(self):
        return self.value


class DummyArray:
    def reshape(self, *args):
        return self


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def encode(self, text):
        return DummyArray()


class DummyFrame:
    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        return None

    def apply(self, function):
        return self

    def to_dict(self, orient="records"):
        return [
            {
                "title": "Local RAG test",
                "description": "Small offline smoke data",
                "published_at": "2024-04-25",
                "updated_at": "2024-04-26",
                "url": "https://example.test",
            }
        ]


class DummyResponse:
    ok = True
    text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return {"response": " local answer "}


class DummyRequests(ModuleType):
    def __init__(self):
        super().__init__("requests")
        self.last_payload = None

    def post(self, url, json):
        self.last_payload = json
        return DummyResponse()


class TinyVector:
    def __init__(self, values):
        self.values = values

    def reshape(self, *args):
        return self


class TinyMatrix:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, column = item
            if rows != slice(None):
                raise TypeError("TinyMatrix only supports [:, column] indexing")
            return [row[column] for row in self.rows]
        return self.rows[item]

    def __len__(self):
        return len(self.rows)


class TinyScores:
    def __init__(self, values):
        self.values = values

    def __neg__(self):
        return TinyScores([-value for value in self.values])

    def __getitem__(self, item):
        return self.values[item]

    def __len__(self):
        return len(self.values)


def load_notebook(path):
    return nbformat.read(path, as_version=4)


def code_cells(path):
    notebook = load_notebook(path)
    return [
        (index, cell.source, cell.metadata)
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
    ]


def classify_code_cell(source, metadata):
    lowered = source.lower()
    categories = set()

    if source.strip():
        categories.add("safe syntax/import cells")
    if "graded" in metadata.get("tags", []) or "graded cell" in lowered:
        categories.add("exercise function definition cells")
    if "generate_with_single_input" in source or "llm_call" in source:
        categories.add("cells requiring Ollama")
    if (
        "SentenceTransformer" in source
        or "retrieve(" in source
        or "embeddings" in lowered
        or "semantic_search" in source
    ):
        categories.add("cells requiring embeddings/model files")
    if "display_widget" in source or "widgets" in source:
        categories.add("widget/display-only cells")

    executable = ast.parse(source)
    has_only_defs_imports_assignments = all(
        isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,
                ast.FunctionDef,
                ast.Import,
                ast.ImportFrom,
            ),
        )
        for node in executable.body
    )
    if not has_only_defs_imports_assignments or categories - {"safe syntax/import cells"}:
        categories.add("cells that should be skipped in automated tests")

    return categories


def function_source_from_notebook(path, function_name):
    for _, source, _ in code_cells(path):
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return ast.unparse(node)
    raise AssertionError(f"{function_name} not found in {path}")


def load_notebook_functions(path, names, namespace):
    for name in names:
        exec(function_source_from_notebook(path, name), namespace)
    return namespace


def install_import_stubs(monkeypatch):
    config = SimpleNamespace(
        hfLocalHub=DummyPath("hf-cache"),
        embeddingModel="dummy-embedding-model",
        ollama={"modelName": "dummy-llm", "url": "http://localhost:11434/api/generate"},
        newsEmbeddings=DummyPath("missing-embeddings.joblib"),
        newsCSV=DummyPath("news.csv"),
        completionOptions={"temperature": 0.2, "maxTokens": 50},
    )
    monkeypatch.setitem(sys.modules, "setting", SimpleNamespace(config=config))

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = DummySentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    pandas = ModuleType("pandas")
    pandas.read_csv = lambda path: DummyFrame()
    monkeypatch.setitem(sys.modules, "pandas", pandas)

    joblib = ModuleType("joblib")
    joblib.load = lambda path: []
    monkeypatch.setitem(sys.modules, "joblib", joblib)

    numpy = ModuleType("numpy")
    numpy.argsort = lambda values: list(range(len(values)))
    monkeypatch.setitem(sys.modules, "numpy", numpy)

    pairwise = ModuleType("sklearn.metrics.pairwise")
    pairwise.cosine_similarity = lambda query, embeddings: [[1.0]]
    metrics = ModuleType("sklearn.metrics")
    sklearn = ModuleType("sklearn")
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics)
    monkeypatch.setitem(sys.modules, "sklearn.metrics.pairwise", pairwise)

    requests = DummyRequests()
    monkeypatch.setitem(sys.modules, "requests", requests)

    widgets = ModuleType("ipywidgets")
    for name in ["Text", "Textarea", "IntSlider", "Output", "Button", "HBox", "Label", "Layout"]:
        setattr(
            widgets,
            name,
            lambda *args, **kwargs: SimpleNamespace(
                value="",
                clear_output=lambda: None,
                append_stdout=lambda text: None,
                on_click=lambda callback: None,
            ),
        )
    monkeypatch.setitem(sys.modules, "ipywidgets", widgets)

    ipython = ModuleType("IPython")
    display_module = ModuleType("IPython.display")
    display_module.display = lambda *args, **kwargs: None
    display_module.Markdown = lambda value: value
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)

    return requests


def configure_module_retrieval(module, scores):
    module.EMBEDDINGS = TinyMatrix([[0.0, 0.1], [0.9, 0.1], [0.3, 0.2]])
    module.EMBEDDINGS_PATH = DummyPath("embeddings.joblib", exists=True)
    module.model = SimpleNamespace(encode=lambda query: TinyVector([1.0, 0.0]))
    module.SentenceTransformer = lambda *args, **kwargs: SimpleNamespace(
        encode=lambda query: TinyVector([1.0, 0.0])
    )
    module.joblib = SimpleNamespace(load=lambda path: module.EMBEDDINGS)
    module.cosine_similarity = lambda query, embeddings: [TinyScores(scores)]
    module.np = SimpleNamespace(
        argsort=lambda values: sorted(range(len(values)), key=values.__getitem__)
    )


def import_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
