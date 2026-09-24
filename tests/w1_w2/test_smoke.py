import py_compile

from .support import (
    NOTEBOOKS,
    PROJECT_ROOT,
    code_cells,
    import_from_path,
    install_import_stubs,
)


def test_w1_and_w2_utils_compile():
    py_compile.compile(PROJECT_ROOT / "w1" / "utils.py", doraise=True)
    py_compile.compile(PROJECT_ROOT / "w2" / "utils.py", doraise=True)


def test_w1_and_w2_utils_import_with_stubs(monkeypatch):
    install_import_stubs(monkeypatch)

    w1_utils = import_from_path("w1_smoke_utils", PROJECT_ROOT / "w1" / "utils.py")
    w2_utils = import_from_path("w2_smoke_utils", PROJECT_ROOT / "w2" / "utils.py")

    assert hasattr(w1_utils, "generate_with_single_input")
    assert hasattr(w1_utils, "retrieve")
    assert hasattr(w2_utils, "generate_with_single_input")
    assert hasattr(w2_utils, "retrieve")


def test_w1_and_w2_notebooks_have_parseable_code_cells():
    for stage, path in NOTEBOOKS.items():
        cells = code_cells(path)

        assert cells, f"{stage} should contain executable code cells"
        for cell_index, source, _ in cells:
            try:
                compile(source, f"{path}:cell-{cell_index}", "exec")
            except SyntaxError as exc:
                raise AssertionError(f"{path} code cell {cell_index} does not parse") from exc
