"""Stateful runner for the extracted C1M3 notebook code cells.

The raw extraction in ``C1M3_Assignment.py`` keeps the notebook code readable.
This module executes those extracted cell bodies in a shared namespace so tests
can simulate the way a notebook carries state from one cell to the next.

When ``adapted=True``, the runner sets the ``adapted`` flag in the cell
namespace so each cell function uses Chroma-backed implementations instead of
Weaviate.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from collections.abc import Callable, Container
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Load sibling module via importlib to avoid package-name conflicts with pytest.
_module_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "C1M3_Assignment",
    _module_dir / "C1M3_Assignment.py",
)
_raw_cells = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _raw_cells
_spec.loader.exec_module(_raw_cells)
raw_cells = _raw_cells


CellFunction = Callable[["NotebookState"], "NotebookState"]


@dataclass
class NotebookState:
    """Shared state that mirrors the notebook's global namespace."""

    client: Any | None = None
    store: Any | None = None
    collection: Any | None = None
    bbc_data: Any | None = None
    object: Any | None = None
    res: Any | None = None
    results: Any | None = None
    query: str | None = None
    prompt: str | None = None
    filter_by_metadata: Callable[..., Any] | None = None
    semantic_search_retrieve: Callable[..., Any] | None = None
    bm25_retrieve: Callable[..., Any] | None = None
    hybrid_retrieve: Callable[..., Any] | None = None
    semantic_search_with_reranking: Callable[..., Any] | None = None
    generate_final_prompt: Callable[..., Any] | None = None
    llm_call: Callable[..., Any] | None = None
    namespace: dict[str, Any] = field(default_factory=dict)


_SYNC_NAMES = (
    "client",
    "store",
    "collection",
    "bbc_data",
    "object",
    "res",
    "results",
    "query",
    "prompt",
    "filter_by_metadata",
    "semantic_search_retrieve",
    "bm25_retrieve",
    "hybrid_retrieve",
    "semantic_search_with_reranking",
    "generate_final_prompt",
    "llm_call",
)


def _prepare_namespace(state: NotebookState) -> None:
    for name in _SYNC_NAMES:
        value = getattr(state, name)
        if value is not None:
            state.namespace[name] = value


def _sync_state(state: NotebookState) -> NotebookState:
    for name in _SYNC_NAMES:
        if name in state.namespace:
            setattr(state, name, state.namespace[name])
    return state


def _function_body_code(raw_cell: Callable[..., None]) -> Any:
    """Extract function body, stripping signature and docstring.

    Prepends an ``adapted = namespace.get('adapted', False)`` guard so the
    function body uses Chroma-backed implementations when the namespace has
    ``adapted=True``.
    """
    source = inspect.getsource(raw_cell)
    module = ast.parse(source)
    function_def = module.body[0]
    if not isinstance(function_def, ast.FunctionDef):
        raise TypeError(f"Expected a function, got {type(function_def).__name__}")

    body = function_def.body
    # Strip docstring
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    # Prepend the adapted guard
    adapted_guard = ast.parse(
        "adapted = namespace.get('adapted', False)"
    ).body
    body = adapted_guard + body

    executable_module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(executable_module)
    return compile(executable_module, inspect.getsourcefile(raw_cell) or "<cell>", "exec")


def _run_raw_cell(state: NotebookState, raw_cell: Callable[..., None]) -> NotebookState:
    _prepare_namespace(state)
    exec(_function_body_code(raw_cell), state.namespace)
    return _sync_state(state)


def cell_04(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_04)


def cell_05(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_05)


def cell_08(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_08)


def cell_10(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_10)


def cell_11(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_11)


def cell_13(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_13)


def cell_14(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_14)


def cell_16(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_16)


def cell_19(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_19)


def cell_20(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_20)


def cell_22(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_22)


def cell_24(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_24)


def cell_25(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_25)


def cell_27(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_27)


def cell_29(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_29)


def cell_30(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_30)


def cell_32(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_32)


def cell_34(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_34)


def cell_35(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_35)


def cell_37(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_37)


def cell_39(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_39)


def cell_41(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_41)


def cell_42(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_42)


def cell_44(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_44)


def cell_46(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_46)


def cell_47(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_47)


def cell_48(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_48)


def cell_50(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_50)


def cell_51(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_51)


def cell_52(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_52)


def cell_54(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_54)


CELL_SEQUENCE: tuple[tuple[int, CellFunction], ...] = (
    (4, cell_04),
    (5, cell_05),
    (8, cell_08),
    (10, cell_10),
    (11, cell_11),
    (13, cell_13),
    (14, cell_14),
    (16, cell_16),
    (19, cell_19),
    (20, cell_20),
    (22, cell_22),
    (24, cell_24),
    (25, cell_25),
    (27, cell_27),
    (29, cell_29),
    (30, cell_30),
    (32, cell_32),
    (34, cell_34),
    (35, cell_35),
    (37, cell_37),
    (39, cell_39),
    (41, cell_41),
    (42, cell_42),
    (44, cell_44),
    (46, cell_46),
    (47, cell_47),
    (48, cell_48),
    (50, cell_50),
    (51, cell_51),
    (52, cell_52),
    (54, cell_54),
)


def run_until(
    cell_number: int,
    state: NotebookState | None = None,
    skip_cells: Container[int] = frozenset(),
    adapted: bool = False,
) -> NotebookState:
    """Run notebook cells in order through ``cell_number`` inclusive.

    Parameters
    ----------
    cell_number :
        Run all cells up to and including this cell number.
    state :
        An optional initial ``NotebookState``.  Created fresh if ``None``.
    skip_cells :
        A set of cell numbers to skip.
    adapted :
        If ``True``, uses Chroma-backed implementations instead of Weaviate.
    """
    state = state or NotebookState()
    state.namespace["adapted"] = adapted
    state.namespace["namespace"] = state.namespace  # self-reference for adapted guard
    for current_cell_number, cell_function in CELL_SEQUENCE:
        if current_cell_number > cell_number:
            break
        if current_cell_number in skip_cells:
            continue
        cell_function(state)
    return state