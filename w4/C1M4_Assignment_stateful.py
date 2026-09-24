"""Stateful runner for the extracted C1M4 notebook code cells.

The raw extraction in ``C1M4_Assignment.py`` keeps the notebook code readable.
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
    "C1M4_Assignment",
    _module_dir / "C1M4_Assignment.py",
)
_raw_cells = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _raw_cells
_spec.loader.exec_module(_raw_cells)
raw_cells = _raw_cells


CellFunction = Callable[["NotebookState"], "NotebookState"]


@dataclass
class NotebookState:
    """Shared state that mirrors the notebook's global namespace."""

    adapted: bool = True
    client: Any | None = None
    store: Any | None = None
    products_data: Any | None = None
    faq: Any | None = None
    faq_layout: str | None = None
    values: Any | None = None
    products_collection: Any | None = None
    kwargs: Any | None = None
    response: Any | None = None
    content: Any | None = None
    label: str | None = None
    result: Any | None = None
    json_string: str | None = None
    json_output: Any | None = None
    filters: Any | None = None
    t: Any | None = None
    query: str | None = None
    queries: Any | None = None
    check_if_faq_or_product: Callable[..., Any] | None = None
    generate_faq_layout: Callable[..., Any] | None = None
    query_on_faq: Callable[..., Any] | None = None
    decide_task_nature: Callable[..., Any] | None = None
    get_params_for_task: Callable[..., Any] | None = None
    generate_metadata_from_query: Callable[..., Any] | None = None
    parse_json_output: Callable[..., Any] | None = None
    get_filter_by_metadata: Callable[..., Any] | None = None
    generate_filters_from_query: Callable[..., Any] | None = None
    get_relevant_products_from_query: Callable[..., Any] | None = None
    generate_items_context: Callable[..., Any] | None = None
    query_on_products: Callable[..., Any] | None = None
    answer_query: Callable[..., Any] | None = None
    namespace: dict[str, Any] = field(default_factory=dict)


_SYNC_NAMES = (
    "adapted",
    "client",
    "store",
    "products_data",
    "faq",
    "faq_layout",
    "values",
    "products_collection",
    "kwargs",
    "response",
    "content",
    "label",
    "result",
    "json_string",
    "json_output",
    "filters",
    "t",
    "query",
    "queries",
    "check_if_faq_or_product",
    "generate_faq_layout",
    "query_on_faq",
    "decide_task_nature",
    "get_params_for_task",
    "generate_metadata_from_query",
    "parse_json_output",
    "get_filter_by_metadata",
    "generate_filters_from_query",
    "get_relevant_products_from_query",
    "generate_items_context",
    "query_on_products",
    "answer_query",
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


def cell_07(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_07)


def cell_09(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_09)


def cell_10(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_10)


def cell_12(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_12)


def cell_13(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_13)


def cell_16(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_16)


def cell_17(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_17)


def cell_22(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_22)


def cell_23(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_23)


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


def cell_33(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_33)


def cell_34(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_34)


def cell_35(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_35)


def cell_37(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_37)


def cell_40(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_40)


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


def cell_49(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_49)


def cell_51(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_51)


def cell_52(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_52)


def cell_53(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_53)


def cell_55(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_55)


def cell_56(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_56)


def cell_59(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_59)


def cell_60(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_60)


def cell_61(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_61)


def cell_63(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_63)


def cell_64(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_64)


def cell_66(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_66)


def cell_68(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_68)


def cell_69(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_69)


def cell_70(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_70)


def cell_72(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_72)


def cell_73(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_73)


def cell_74(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_74)


def cell_75(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_75)


def cell_78(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_78)


def cell_79(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_79)


def cell_81(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_81)


def cell_82(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_82)


def cell_83(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_83)


def cell_84(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_84)


def cell_85(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_85)


def cell_87(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_87)


def cell_88(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_88)


def cell_89(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_89)


def cell_91(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_91)


CELL_SEQUENCE: tuple[tuple[int, CellFunction], ...] = (
    (4, cell_04),
    (5, cell_05),
    (7, cell_07),
    (9, cell_09),
    (10, cell_10),
    (12, cell_12),
    (13, cell_13),
    (16, cell_16),
    (17, cell_17),
    (22, cell_22),
    (23, cell_23),
    (25, cell_25),
    (27, cell_27),
    (29, cell_29),
    (30, cell_30),
    (32, cell_32),
    (33, cell_33),
    (34, cell_34),
    (35, cell_35),
    (37, cell_37),
    (40, cell_40),
    (41, cell_41),
    (42, cell_42),
    (44, cell_44),
    (46, cell_46),
    (47, cell_47),
    (49, cell_49),
    (51, cell_51),
    (52, cell_52),
    (53, cell_53),
    (55, cell_55),
    (56, cell_56),
    (59, cell_59),
    (60, cell_60),
    (61, cell_61),
    (63, cell_63),
    (64, cell_64),
    (66, cell_66),
    (68, cell_68),
    (69, cell_69),
    (70, cell_70),
    (72, cell_72),
    (73, cell_73),
    (74, cell_74),
    (75, cell_75),
    (78, cell_78),
    (79, cell_79),
    (81, cell_81),
    (82, cell_82),
    (83, cell_83),
    (84, cell_84),
    (85, cell_85),
    (87, cell_87),
    (88, cell_88),
    (89, cell_89),
    (91, cell_91),
)


def run_until(
    cell_number: int,
    state: NotebookState | None = None,
    skip_cells: Container[int] = frozenset(),
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
    """
    state = state or NotebookState()
    state.namespace["namespace"] = state.namespace  # self-reference
    state.namespace["adapted"] = state.adapted  # adapted flag for cell guard
    for current_cell_number, cell_function in CELL_SEQUENCE:
        if current_cell_number > cell_number:
            break
        if current_cell_number in skip_cells:
            continue
        cell_function(state)
    # Sync adapted back
    state.adapted = state.namespace.get("adapted", state.adapted)
    return state