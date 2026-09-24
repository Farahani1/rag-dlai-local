"""Stateful runner for the extracted C1M5 notebook code cells.

The raw extraction in ``C1M5_Assignment.py`` keeps the notebook code readable.
This module executes those extracted cell bodies in a shared namespace so tests
can simulate the way a notebook carries state from one cell to the next.

When ``adapted=True``, the runner sets the ``adapted`` flag in the cell
namespace so each cell function uses Chroma/Ollama-backed implementations and
a local no-op tracer instead of Weaviate/Together.ai/Arize Phoenix.
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
    "C1M5_Assignment",
    _module_dir / "C1M5_Assignment.py",
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
    session: Any | None = None
    tracer: Any | None = None
    products_data: Any | None = None
    faq: Any | None = None
    faq_layout: str | None = None
    values: Any | None = None
    products_collection: Any | None = None
    products_collection_name: str | None = None
    faq_collection: Any | None = None
    res: Any | None = None
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
    labels: Any | None = None
    total_tokens: Any | None = None
    chat_widget_standard: Any | None = None
    chat_widget_simplified: Any | None = None
    generate_params_dict: Callable[..., Any] | None = None
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
    "session",
    "tracer",
    "products_data",
    "faq",
    "faq_layout",
    "values",
    "products_collection",
    "products_collection_name",
    "faq_collection",
    "res",
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
    "labels",
    "total_tokens",
    "chat_widget_standard",
    "chat_widget_simplified",
    "generate_params_dict",
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
    function body uses Chroma/Ollama-backed implementations when the
    namespace has ``adapted=True``.
    """
    source = inspect.getsource(raw_cell)
    module = ast.parse(source)
    function_def = module.body[0]
    if not isinstance(function_def, ast.FunctionDef):
        raise TypeError(f"Expected a function, got {type(function_def).__name__}")

    body = function_def.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

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


def cell_03(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_03)


def cell_04(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_04)


def cell_06(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_06)


def cell_08(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_08)


def cell_09(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_09)


def cell_11(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_11)


def cell_14(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_14)


def cell_17(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_17)


def cell_18(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_18)


def cell_21(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_21)


def cell_22(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_22)


def cell_26(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_26)


def cell_27(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_27)


def cell_28(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_28)


def cell_32(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_32)


def cell_33(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_33)


def cell_34(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_34)


def cell_37(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_37)


def cell_38(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_38)


def cell_40(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_40)


def cell_42(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_42)


def cell_43(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_43)


def cell_44(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_44)


def cell_46(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_46)


def cell_48(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_48)


def cell_50(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_50)


def cell_51(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_51)


def cell_53(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_53)


def cell_54(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_54)


def cell_55(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_55)


def cell_56(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_56)


def cell_58(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_58)


def cell_60(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_60)


def cell_61(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_61)


def cell_63(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_63)


def cell_64(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_64)


def cell_65(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_65)


def cell_66(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_66)


def cell_67(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_67)


def cell_70(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_70)


def cell_71(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_71)


def cell_72(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_72)


def cell_73(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_73)


def cell_75(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_75)


def cell_77(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_77)


def cell_79(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_79)


def cell_80(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_80)


def cell_82(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_82)


def cell_83(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_83)


def cell_84(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_84)


def cell_85(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_85)


def cell_88(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_88)


def cell_89(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_89)


def cell_91(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_91)


def cell_94(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_94)


def cell_95(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_95)


def cell_96(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_96)


def cell_97(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_97)


def cell_99(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_99)


def cell_100(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_100)


def cell_102(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_102)


def cell_104(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_104)


def cell_105(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_105)


def cell_107(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_107)


def cell_109(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_109)


def cell_110(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_110)


def cell_112(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_112)


def cell_114(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_114)


def cell_115(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_115)


def cell_116(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_116)


def cell_117(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_117)


def cell_120(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_120)


def cell_121(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_121)


def cell_122(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_122)


def cell_123(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_123)


def cell_124(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_124)


def cell_125(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_125)


def cell_126(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_126)


def cell_128(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_128)


def cell_130(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_130)


def cell_132(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_132)


def cell_133(state: NotebookState) -> NotebookState:
    return _run_raw_cell(state, raw_cells.cell_133)


CELL_SEQUENCE: tuple[tuple[int, CellFunction], ...] = (
    (3, cell_03),
    (4, cell_04),
    (6, cell_06),
    (8, cell_08),
    (9, cell_09),
    (11, cell_11),
    (14, cell_14),
    (17, cell_17),
    (18, cell_18),
    (21, cell_21),
    (22, cell_22),
    (26, cell_26),
    (27, cell_27),
    (28, cell_28),
    (32, cell_32),
    (33, cell_33),
    (34, cell_34),
    (37, cell_37),
    (38, cell_38),
    (40, cell_40),
    (42, cell_42),
    (43, cell_43),
    (44, cell_44),
    (46, cell_46),
    (48, cell_48),
    (50, cell_50),
    (51, cell_51),
    (53, cell_53),
    (54, cell_54),
    (55, cell_55),
    (56, cell_56),
    (58, cell_58),
    (60, cell_60),
    (61, cell_61),
    (63, cell_63),
    (64, cell_64),
    (65, cell_65),
    (66, cell_66),
    (67, cell_67),
    (70, cell_70),
    (71, cell_71),
    (72, cell_72),
    (73, cell_73),
    (75, cell_75),
    (77, cell_77),
    (79, cell_79),
    (80, cell_80),
    (82, cell_82),
    (83, cell_83),
    (84, cell_84),
    (85, cell_85),
    (88, cell_88),
    (89, cell_89),
    (91, cell_91),
    (94, cell_94),
    (95, cell_95),
    (96, cell_96),
    (97, cell_97),
    (99, cell_99),
    (100, cell_100),
    (102, cell_102),
    (104, cell_104),
    (105, cell_105),
    (107, cell_107),
    (109, cell_109),
    (110, cell_110),
    (112, cell_112),
    (114, cell_114),
    (115, cell_115),
    (116, cell_116),
    (117, cell_117),
    (120, cell_120),
    (121, cell_121),
    (122, cell_122),
    (123, cell_123),
    (124, cell_124),
    (125, cell_125),
    (126, cell_126),
    (128, cell_128),
    (130, cell_130),
    (132, cell_132),
    (133, cell_133),
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
