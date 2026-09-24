"""Run the benchmark question set through the W5 pipeline for one model.

    python benchmark/run.py --model qwen2.5:1.5b --repeats 3
    python benchmark/run.py --model qwen2.5:1.5b --repeats 1 --ids d1   # smoke run

Every question goes through W5's ``answer_query(query, simplified=False)``
followed by the final generation call, exactly as in the notebook. The
notebook's own function definitions are loaded with the W5 stateful runner;
only ``generate_with_single_input`` is wrapped, so that every LLM call in the
pipeline uses the model under test at temperature 0.

Output: ``benchmark/results/<model>_<date>.csv``. Lines starting with ``#``
at the top record the run conditions (model digest, CPU, RAM, date); read
the CSV with ``pandas.read_csv(path, comment=None, skiprows=<n>)`` or with
``benchmark.run.read_results``.

Requires a running Ollama server with the model pulled, and the
``products_w5`` Chroma collection (``python w5/populate_products.py``).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
W5_DIR = PROJECT_ROOT / "w5"
for p in (str(PROJECT_ROOT), str(W5_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark.scoring import score  # noqa: E402
from benchmark.variants import VARIANTS, apply_variant  # noqa: E402

QUESTIONS_PATH = PROJECT_ROOT / "benchmark" / "questions.yaml"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"

# W5 cells that set up state or define functions; demo cells that call the
# LLM on the notebook's own examples are skipped.
SETUP_CELLS = frozenset({
    3, 4, 6, 8, 14, 17, 21, 32, 37, 42, 43, 46, 48, 53,
    70, 75, 79, 82, 88, 91, 94, 104, 107, 120,
})
LAST_SETUP_CELL = max(SETUP_CELLS)

COLUMNS = [
    "model", "variant", "question_id", "category", "repeat", "route",
    "latency_s", "filter_json_parsed", "fallback", "filters",
    "filter_raw_output", "answer", "correct", "manual_override", "latency_excluded", "error",
]
# manual_override (y/n) and latency_excluded (a reason, e.g. "laptop slept") are
# filled in by hand after a run; the summaries honour both.


# ---------------------------------------------------------------------------
# Run conditions
# ---------------------------------------------------------------------------


def _ollama_base_url() -> str:
    from setting import config

    return config.ollama["url"].split("/api/")[0]


def model_digest(model: str) -> str:
    """Return the model's digest from Ollama, or raise if it is not available."""
    resp = requests.get(f"{_ollama_base_url()}/api/tags", timeout=10)
    resp.raise_for_status()
    for entry in resp.json().get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            return entry.get("digest", "")
    raise SystemExit(f"Model '{model}' is not pulled in Ollama (run: ollama pull {model}).")


def total_ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 1024**3, 1)
        with open("/proc/meminfo", encoding="utf-8") as f:
            kb = int(f.readline().split()[1])
        return round(kb / 1024**2, 1)
    except Exception:
        return None


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Recorder:
    """Collects what happened during one question's pipeline run."""

    def __init__(self) -> None:
        self.filter_raw_output = ""
        self.filter_json_parsed = "na"
        self.filters: Any = None
        self.used_filter: bool | None = None

    @property
    def fallback(self) -> str:
        if self.used_filter is None:
            return "na"
        return "n" if self.used_filter else "y"


def load_pipeline(model: str) -> tuple[dict[str, Any], list[Recorder]]:
    """Execute W5's setup cells and wrap the calls the benchmark observes.

    Returns the notebook namespace and a one-element list holding the
    current ``Recorder`` (replace ``holder[0]`` before each question).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "C1M5_Assignment_stateful", W5_DIR / "C1M5_Assignment_stateful.py"
    )
    stateful = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stateful
    spec.loader.exec_module(stateful)

    all_cells = {number for number, _ in stateful.CELL_SEQUENCE}
    state = stateful.NotebookState(adapted=True)
    stateful.run_until(LAST_SETUP_CELL, state=state, skip_cells=all_cells - SETUP_CELLS)
    ns = state.namespace
    holder = [Recorder()]

    original_generate = ns["generate_with_single_input"]
    model_under_test = model

    def generate_fixed(prompt, role="user", top_p=None, temperature=None,
                       max_tokens=None, model=None, **kwargs):
        # Ignore the caller's model and temperature: every call uses the model under test at 0.
        return original_generate(
            prompt, role=role, top_p=top_p, temperature=0,
            max_tokens=max_tokens, model=model_under_test, **kwargs,
        )

    ns["generate_with_single_input"] = generate_fixed

    original_metadata = ns["generate_metadata_from_query"]

    def metadata_recorded(query):
        content, tokens = original_metadata(query)
        holder[0].filter_raw_output = content
        return content, tokens

    ns["generate_metadata_from_query"] = metadata_recorded

    original_parse = ns["parse_json_output"]

    def parse_recorded(llm_output):
        parsed = original_parse(llm_output)
        holder[0].filter_json_parsed = "y" if parsed is not None else "n"
        return parsed

    ns["parse_json_output"] = parse_recorded

    original_filters = ns["generate_filters_from_query"]

    def filters_recorded(query):
        filters, tokens = original_filters(query)
        holder[0].filters = filters
        return filters, tokens

    ns["generate_filters_from_query"] = filters_recorded

    # The last store call in get_relevant_products_from_query decides whether
    # the final product list came from a filtered or an unfiltered search.
    store = ns["store"]
    original_query, original_query_with_filter = store.query, store.query_with_filter

    def query_recorded(collection, *args, **kwargs):
        if collection == "products_w5":
            holder[0].used_filter = False
        return original_query(collection, *args, **kwargs)

    def query_with_filter_recorded(collection, *args, **kwargs):
        if collection == "products_w5":
            holder[0].used_filter = True
        return original_query_with_filter(collection, *args, **kwargs)

    store.query = query_recorded
    store.query_with_filter = query_with_filter_recorded

    return ns, holder


def ask(ns: dict[str, Any], question: str) -> tuple[str, str]:
    """Run one question end to end; return (route, final answer)."""
    route_label: list[str] = []
    original_check = ns["check_if_faq_or_product"]

    def check_recorded(query, simplified=False):
        label, tokens = original_check(query, simplified=simplified)
        route_label.append(label)
        return label, tokens

    ns["check_if_faq_or_product"] = check_recorded
    try:
        out = ns["answer_query"](question, simplified=False)
    finally:
        ns["check_if_faq_or_product"] = original_check

    # answer_query returns a bare dict (no token count) for unrecognised labels.
    kwargs = out[0] if isinstance(out, tuple) else out
    result = ns["generate_with_single_input"](**kwargs)
    answer = result["choices"][0]["message"]["content"]
    return (route_label[0] if route_label else ""), answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_catalogue() -> dict[int, dict[str, Any]]:
    import joblib
    from setting import config

    return {int(p["product_id"]): p for p in joblib.load(str(config.clothesData))}


def results_path(model: str, date: str, variant: str = "baseline") -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    suffix = "" if variant == "baseline" else f"_{variant}"
    return RESULTS_DIR / f"{safe}_{date}{suffix}.csv"


def read_results(path: Path):
    """Read a results CSV, skipping the ``#`` run-condition lines."""
    import pandas as pd

    with open(path, encoding="utf-8") as f:
        skip = 0
        for line in f:
            if not line.startswith("#"):
                break
            skip += 1
    df = pd.read_csv(path, skiprows=skip, keep_default_na=False)
    for column in ("manual_override", "latency_excluded"):
        if column not in df.columns:
            df[column] = ""
    if "variant" not in df.columns:  # CSVs written before variants existed
        df.insert(1, "variant", "baseline")
    return df


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (a sleeping laptop stalls a run).

    Uses SetThreadExecutionState; nothing persists after the process exits, and no
    power settings are changed. No-op on other platforms.
    """
    if sys.platform != "win32":
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Ollama model tag, e.g. qwen2.5:1.5b")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ids", nargs="*", help="Only run these question ids (smoke runs)")
    parser.add_argument("--variant", default="baseline", choices=list(VARIANTS),
                        help="Pipeline variant (see benchmark/variants.py)")
    parser.add_argument("--out", type=Path, help="Output CSV (default: benchmark/results/<model>_<date>[_<variant>].csv)")
    args = parser.parse_args(argv)

    spec = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.ids:
        unknown = set(args.ids) - {q["id"] for q in questions}
        if unknown:
            parser.error(f"unknown question ids: {sorted(unknown)}")
        questions = [q for q in questions if q["id"] in args.ids]

    digest = model_digest(args.model)
    keep_awake()
    started = dt.datetime.now()
    out = args.out or results_path(args.model, started.strftime("%Y-%m-%d"), args.variant)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading W5 pipeline for {args.model} ...", flush=True)
    ns, holder = load_pipeline(args.model)
    variant_info = apply_variant(args.variant, ns)
    catalogue = load_catalogue()

    # Load the LLM and the embedding model before timing, so the first
    # question's latency doesn't include model loading.
    ns["embed_query"]("warm-up")
    ns["generate_with_single_input"]("Reply with OK.", max_tokens=5)

    header = {
        "model": args.model,
        "model_digest": digest,
        "question_set_version": spec.get("version"),
        "questions": ",".join(q["id"] for q in questions),
        "repeats": args.repeats,
        "variant": args.variant,
        **variant_info,
        "temperature": 0,
        "cpu": cpu_name(),
        "ram_gb": total_ram_gb(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "started": started.isoformat(timespec="seconds"),
    }

    with open(out, "w", encoding="utf-8", newline="") as f:
        for key, value in header.items():
            f.write(f"# {key}: {value}\n")
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for repeat in range(1, args.repeats + 1):
            for q in questions:
                holder[0] = Recorder()
                row = {"model": args.model, "variant": args.variant, "question_id": q["id"],
                       "category": q["category"], "repeat": repeat,
                       "manual_override": "", "latency_excluded": "", "error": ""}
                start = time.perf_counter()
                try:
                    route, answer = ask(ns, q["question"])
                except Exception as exc:  # recorded, not fatal: one bad run must not stop the set
                    route, answer = "", ""
                    row["error"] = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - start
                rec = holder[0]
                row.update({
                    "route": route,
                    "latency_s": f"{elapsed:.2f}",
                    "filter_json_parsed": rec.filter_json_parsed,
                    "fallback": rec.fallback,
                    "filters": json.dumps(rec.filters) if rec.filters is not None else "",
                    "filter_raw_output": rec.filter_raw_output,
                    "answer": answer,
                    "correct": "" if row["error"] else ("y" if score(q["expected"], answer, catalogue) else "n"),
                })
                writer.writerow(row)
                f.flush()
                print(f"[{repeat}/{args.repeats}] {q['id']:>3} {row['latency_s']:>7}s "
                      f"route={route or '-'} filter={rec.filter_json_parsed} "
                      f"fallback={rec.fallback} correct={row['correct'] or 'ERR'}", flush=True)

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
