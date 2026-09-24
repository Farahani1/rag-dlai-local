"""Split each benchmark question's latency into pipeline steps, using Ollama's server log.

    python benchmark/step_times.py benchmark/results/<model>_<date>.csv
    python benchmark/step_times.py <csv> --log path/to/server.log --out steps.csv

The runner only records total latency per question. Ollama's server log
records every /api/generate call with its end time and duration, and each
question makes its LLM calls in a fixed order:

    FAQ route:      route, answer
    Product route:  route, task (decide_task_nature), filter (JSON metadata), answer

The router variants skip the routing call (see LLM_CALLS in variants.py);
the sequence is chosen from the CSV header's ``variant``.

The runner makes one warm-up call first. Calls are matched to questions in
that order, starting at the run's ``started`` time from the CSV header.
"t_other" is the rest of the question's latency: embedding, Chroma searches,
the retry-with-fewer-filters loop and Python overhead.

Only valid if nothing else used Ollama during the run. Calls longer than a
minute are logged with whole-second precision ("1m6s").
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.run import read_results  # noqa: E402
from benchmark.variants import LLM_CALLS  # noqa: E402

DEFAULT_LOG = Path(os.path.expandvars(r"%LOCALAPPDATA%\Ollama\server.log")) if sys.platform == "win32" \
    else Path.home() / ".ollama" / "logs" / "server.log"

ROUTES = ("FAQ", "Product")
ALL_STEPS = ["t_route", "t_task", "t_filter", "t_answer"]

_GIN = re.compile(r'\[GIN\] (\S+) - (\S+) \| \d+ \|\s*(\S+) \|.*POST\s+"/api/generate"')
_UNITS = {"ms": 1e-3, "µs": 1e-6, "us": 1e-6, "ns": 1e-9, "m": 60.0, "h": 3600.0, "s": 1.0}


def parse_duration(text: str) -> float:
    """Parse a Go duration string such as '1m6s', '8.02s' or '350ms' into seconds."""
    return sum(float(v) * _UNITS[u] for v, u in re.findall(r"([\d.]+)(ms|µs|us|ns|h|m|s)", text))


def read_calls(log_path: Path, since: dt.datetime) -> list[float]:
    calls = []
    for match in _GIN.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        end = dt.datetime.strptime(f"{match[1]} {match[2]}", "%Y/%m/%d %H:%M:%S")
        if end >= since:
            calls.append(parse_duration(match[3]))
    return calls


def header_value(csv_path: Path, key: str, default: str | None = None) -> str | None:
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith(f"# {key}:"):
                return line.split(":", 1)[1].strip()
    return default


def run_started(csv_path: Path) -> dt.datetime:
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# started:"):
                return dt.datetime.fromisoformat(line.split(":", 1)[1].strip())
    raise SystemExit(f"No '# started:' header in {csv_path}")


def split_steps(csv_path: Path, log_path: Path) -> list[dict]:
    df = read_results(csv_path)
    calls = read_calls(log_path, run_started(csv_path))
    # The router variants make no routing LLM call, so the sequence depends on the variant.
    variant = header_value(csv_path, "variant", "baseline")
    sequence = LLM_CALLS[variant]
    i = 1  # skip the warm-up call
    rows = []
    for _, r in df.iterrows():
        steps = sequence.get(r["route"], sequence["FAQ"])
        seg = calls[i:i + len(steps)]
        if len(seg) < len(steps):
            raise SystemExit(f"Log has too few calls for {r['question_id']} (repeat {r['repeat']}).")
        i += len(steps)
        row = {"question_id": r["question_id"], "repeat": r["repeat"], "route": r["route"],
               "latency_s": float(r["latency_s"])}
        row.update({s: "" for s in ALL_STEPS})
        row.update({s: round(t, 1) for s, t in zip(steps, seg)})
        row["t_other"] = round(row["latency_s"] - sum(seg), 1)
        row["latency_excluded"] = str(r.get("latency_excluded", "")).strip()
        rows.append(row)
    return rows


def _usable(row: dict, step: str) -> bool:
    # LLM step durations come from Ollama and stay valid; a latency_excluded row
    # (e.g. the laptop slept mid-question) is left out of the total and "everything else".
    return row[step] != "" and not (step in ("t_other", "latency_s") and row["latency_excluded"])


def summary(rows: list[dict]) -> str:
    """Markdown table: median seconds per step and route, with share of total."""
    lines = ["| Step | FAQ route (median s) | Product route (median s) | Share of a product question |",
             "| --- | --- | --- | --- |"]
    by_route = {route: [r for r in rows if r["route"] == route] for route in ROUTES}
    prod_ok = [r["latency_s"] for r in by_route["Product"] if _usable(r, "latency_s")]
    prod_total = statistics.median(prod_ok) if prod_ok else None
    for step in ALL_STEPS + ["t_other", "latency_s"]:
        cells = []
        for route in ("FAQ", "Product"):
            vals = [r[step] for r in by_route[route] if _usable(r, step)]
            cells.append(f"{statistics.median(vals):.1f} ({min(vals):.0f}–{max(vals):.0f})" if vals else "—")
        prod_vals = [r[step] for r in by_route["Product"] if _usable(r, step)]
        share = f"{statistics.median(prod_vals) / prod_total:.0%}" if prod_vals and prod_total and step != "latency_s" else ""
        name = {"t_route": "Route (LLM)", "t_task": "Task type (LLM)", "t_filter": "Filter JSON (LLM)",
                "t_answer": "Final answer (LLM)", "t_other": "Everything else (embedding, Chroma, Python)",
                "latency_s": "**Total**"}[step]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {share} |")
    counts = ", ".join(f"{len(v)} {k}" for k, v in by_route.items())
    excluded = [f"{r['question_id']} repeat {r['repeat']} ({r['latency_excluded']})"
                for r in rows if r["latency_excluded"]]
    note = f" Left out of total and everything-else: {'; '.join(excluded)}." if excluded else ""
    lines.append(f"\nQuestions: {counts}. Cells show median (min–max).{note}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out", type=Path, help="Write per-question step times to this CSV")
    args = parser.parse_args(argv)

    rows = split_steps(args.csv, args.log)
    if args.out:
        import csv

        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
