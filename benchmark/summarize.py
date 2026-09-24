"""Summarise benchmark result CSVs into the results table.

    python benchmark/summarize.py benchmark/results/*.csv

Per model and variant:
- latency: median over all runs, with min and max (repeats are noisy on a laptop);
  rows with a ``latency_excluded`` reason are left out of the latency figures only;
- correct: questions answered correctly in the majority of repeats, out of the
  number of questions (manual_override, when filled in, replaces the script's score);
- routed correctly: runs whose route matches the one the question needs (FAQ for fact
  questions, Product for product questions; the two insufficient-context questions excluded);
- filter JSON: runs where the filter JSON parsed / runs where a filter was generated;
- fallback: runs that ended in unfiltered search / runs on the product route.

Prints a Markdown table, then a per-question breakdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.run import read_results  # noqa: E402


def _final_correct(row) -> bool:
    override = str(row.get("manual_override", "")).strip().lower()
    if override in {"y", "n"}:
        return override == "y"
    return str(row["correct"]).strip().lower() == "y"


QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"


def expected_routes() -> dict[str, str]:
    """Route each question needs, derived from how it is scored (the question set stays frozen):
    fact questions are answered from the FAQ, product questions need the product branch.
    "Insufficient context" questions have no single right route and are left out."""
    questions = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    routes = {}
    for q in questions:
        if "facts" in q["expected"]:
            routes[q["id"]] = "FAQ"
        elif "products" in q["expected"]:
            routes[q["id"]] = "Product"
    return routes


def summarise(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([read_results(p) for p in paths], ignore_index=True)
    routes = expected_routes()
    df["route_expected"] = df["question_id"].map(routes)
    df["ok"] = df.apply(_final_correct, axis=1)
    df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce")
    # Rows with a latency_excluded reason still count for accuracy, not for timing.
    df.loc[df["latency_excluded"].astype(str).str.strip() != "", "latency_s"] = float("nan")

    per_question = (
        df.groupby(["model", "variant", "question_id", "category"])
        .agg(correct_runs=("ok", "sum"), runs=("ok", "size"))
        .reset_index()
    )
    per_question["majority"] = per_question["correct_runs"] * 2 > per_question["runs"]

    rows = []
    for (model, variant), g in df.groupby(["model", "variant"], sort=False):
        pq = per_question[(per_question["model"] == model) & (per_question["variant"] == variant)]
        filt = g[g["filter_json_parsed"].isin(["y", "n"])]
        prod = g[g["fallback"].isin(["y", "n"])]
        rows.append({
            "model": model,
            "variant": variant,
            "median latency (s)": round(g["latency_s"].median(), 1),
            "min–max (s)": f"{g['latency_s'].min():.0f}–{g['latency_s'].max():.0f}",
            "correct": f"{int(pq['majority'].sum())} / {len(pq)}",
            "routed correctly": f"{(g['route'] == g['route_expected']).sum()} / {g['route_expected'].notna().sum()}",
            "filter JSON parsed": f"{(filt['filter_json_parsed'] == 'y').sum()} / {len(filt)}",
            "fell back to unfiltered": f"{(prod['fallback'] == 'y').sum()} / {len(prod)}",
            "errors": int((g["error"].astype(str).str.len() > 0).sum()),
        })
    return pd.DataFrame(rows), per_question


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    # Step-time exports (*_steps.csv) sit next to the run CSVs; they aren't runs.
    paths = [Path(a) for a in argv if not a.endswith("_steps.csv")]
    table, per_question = summarise(paths)
    print(to_markdown(table))
    print()
    per_question["run"] = per_question["model"] + " / " + per_question["variant"]
    wide = per_question.pivot_table(
        index=["question_id", "category"], columns="run",
        values="correct_runs", aggfunc="first",
    )
    print(to_markdown(wide.reset_index()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
