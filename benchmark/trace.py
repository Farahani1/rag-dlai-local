"""Trace one benchmark run's filter step end to end, without calling the model again.

    python benchmark/trace.py benchmark/results/qwen2.5-1.5b_2026-09-24.csv f2 --repeat 1
    python benchmark/trace.py <csv> f2 --out benchmark/results/trace_f2_r1.md

The runner stores the filter step's raw model output and the final answer,
but not the prompts. Prompts are deterministic, so this script rebuilds them:
it loads the W5 pipeline exactly as ``run.py`` does, replaces the LLM call
with a stub that records the prompt and returns the recorded raw output, and
then runs the notebook's own ``parse_json_output`` and
``get_filter_by_metadata`` on it to reproduce the parse error and the
fallback decision.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.run import load_pipeline, read_results  # noqa: E402

import yaml  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"


def trace(csv_path: Path, question_id: str, repeat: int) -> str:
    df = read_results(csv_path)
    row = df[(df["question_id"] == question_id) & (df["repeat"].astype(int) == repeat)]
    if row.empty:
        raise SystemExit(f"No row for {question_id} repeat {repeat} in {csv_path}")
    row = row.iloc[0]
    questions = {q["id"]: q for q in yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]}
    question = questions[question_id]["question"]
    raw_output = row["filter_raw_output"]
    if not raw_output:
        raise SystemExit(f"{question_id} repeat {repeat} has no filter step (route: {row['route']}).")

    with contextlib.redirect_stdout(io.StringIO()):  # setup cells print their demo output
        ns, _ = load_pipeline(row["model"])

    captured: list[dict] = []

    def stub_generate(prompt, **kwargs):
        captured.append({"prompt": prompt, **kwargs})
        return {
            "choices": [{"message": {"content": raw_output}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": row["model"],
        }

    ns["generate_with_single_input"] = stub_generate
    content, _ = ns["generate_metadata_from_query"](question)
    assert content == raw_output and len(captured) == 1
    prompt = captured[0]["prompt"]

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        parsed = ns["parse_json_output"](raw_output)
    filters = ns["get_filter_by_metadata"](parsed)

    colour_list = re.search(r"'baseColour': \{[^}]*\}", prompt)
    nan_in_prompt = "nan" in (colour_list.group(0) if colour_list else "")
    fenced = raw_output.lstrip().startswith("```")

    lines = [
        f"# Trace: {question_id}, repeat {repeat} ({row['model']})",
        "",
        f"Source: `{csv_path.as_posix()}`. Rebuilt with `benchmark/trace.py`; no model call was made.",
        "",
        "## 1. Question",
        "",
        f"> {question}",
        "",
        f"Routed to: **{row['route']}**.",
        "",
        "## 2. Filter prompt sent to the model",
        "",
        f"{len(prompt):,} characters. The allowed values are pasted in as a Python dict of sets "
        f"(single quotes, `{{...}}` sets), not JSON."
        + (" The colour list contains `nan` (catalogue products without a colour)." if nan_in_prompt else ""),
        "",
        "````text",
        prompt.strip(),
        "````",
        "",
        "## 3. Raw model output",
        "",
        "````text",
        raw_output,
        "````",
        "",
        "## 4. Parse result",
        "",
        f"- Output starts with a Markdown code fence: **{'yes' if fenced else 'no'}**.",
        f"- `parse_json_output` printed: `{printed.getvalue().strip() or '(nothing)'}`",
        f"- Parsed JSON: `{parsed}`",
        f"- Filters from `get_filter_by_metadata`: `{filters}`",
        "",
        "## 5. Retrieval",
        "",
        "No filters, so `get_relevant_products_from_query` fell back to unfiltered semantic search "
        f"(recorded fallback: `{row['fallback']}`)." if not filters else f"Filtered search with `{filters}`.",
        "",
        "## 6. Final answer",
        "",
        "````text",
        row["answer"],
        "````",
        "",
        f"Scored: `{row['correct']}`"
        + (f", manual override: `{row['manual_override']}`" if row["manual_override"] else "")
        + ".",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("question_id")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    text = trace(args.csv, args.question_id, args.repeat)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
