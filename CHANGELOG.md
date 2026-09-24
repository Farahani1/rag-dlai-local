# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-09-25

Focus: a fresh clone installs and runs on Windows, Linux and macOS; a benchmark measures what a small local model does on the course's Week 5 pipeline; tests run in CI.

### Added

- `requirements-dev.txt` with the test-suite dependencies (pytest, plus nbclient, nbformat and ipykernel for the optional notebook tiers).
- `requirements-lock-windows.txt`: an exact snapshot (199 packages) of the Windows 11 / Python 3.14 environment the project was tested in.
- `.gitattributes`: the repository stores LF line endings and each checkout uses the platform default. Joblib and SQLite files are marked binary.
- `docs/decisions.md`: three design decisions (Chroma, `.py` mirrors with stateful runners, separate weeks), each as constraint, decision and consequence.
- Benchmark for the W5 pipeline: a fixed 12-question set (`benchmark/questions.yaml`), a runner that sends each question through the notebook's own functions at temperature 0 and records latency, routing, filter parsing and fallback (`benchmark/run.py`), rule-based scoring with unit tests (`benchmark/scoring.py`), and a summary table (`benchmark/summarize.py`).
- `benchmark/step_times.py` splits each question's latency into pipeline steps (route, task type, filter JSON, answer, everything else) using the durations in Ollama's server log; `docs/results.md` reports the first results.
- Before/after experiments on the W5 pipeline (`benchmark/variants.py`, `--variant parse|router|both`): an embedding router instead of the routing LLM call raises the benchmark from 4/12 to 8/12 and median latency falls from 161 s to 36 s; a parser fix alone doesn't help. Router threshold chosen on a separate dev set (`benchmark/router_dev.yaml`, `benchmark/tune_router.py`). One failure traced end to end (`benchmark/trace.py`). Results in `docs/results.md`; the README's first screen shows the headline.
- GitHub Actions workflow running the default test tier on Ubuntu with Python 3.12 and CPU-only PyTorch.
- Stated Python requirement: **3.12 or newer**, in `README.md` and `requirements.txt`. The W1 notebook uses f-strings that nest the same quote type (PEP 701), which is a syntax error on Python 3.11.

### Changed

- `requirements.txt` rewritten as 13 direct dependencies with version ranges, in UTF-8. It replaces a 125-line UTF-16 `pip freeze` of the whole environment, which GitHub displayed as a binary file.
- `setting.py`: importing the configuration no longer requires a running Ollama server. `ensure_ollama_running()` is still available to call explicitly. Generation calls still stop with a clear "Failed to connect to Ollama" error when it is not running.
- `setting.py`: `resolve_path()` accepts both `\` and `/` as separators, so the same `config.yaml` works on every operating system.
- `config_example.yaml`: paths use forward slashes (`data/embeddings.joblib`).
- `README.md`: separate install blocks for Windows (PowerShell, `Activate.ps1`, with a note on the execution policy) and Linux/macOS; a step for installing the dev requirements before running tests; corrected description of the Ollama check.
- `data/faq.joblib` (Weeks 4–5) is now built from `data/faq.yaml`: 25 FAQ entries written for this project, replacing the FAQ from the course materials. Schema, categories and store name are unchanged.
- `w5/populate_faq.py` upserts entries, so re-running it after editing the FAQ replaces the stored entries.
- `data/chroma_db/chroma.sqlite3` is no longer tracked. The W3 notebook builds it on first run.

- W4 and W5 notebooks re-executed so their saved outputs show the new FAQ.
- `README.md` is now a short landing page (summary, CI badge, the service swap as a diagram, a five-line quick start). The details moved to `docs/setup.md` (with a troubleshooting table), `docs/testing.md` and `docs/data.md`.

### Removed

- `data/bbc_news.csv` (13.5 MB), which no code read. The README still credits the dataset the notebooks mention.

### Fixed

- A fresh install could not run Weeks 3–5: `bm25s` (retrieval) and `Markdown` (chat widget) were imported but missing from `requirements.txt`.
- Configs written with Windows-style paths (`.\data\...`) did not resolve on Linux or macOS.
- On Linux the default test tier ended with 3 failures and 17 skipped tests. It now matches the Windows result.
- 2 tests in `tests/w3/test_get_embedding.py` failed because they mocked `os` completely and then checked the real filesystem. They now keep `os` real.
- Running the default test tier rewrote `data/chroma_db/chroma.sqlite3`. The W3 stateful tests now use a temporary directory, and the default tier leaves `data/` unchanged.

### Verification

- Clean install from `requirements.txt` + `requirements-dev.txt` on Linux, Python 3.12: `python -m pytest` gave 308 passed, 2 failed (the two cache-dir tests, since fixed), 13 deselected.
- After the fixes and the data review, on Windows: 307 passed, 0 failed, 13 deselected (the two cache-dir tests now pass; the three `bbc_news.csv` contract tests were removed).
- `import setting` succeeds with Ollama stopped.
- Windows, fresh clone in a new venv (Python 3.14.7, Git Bash), 2026-09-25: 339 passed, 13 deselected, 0 failed.

## [1.0.0] - 2026-09-24

First public release.

### Added

- All five weeks (W1–W5) of the course's assignments adapted to run fully offline: Ollama replaces the hosted LLM, Chroma replaces Weaviate Cloud, and local no-op tracing replaces Arize Phoenix.
- `.py` mirrors of each notebook (one `cell_NN(adapted)` function per code cell) and stateful runners that replay a whole notebook in one namespace.
- Regression test suite with three tiers: default (unit, contract and stateful tests), `notebook`, and `local_integration`.
- `data/clothes_json.joblib` cache, so Weeks 4–5 run from a fresh clone.
- "Data sources" section crediting each dataset's author and license.

### Changed

- Notebook narrative rewritten to describe the local adaptation in the third person; graded unittest cells removed.
- Copyright note rewritten: this is an unofficial adaptation, and the MIT license covers the adaptation code only, not course material or third-party data.
- Default generative model named consistently as `gemma3:1b` across README and example config.

### Removed

- The course's original, unmodified notebooks and utility files from the public release.

[Unreleased]: https://github.com/Farahani1/rag-dlai-local/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Farahani1/rag-dlai-local/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Farahani1/rag-dlai-local/releases/tag/v1.0.0
