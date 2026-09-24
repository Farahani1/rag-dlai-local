# Week 1 and Week 2 tests

This directory contains the regression baseline for the locally adapted Week 1 and Week 2 assignments. The suite deliberately uses several levels of testing: fast isolated checks for everyday development, selected notebook execution with stubs, machine-readiness checks, and optional execution with real local models and data.

It does **not** currently execute every notebook cell from top to bottom. The real-notebook tier executes a dependency-safe selection of original cells. Long free-form LLM demonstrations, exploratory output, and interactive widgets are excluded explicitly in `notebook_execution_plans.py`.

## Test levels

| Level | Command | Real resources | Mocked or excluded |
|---|---|---|---|
| Fast default | `.\.env\Scripts\python.exe -m pytest` | Canonical source files and notebook JSON | Model loading, Ollama, embeddings, widgets, and expensive imports are stubbed where necessary |
| Stubbed notebook execution | `.\.env\Scripts\python.exe -m pytest -m notebook` | Original selected function-cell source executed by a real Jupyter kernel | Dataset, embeddings, retrieval model, and external calls use tiny in-memory stubs |
| Readiness only | `.\.env\Scripts\python.exe -m pytest tests\w1_w2\test_local_integration_readiness.py -m local_integration` | Configuration, filesystem paths, model-cache metadata, Ollama endpoint, and installed model list | No model inference or notebook execution |
| Real local integration | `.\.env\Scripts\python.exe -m pytest -m local_integration` | Canonical utilities, CSV data, saved embeddings, local embedding model, BM25, Ollama, and selected original notebook cells | Widget cells, manual interactions, redundant demonstrations, and long notebook LLM calls remain excluded |

Run commands from the repository root. The real local-integration tier requires Ollama to be running with the model configured in `config.yaml`; it should not download models or modify files under `data/`.

## Files

### Fast tests

- `test_smoke.py` compiles `w1/utils.py` and `w2/utils.py`, imports them with lightweight dependency stubs, and parses all notebook code cells.
- `test_assignment_behaviors.py` tests utility behavior and key exercise functions with tiny deterministic substitutes for model, retrieval, and external-service behavior.
- `test_notebooks.py` checks notebook cell classification, required exercise functions, and one explicit automation decision for every code cell.

### Optional stubbed notebook tests

- `test_notebook_execution.py` starts a Jupyter kernel and executes selected original function cells against tiny in-memory data and retrieval stubs. It validates notebook execution mechanics without requiring local models or Ollama.

### Optional real local-integration tests

- `test_local_integration_readiness.py` checks packages, configuration, data paths, Hugging Face cache structure, Ollama reachability, and configured-model availability. Missing machine prerequisites produce descriptive skips.
- `test_real_utils.py` imports the canonical Week 1 and Week 2 utilities, performs real semantic retrieval using local embeddings, loads the real embedding model, and sends bounded generation requests to Ollama.
- `test_real_notebook_execution.py` executes the original cells marked `execute-real` with real local data, BM25, embeddings, and the embedding model. It verifies retrieval, reciprocal-rank fusion, formatting, and final prompt generation without invoking the notebooks' long LLM demonstrations or widgets.

### Support files

- `support.py` contains notebook parsing, function extraction, import helpers, and deterministic stubs shared by the fast tests.
- `notebook_execution_plans.py` records the execution decision and reason for every Week 1 and Week 2 code cell.
- `__init__.py` makes this directory a package so tests can use stable relative imports.

## What “real notebook execution” means here

The real tier uses unchanged source from selected cells in the canonical notebooks and runs it in fresh Jupyter kernels. It covers the important non-interactive assignment path, but it is not a full-notebook test. A future `full_notebook` tier would be needed to claim complete top-to-bottom execution, and it would still need an explicit policy for widgets and manual-interaction cells.

Keep this README synchronized whenever test files, markers, mock boundaries, or execution scope change.
