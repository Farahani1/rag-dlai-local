# Testing

How a notebook-based project is tested without running the notebooks by hand.

## The problem

Each week is a Jupyter notebook. Notebooks can't be imported, their cells share one global namespace, and running one end to end needs Ollama, local models and populated vector stores. A change in one cell can break a cell fifty cells later, and the only way to notice would be a full manual re-run.

## Mirrors and stateful runners (W3–W5)

- **Mirror** (`wN/C1MN_Assignment.py`): every code cell becomes one function, `cell_NN(adapted: bool = False)`. `adapted=True` is the local Chroma/Ollama code the notebook runs; `adapted=False` keeps the original cloud-based code next to it for comparison. Unit tests import these functions directly.
- **Stateful runner** (`wN/C1MN_Assignment_stateful.py`): replays the mirror's cell bodies in order in one shared namespace, like a notebook kernel. `run_until(cell_number, skip_cells=...)` runs the notebook up to a cell. Regression tests use it to run the whole notebook with Ollama, Chroma and embeddings mocked, which catches cross-cell breakage in seconds.

The notebooks are still what a learner opens and runs. The mirrors exist for testing and have to be kept in sync with the notebooks by hand; that is the cost of this design (see [decisions.md](decisions.md)).

W1 and W2 have no mirrors; their tests load the notebooks' own `utils.py` and selected notebook cells directly.

## Test tiers

| Tier | Command | Needs | What it covers |
| --- | --- | --- | --- |
| Default | `python -m pytest` | Nothing beyond `requirements*.txt` | Unit, data-contract and stateful regression tests; Ollama, models and vector stores are mocked. Leaves `data/` untouched. |
| `notebook` | `python -m pytest -m notebook` | A Jupyter kernel | Executes selected real notebook cells with small in-memory stubs. |
| `local_integration` | `python -m pytest -m local_integration` | Ollama running, local models, populated Chroma | Runs real models and data through selected notebook paths. |
| All | `python -m pytest -m ""` | Everything above | |

Install the test dependencies once with `pip install -r requirements-dev.txt`.

The default tier needs no downloaded models: it passes with an empty Hugging Face cache and `HF_HUB_OFFLINE=1`.

## Continuous integration

`.github/workflows/tests.yml` runs the default tier on every push and pull request (Ubuntu, Python 3.12, CPU-only PyTorch). The `notebook` and `local_integration` tiers are local-only by design: CI has no Ollama, no model downloads, no Chroma population and no notebook execution.

## Layout

| Folder | Covers |
| --- | --- |
| `tests/w1_w2/` | Weeks 1–2 (shared conventions; neither week has a vector DB yet) |
| `tests/w3/`, `tests/w4/`, `tests/w5/` | Each week's graded cells, adapted modules (Chroma store, retrieval, utils) and a full stateful regression run |
| `tests/benchmark/` | The benchmark's scoring rules and the question set's shape |

Several weeks ship their own `utils.py` and `chroma_store.py` with the same names. Tests load them by file path with `importlib` so they never collide; `tests/README.md` explains the pattern, and each week folder has its own README listing its tests.
