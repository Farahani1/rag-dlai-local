# Design decisions

Three decisions that shaped this adaptation, each as constraint → decision → consequence.

## 1. Chroma as the vector database

**Constraint.** The course uses Weaviate Cloud. The replacement had to run in-process on Windows, with no server to start, no account and no Docker.

**Decision.** Chroma, used through a thin `ChromaStore` wrapper per week (`w3/`, `w4/`, `w5/chroma_store.py`). FAISS was ruled out because it has no metadata filtering, which W3 and W5 depend on. Qdrant and Weaviate can run locally, but only as a separate server, usually in Docker.

**Consequence.** Setup is `pip install` and nothing else, and the data lives in a folder under `data/`. But Chroma has no built-in BM25, hybrid search or reranking, so these were reimplemented in each week's `retrieval.py`. Weaviate's filter objects also had to be translated into Chroma `where` clauses.

## 2. `.py` mirrors and stateful runners

**Constraint.** Notebooks can't be unit-tested directly, and running one end to end needs Ollama, local models and populated vector stores. That's too slow and too fragile to do after every change.

**Decision.** Each notebook (W3–W5) is mirrored as one `cell_NN(adapted)` function per code cell. A stateful runner replays those cell bodies in one shared namespace, the way a kernel does. See [testing.md](testing.md).

**Consequence.** A regression test runs a whole notebook in seconds, with the LLM and vector store mocked, and catches a change in one cell breaking a later one. The cost is keeping the mirror and the notebook in sync by hand: a notebook edit that isn't copied to the mirror goes untested.

## 3. Weeks stay separate

**Constraint.** Each week has to match its own course notebook, which imports its own `utils.py` and helpers.

**Decision.** Every week keeps its own copies (`utils.py`, `chroma_store.py`, `embedding.py`, `retrieval.py`), even where they overlap. The only shared code is `setting.py`, which reads `config.yaml`.

**Consequence.** A learner can open any week on its own, and changing one week can't break another. The price is duplication: a bug fix sometimes has to be applied in three places. This isn't the right architecture for an application. It is deliberate here, because the course is organised week by week.
