# rag-dlai-local

Hi! If you're here, you probably want to learn RAG (Retrieval-Augmented Generation), specially using the excellent material from deeplearning.ai, but you might, like me, have had trouble accessing the original hosted materials (Weaviate Cloud, Together.ai, Arize Phoenix, ...). This project rewrites the backend code of each week's assignment so the notebooks can be run **completely offline on a low-resource laptop**, while preserving the content, structure, and interface of the original notebooks as closely as possible.

## Why you might be interested in this project

- **Runs 100% locally on low‑resource hardware** – all LLM calls and embedding generation happen on your machine using [Ollama](https://ollama.com). I tested it on a very basic laptop (10th‑gen i3, 12 GB RAM, M.2 SSD, Windows 11)

- **No payment or registration required** – all the resources are free

- **Respects the original course** – the core learning flow is preserved, so it would be in the deeplearning.ai way.


#### **My Setup:**
And the models that ran smoothly in the following system:

| SPECS | |
| ----- | ----- |
| CPU | 10th-gen Intel Core i3 |
| RAM | 12 GB |
| Storage | M.2 SSD |
| OS | Windows 11 |
| Generative model (Ollama) | `gemma3:1b` (W1–W4), `qwen2.5:1.5b` (W5) |
| Embedding model | `all-MiniLM-L6-v2` |

*Other recommended models (computation-friendly for home PCs):*

**Generative (Ollama)**
- `phi3:mini` – Microsoft Phi-3-mini
- `qwen2.5:1.5b-instruct` – Qwen2.5-1.5B-Instruct
- `gemma3:4b` (quantised, e.g. `google/gemma-3-4b-it-qat-q4_0`)

**Embedding**
- `sentence-transformers/all-MiniLM-L6-v2` (what I used)
- `multi-qa-MiniLM-L6-cos-v1` – Multi-QA MiniLM
- `ms-marco-MiniLM-L-6-v2` – a reranker, useful from Week 3 onward

You just need to put the model name in `config.yaml`.

## What's in this repo, week by week

Each week lives in its own `wN/` directory with its own `utils.py` (and, from W3 onward, its own `chroma_store.py`, `embedding.py`, etc.) — every week is adapted independently, so nothing you change in one week's directory affects another week.

| Week | Notebook | Original course topic | What had to be replaced locally |
|------|----------|------------------------|----------------------------------|
| **W1** | `w1/C1M1_Assignment.ipynb` | Introduction to RAG Systems | Together.ai LLM calls → Ollama; embedding + cosine-similarity retrieval stays local |
| **W2** | `w2/C1M2_Assignment.ipynb` | Implementing Retriever Functions in a RAG System | Together.ai LLM calls → Ollama; BM25 / embedding / hybrid retrieval implemented locally |
| **W3** | `w3/C1M3_Assignment.ipynb` | Building RAG Systems with a Vector Database | Weaviate Cloud → local Chroma (`w3/chroma_store.py`); metadata filtering, semantic/BM25/hybrid search, and reranking reimplemented against Chroma; Together.ai → Ollama |
| **W4** | `w4/C1M4_Assignment.ipynb` | Developing a RAG-based Chatbot | Weaviate → Chroma product catalog; Together.ai chat completions → Ollama; a small clothing-product dataset is embedded and stored locally for the chatbot to query |
| **W5** | `w5/C1M5_Assignment.ipynb` | Improving a RAG System | Builds on W4's chatbot with a token-usage/performance layer (`simplified` mode for every RAG step) and Arize-Phoenix-style tracing, replaced with local no-op tracing stand-ins; adds a local FAQ Chroma collection alongside the product one |

The course's original, unmodified notebooks are not redistributed here. If you are enrolled in the course, you can download them from the course platform and diff them against the adapted `C1M*_Assignment.ipynb` to see exactly what changed.

### The `.py` mirrors and stateful runners (W3–W5)

From Week 3 onward, every notebook's code cells are additionally mirrored into a plain Python module, e.g. `w5/C1M5_Assignment.py`, as one `cell_NN(adapted: bool = False)` function per notebook cell. Passing `adapted=False` runs the original reference implementation (Weaviate/Together.ai-shaped code, for comparison); `adapted=True` runs the local Chroma/Ollama version actually used by the notebook. This mirror is what the automated test suite exercises — it lets the graded exercises and the overall notebook flow be unit-tested without needing a running Jupyter kernel, real Ollama server, or populated vector DB for every single test.

A companion `w5/C1M5_Assignment_stateful.py` (and the W3/W4 equivalents) simulates running the whole notebook top-to-bottom in one shared namespace (`run_until(cell_number, ...)`), which is what the project's regression tests use to catch a change in one cell breaking a later cell — the same class of bug a human would only notice by re-running the whole notebook.

**The notebook files themselves (`C1M*_Assignment.ipynb`) are what you actually open and run** — the `.py` mirrors exist for fast, automated testing, not as a substitute for running the notebook.

## Step-by-step guide

### 1. Get the code

```bash
git clone https://github.com/Farahani1/rag-dlai-local.git
cd rag-dlai-local
```

### 2. Install Python dependencies

```bash
python -m venv .env
.env\Scripts\activate   # On Linux/macOS: source .env/bin/activate
pip install -r requirements.txt
```

### 3. Install Ollama and pull a model

Download Ollama from https://ollama.com/download and install it.

Pull the small model used above (or any of the alternatives listed):
```bash
ollama pull gemma3:1b
```
For W5 specifically, `qwen2.5:1.5b` is the model this adaptation was verified against:
```bash
ollama pull qwen2.5:1.5b
```
`setting.py` checks that Ollama is reachable at import time and raises a clear error if it isn't — start Ollama (`ollama serve`, or just run any `ollama` command once) before opening a notebook.

### 4. Copy and edit the config file

```bash
cp config_example.yaml config.yaml
```
Then open `config.yaml` and set your model names:
```yaml
models:
  ollama:
    modelName: "gemma3:1b"
    url: http://localhost:11434/api/generate
  embeddingModel: "sentence-transformers/all-MiniLM-L6-v2"
```
All file paths used by any week (embeddings, CSVs, Chroma DB folders, ...) are declared under `data:` in `config.yaml` and resolved relative to the project root by `setting.py` — you shouldn't need to hardcode a path anywhere else.

### 5. Populate the local vector databases (W3–W5 only)

W1 and W2 work directly off the CSV/joblib files already under `data/`. From W3 onward the notebooks read from a local Chroma collection, which needs to be built once before first use:

```bash
# W4 — product catalog used by the chatbot
python w4/populate_products.py

# W5 — its own product collection + a small FAQ collection
python w5/populate_products.py
python w5/populate_faq.py
```
These are one-time, CPU-bound scripts (encoding ~44k product rows takes a few minutes on modest hardware) — they are never invoked automatically by tests or by opening a notebook. Re-run them if you ever delete `data/chroma_db_products/`.

W3's Chroma collection is built from `data/news_data_dedup.csv` + the already-provided `data/embeddings.joblib`, and is populated by the notebook's own setup cell — no separate script needed.

### 6. Launch Jupyter and run a notebook

```bash
jupyter notebook
```
Open any week's `C1M*_Assignment.ipynb` and run all cells top to bottom. Everything stays offline — no request leaves your machine.

## Running the test suite

Each week's adaptation is covered by an automated test suite under `tests/`, checked in alongside a `tests/README.md` explaining the import/path patterns used to keep tests isolated from each other (see [`tests/README.md`](tests/README.md) for the details — this matters because several weeks each ship their own same-named `utils.py`/`chroma_store.py`).

```bash
python -m pytest
```

By default (`pytest.ini`) this runs unit and contract tests only — it skips two optional, heavier tiers:

- `notebook` — executes selected real notebook cells
- `local_integration` — requires a running local Ollama server, local models, and populated data/vector stores

Run everything, including the slow/local tiers:
```bash
python -m pytest -m ""
```

`tests/w1_w2/` covers Weeks 1–2 (shared conventions, since neither week introduced a vector DB yet); `tests/w3/`, `tests/w4/`, `tests/w5/` each cover their own week's graded cells, adapted modules, and a full stateful regression run through the entire notebook with the LLM/vector-store calls mocked.

## Project conventions (for anyone extending this further)

These are the ground rules this adaptation follows:

- Every path used by any assignment module is declared in `config.yaml` and resolved through `setting.py` — no hardcoded paths inside `wN/` modules.
- No code path makes an online API call, or downloads data/models at runtime — everything needed must already be local.
- All data (raw CSVs, joblib caches, Chroma DB folders) lives under `data/` at the project root, shared across weeks where it makes sense (e.g. W4's and W5's product collections live in the same Chroma path, under different collection names) and otherwise namespaced per week.
- The course's original notebooks are not redistributed, but each `.py` mirror keeps the original cloud-based code path (`adapted=False`) next to the local one (`adapted=True`), so every change stays inspectable.
- Educational behavior (the exercises, their structure, and their intended learning outcome) is preserved, apart from a few small, deliberate deviations needed to run everything locally.

## Known limitations / skipped checks

- Small local models (1–1.5B parameters) are noticeably less reliable at structured-output tasks (e.g. W5's JSON filter generation) than the large cloud models the course was originally designed around. Where this happens, the notebook's own fallback logic (e.g. falling back to unfiltered semantic search when JSON parsing fails) is exercised for real rather than being purely theoretical — this is expected behavior with a small local model, not a bug.
- `w3/retrieval.py`'s / `w5/retrieval.py`'s reranking helpers are carried over from the original course code but are not exercised by every notebook path.
- Vector DB population scripts (`w4/populate_products.py`, `w5/populate_products.py`, `w5/populate_faq.py`) must be run manually once per environment (Step 5 above); they are intentionally not run automatically by anything (tests, notebook startup) to keep test runs fast and side-effect-free.

## Final words

### A note on copyright and fairness

This is an unofficial, independent adaptation. It is not affiliated with or endorsed by DeepLearning.AI. The course's lessons, videos and original notebooks are not redistributed here; to take the course itself, enrol through DeepLearning.AI.

What this repo does contain:

- **Adapted assignment notebooks and code.** These are derived from the course's assignments. The parts that called cloud services (Together.ai, Weaviate Cloud, Arize Phoenix) were rewritten to use local tools (Ollama, Chroma, local no-op tracing), and the notebooks' narrative text was rewritten. The `.py` mirrors keep the original cloud-based code path (`adapted=False`) next to the local one for comparison.
- **Data files under `data/`.** The CSV datasets and precomputed embedding caches come from the course materials or are derived from them, and they remain the property of their original owners. They are included only so the exercises can run offline.

The MIT license in `LICENSE` covers the code written for this adaptation (the local backends, the test suite and the tooling). It does not re-license any course material or third-party data. If you hold rights to anything included here and want it removed, please open an issue and it will be taken down.

I built this so it's accessible to everyone, regardless of ability to pay for cloud API credits or hosted services.

### Thanks

deeplearning.ai, for creating such a thoughtful RAG course :)

The Ollama team, for making local LLMs so easy.

The Chroma team, for a vector DB that just works locally with no server to stand up.

All the open-source model creators who made small, capable models that can run on my i3 computer :))
