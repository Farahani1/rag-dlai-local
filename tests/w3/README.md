# Week 3 tests

This directory contains tests for the locally adapted Week 3 assignment, which replaces Weaviate with Chroma as the vector database. The tests were built incrementally, one adapted behavior at a time.

## Test levels

| Level | Command | What it verifies | Mocked or excluded |
|---|---|---|---|
| Fast default | `python -m pytest tests/w3/` | Data shape, Chroma adapter, retrieval functions, prompt generation, LLM call orchestration | Ephemeral Chroma, stub embed function, synthetic 3-doc fixtures, no network, no model inference |

Run commands from the repository root.

---

## Stage 1 — Contract tests for the data shape

**Goal**: Prove that the local data files can be transformed into the document shape the assignment expects, *before* writing any Chroma or database code.

**File**: `test_data_contract.py`

### What is tested

| Test | Purpose |
|------|---------|
| `test_news_csv_exists` | `data/news_data_dedup.csv` must be present |
| `test_embeddings_exist` | `data/embeddings.joblib` must be present |
| `test_row_count_matches_embedding_count` | The CSV must have exactly 870 rows, matching the 870 embeddings |
| `test_embeddings_shape_and_dtype` | Embeddings must be a 2D float32 array of shape (870, 384) |
| `test_each_row_transforms_to_required_fields` | Every row maps to a document with: `id`, `title`, `chunk`, `pubDate`, `link` |
| `test_required_fields_have_non_null_values` | No empty or None values in required fields |
| `test_single_row_demo` | Type-level verification on a representative row |

### Assumptions

- **Primary data source**: `data/news_data_dedup.csv` (870 rows), whose rows match `data/embeddings.joblib`.
- **Column mapping** (news_data_dedup → assignment document):

  | CSV column | Document field | Notes |
  |------------|---------------|-------|
  | `guid` | `id` | Unique identifier |
  | `title` | `title` | Used directly |
  | `description` | `chunk` | Main text to embed |
  | `url` | `link` | Source URL |
  | `published_at` | `pubDate` | Publication timestamp |

- **Embeddings**: Pre-computed and stored in `data/embeddings.joblib`. They are (870, 384) float32, matching `sentence-transformers/all-MiniLM-L6-v2`.
- **No database is needed**: These tests run entirely on flat files. No Weaviate, Chroma, or any other vector store is required.
- **No model inference**: Tests never load embedding models, LLMs, or rerankers.
- **No network calls**: Everything is local.

### What is not tested (deferred to later stages)

- Chroma collection creation
- Document insertion and querying
- Streaming search functions (`semantic_search_retrieve`, `bm25_retrieve`, `hybrid_retrieve`, etc.)
- Prompt generation and LLM calls
- Notebook state execution

### Files

| File | Purpose |
|------|---------|
| `test_data_contract.py` | Data shape contract tests |
| `__init__.py` | Package marker |

---

## Stage 2 — Chroma adapter behind tests

**Goal**: Build a thin Chroma adapter (``ChromaStore``) that wraps ChromaDB and returns plain Python dictionaries. Notebook functions should talk to this adapter, not to Chroma directly.

**Files**: `w3/chroma_store.py`, `test_chroma_store.py`

### What is tested

| Test | Purpose |
|------|---------|
| `test_create_and_list_collection` | Creating a collection and verifying it appears in the listing |
| `test_create_duplicate_collection` | Creating the same collection twice does not raise |
| `test_create_multiple_collections` | Multiple collections can coexist |
| `test_add_and_count` | Adding documents with embeddings increments the count |
| `test_add_empty_list_rejected` | Chroma rejects empty embeddings lists (expected limitation) |
| `test_add_to_nonexistent_collection_raises` | Adding to a missing collection raises ``ValueError`` |
| `test_query_returns_list_of_dicts` | Query results are a list of plain dicts |
| `test_query_returns_required_fields` | Each result dict contains ``id``, ``title``, ``chunk``, ``pubDate``, ``link`` |
| `test_query_respects_top_k` | Query returns exactly the requested number of results |
| `test_query_returns_only_dicts` | No Chroma-native objects leak through the adapter |
| `test_query_empty_collection_returns_empty_list` | Querying an empty collection yields ``[]`` |
| `test_query_nonexistent_collection_raises` | Querying a missing collection raises ``ValueError`` |
| `test_delete_collection` | Deleting a collection removes it from the listing |
| `test_delete_nonexistent_collection_raises` | Deleting a missing collection raises ``ValueError`` |

### Adapter design

```python
class ChromaStore:
    # Collection management
    def create_collection(self, name: str) -> None: ...
    def list_collections(self) -> list[str]: ...
    def delete_collection(self, name: str) -> None: ...
    def count(self, collection_name: str) -> int: ...

    # Document operations
    def add_documents(
        self,
        collection_name: str,
        documents: list[dict],    # each dict has id, title, chunk, pubDate, link
        embeddings: np.ndarray,   # shape (N, D)
    ) -> None: ...

    # Vector search
    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]: ...
```

### Design principles

- **Hide Chroma internals**: The adapter returns plain Python dicts, never Chroma ``Collection``, ``Document``, or ``QueryResult`` objects.
- **Translate exceptions**: Chroma's ``NotFoundError`` is translated to ``ValueError`` for a consistent public interface.
- **Ephemeral by default**: ``ChromaStore()`` uses an in-memory Chroma client; pass ``persist_directory`` for persistent storage.
- **Idempotent create**: ``create_collection`` uses ``get_or_create_collection`` so repeated calls are safe.

### Assumptions

- **chomadb 1.5.9** is already installed (added to ``requirements.txt``).
- **No real data**: Tests use tiny synthetic fixtures (3 documents with random 384-dim vectors), not the real CSV or embeddings.
- **No network**: Ephemeral Chroma requires no network calls, no local server, and no file I/O.
- **No model inference**: The adapter works with pre-computed embeddings only.
- **Tests run fast**: The full suite completes in under 5 seconds.

### Files

| File | Purpose |
|------|---------|
| ``w3/chroma_store.py`` | ChromaDB adapter — the production module |
| ``test_chroma_store.py`` | Adapter unit tests |
| ``tests/conftest.py`` | Root-level path configuration |

---

## Stage 3 — Retrieval functions (Weaviate → Chroma adaptation)

**Goal**: Replace the original 7 Weaviate-based retrieval/LLM functions with Chroma-backed local versions, tested behavior-by-behavior using TDD. Each function is adapted to accept a ``ChromaStore`` instance and a collection name instead of a Weaviate ``collection`` object.

**Production files**: `w3/retrieval.py`, `w3/embedding.py`, `w3/chroma_store.py`

**Test files**: `test_retrieval.py`, `test_chroma_store_extra.py`

### What is tested

| Test | Purpose |
|------|---------|
| ``TestFilterByMetadata::test_filter_by_exact_title`` | Metadata ``$in`` filter returns the matching document |
| ``TestFilterByMetadata::test_filter_returns_multiple`` | Multiple values in ``$in`` return multiple matches |
| ``TestFilterByMetadata::test_filter_no_match`` | No match returns an empty list |
| ``TestFilterByMetadata::test_filter_respects_limit`` | Result count does not exceed the requested ``limit`` |
| ``TestFilterByMetadata::test_filter_returns_dicts_with_required_fields`` | Each result has ``id``, ``title``, ``chunk``, ``pubDate``, ``link`` |
| ``TestSemanticSearchRetrieve::test_returns_list_of_dicts`` | Semantic search returns a list of plain dicts |
| ``TestSemanticSearchRetrieve::test_respects_top_k`` | Exactly ``top_k`` results returned |
| ``TestSemanticSearchRetrieve::test_required_fields_present`` | Each result has required fields |
| ``TestSemanticSearchRetrieve::test_empty_collection_returns_empty`` | Empty collection yields ``[]`` |
| ``TestBm25Retrieve::test_returns_list_of_dicts`` | BM25 search returns a list of plain dicts |
| ``TestBm25Retrieve::test_respects_top_k`` | Exactly ``top_k`` results returned |
| ``TestBm25Retrieve::test_keyword_relevance`` | Query ``"economy"`` returns the document about economic growth |
| ``TestBm25Retrieve::test_required_fields_present`` | Each result has required fields |
| ``TestBm25Retrieve::test_empty_collection`` | Empty collection returns ``[]`` (graceful ``bm25s`` skip) |
| ``TestHybridRetrieve::test_returns_list_of_dicts`` | Hybrid search returns a list of plain dicts |
| ``TestHybridRetrieve::test_respects_top_k`` | Exactly ``top_k`` results returned |
| ``TestHybridRetrieve::test_alpha_zero_is_bm25`` | ``alpha=0.0`` produces same result count as pure BM25 |
| ``TestHybridRetrieve::test_empty_collection`` | Empty collection returns ``[]`` |
| ``TestHybridRetrieve::test_required_fields_present`` | Each result has required fields |
| ``TestSemanticSearchWithReranking::test_returns_list_of_dicts`` | Reranked search returns a list of plain dicts |
| ``TestSemanticSearchWithReranking::test_respects_top_k`` | Exactly ``top_k`` results returned |
| ``TestSemanticSearchWithReranking::test_rerank_by_chunk`` | Query ``"Grammy awards ceremony"`` ranks the Grammy document first when reranking by ``chunk`` |
| ``TestSemanticSearchWithReranking::test_rerank_query_differs`` | Separate rerank query is accepted and used |
| ``TestSemanticSearchWithReranking::test_empty_collection`` | Empty collection returns ``[]`` |
| ``TestGenerateFinalPrompt::test_no_rag_returns_query`` | ``use_rag=False`` returns the raw query unchanged |
| ``TestGenerateFinalPrompt::test_use_rerank_without_property_raises`` | ``use_rerank=True`` with ``rerank_property=None`` raises ``ValueError`` |
| ``TestGenerateFinalPrompt::test_rag_formats_documents`` | RAG prompt contains ``Title:``, ``Chunk:``, ``Published at:``, ``URL:`` sections |
| ``TestGenerateFinalPrompt::test_prompt_contains_query`` | The original query text is embedded in the prompt |
| ``TestLlmCall::test_returns_prompt_when_no_backend`` | No ``llm_backend`` returns the prompt string itself |
| ``TestLlmCall::test_uses_llm_backend`` | A callable ``llm_backend`` is invoked with the prompt |
| ``TestLlmCall::test_no_rag_passes_through`` | ``use_rag=False`` + ``llm_backend`` returns the LLM response for the raw query |

**ChromaStore extra methods** (tested in ``test_chroma_store_extra.py``):

| Test | Purpose |
|------|---------|
| ``TestGetAllDocuments::test_get_all_returns_all_docs`` | ``get_all_documents`` returns every document in the collection |
| ``TestGetAllDocuments::test_get_all_empty_collection`` | Returns ``[]`` for an empty collection |
| ``TestGetAllDocuments::test_get_all_nonexistent_collection_raises`` | Missing collection raises ``ValueError`` |
| ``TestQueryWithFilter::test_filter_by_title_exact_match`` | ``$in`` filter on ``title`` returns the matching document |
| ``TestQueryWithFilter::test_filter_by_title_set_match`` | ``$in`` with multiple values returns all matches |
| ``TestQueryWithFilter::test_filter_no_match_returns_empty`` | No match returns ``[]`` |
| ``TestQueryWithFilter::test_filter_nonexistent_collection_raises`` | Missing collection raises ``ValueError`` |

### Adapted function signatures

Each function replaces a Weaviate ``collection`` parameter with a ``ChromaStore`` + ``collection_name`` pair:

```python
# Original (Weaviate)
def filter_by_metadata(property, values, collection, limit=5): ...

# Adapted (Chroma)
def filter_by_metadata(
    metadata_property: str, values: list[str],
    store: ChromaStore, collection_name: str, limit: int = 5,
) -> list[dict]: ...
```

| Original parameter | Replacement |
|---|---|
| ``collection`` (Weaviate collection object) | ``store`` (``ChromaStore``) + ``collection_name`` (``str``) |
| ``client`` (Weaviate client) | Removed — not needed by individual functions |
| *(implicit)* | ``embed_function`` — injected for semantic/rerank/hybrid functions |

### Design principles

- **Chroma metadata filter uses ``$in``** for exact matching (ChromaDB 1.5.9 does not support substring ``$contains`` on string fields).
- **BM25 uses ``bm25s`` library** (same library used by the Week 2 notebook, already installed in the environment). The BM25 index is built lazily and cached per collection name.
- **Local reranker** scores documents by counting query tokens present in the target field — no external reranking model required.
- **Embedding is injected as a callable** (``embed_function``) rather than imported directly, making functions testable with a stub that returns a zero vector.
- **Type hints use ``from __future__ import annotations``** so ``ChromaStore`` only needs a ``TYPE_CHECKING`` guard — no runtime import chain to manage.

### Assumptions

- **Ephemeral Chroma**: All tests use ``ChromaStore()`` (in-memory), never persistent storage.
- **No real data**: Tests use 3 synthetic documents with random 384-dim vectors, not the real 870-row CSV.
- **No model inference**: The test ``_stub_embed`` returns ``[0.0] * 384`` — no embedding model is loaded.
- **No network**: Ephemeral Chroma and ``bm25s`` operate entirely locally.
- **No ``sentence-transformers``**: The ``w3/embedding.py`` production module is importable but never loaded by tests.

### Files

| File | Purpose |
|------|---------|
| ``w3/retrieval.py`` | All 7 adapted retrieval/LLM functions |
| ``w3/embedding.py`` | ``embed_query(text)`` using ``all-MiniLM-L6-v2`` (production only, not used by tests) |
| ``w3/chroma_store.py`` | ChromaDB adapter — extended with ``get_all_documents`` and ``query_with_filter`` |
| ``test_retrieval.py`` | 31 unit tests covering all 7 functions |
| ``test_chroma_store_extra.py`` | 7 unit tests for the new ChromaStore methods |

---

## Stage 7 — Notebook execution scaffold (`adapted` flag)

**Goal**: Wire the new Chroma-backed retrieval functions into the notebook cell sequence so the existing stateful runner can execute the adapted paths end-to-end.  This stage introduces an ``adapted`` parameter on every cell function and updates the runner to pass it through — no wrapper module, no adapter class.

**Production files**: `w3/C1M3_Assignment.py`, `w3/C1M3_Assignment_stateful.py`, `w3/utils.py`

**Test file**: `test_stateful_cells.py`

### What is tested

| Test | Purpose |
|------|---------|
| ``test_setup_loads_and_populates_collection`` | Cells 04→08→10→13: import Chroma modules, create ``ChromaStore``, load 870 rows from ``news_data_dedup.csv`` + ``embeddings.joblib``, transform fields, populate the ``bbc_collection``.  Asserts ``store.count(collection) == 870``. |
| ``test_cell_14_prints_count`` | Cell 14 runs ``store.count()`` and prints without error |
| ``test_cell_16_inspects_first_document`` | Cell 16 fetches and prints the first document via ``get_all_documents`` + ``print_object_properties`` |

### How the `adapted` flag works

Each cell in ``C1M3_Assignment.py`` now accepts ``adapted: bool = False``:

```python
def cell_13(adapted: bool = False):
    if adapted:
        # Chroma path
        store.create_collection("bbc_collection")
        store.add_documents(...)
        collection = "bbc_collection"
    else:
        # Original Weaviate path (preserved)
        collection = client.collections.get("bbc_collection")
```

The stateful runner's ``run_until(cell_number, adapted=True)`` sets ``state.namespace["adapted"] = True``.  The ``_function_body_code`` compiler prepends ``adapted = namespace.get('adapted', False)`` to every cell body, so the ``adapted`` guard is visible in the executed namespace without the cells needing to receive it as a function parameter.

**Critical design choice**: ``adapted=True`` does **not** rewrite notebook cells.  It forks each cell at the point where an infrastructure decision is made (which imports, which client, which retrieval path).  The original Weaviate code is preserved verbatim in the ``else`` branch for reference and comparison.

### What cells skip in adapted mode

| Cells | Why skipped | Tested elsewhere |
|---|---|---|
| 05 | ``flask_app``, ``weaviate_server`` imports — not needed with Chroma | — |
| 22, 27, 32, 37, 44 | Original Coursera ``unittests.py`` — Weaviate-specific | — |
| 25, 29, 30, 34, 35, 39, 41, 42 | Trigger ``sentence-transformers`` model inference or ``bm25s`` index build on 870 docs — too slow for CI | ``test_retrieval.py`` (31 tests, 5 s, 3-doc fixtures) |
| 47, 48 | Require ``generate_final_prompt`` (defined in cell 46); the cell-46 guard uses semantic search which needs embedding | ``test_retrieval.py`` |
| 50, 52, 54 | ``llm_call`` + ``display_widget`` — need IPython kernel or Ollama | ``test_retrieval.py`` |

### Local Ollama migration (`w3/utils.py`)

The original ``utils.py`` imported ``openai`` and ``httpx`` at module level, making it impossible to import ``print_object_properties`` without installing those packages.  The adapted version:

- **Removes** the ``openai`` / ``Together`` API dependency entirely
- **Replaces** ``generate_with_single_input``, ``generate_with_multiple_input``, and ``generate_embedding`` with Ollama-based implementations following the pattern established in ``w1/utils.py`` and ``w2/utils.py``
- **Reads** model name, URL, and completion options from ``config.yaml`` via ``setting.py``
- **Preserves** ``print_object_properties`` and ``print_properties`` unchanged

This means ``from utils import print_object_properties`` works without any external API dependencies, and real LLM calls use the locally running Ollama model (whichever model `config.yaml` names, e.g. `gemma3:1b`).

### Design principles

- **Original code preserved**: Every cell keeps its Weaviate path as the ``else`` branch.  Reading the file side-by-side shows exactly what changed.
- **No wrapper module**: The ``adapted`` flag lives directly in the cell functions — no ``chroma_adapter.py`` indirection.
- **Minimal stateful tests**: Only 3 tests that verify the Chroma data pipeline works with real data.  Everything else is covered by the fast unit tests.
- **Ollama is the only LLM backend**: No Together AI, no OpenAI, no Coursera proxy — matches the local-first philosophy of the project.

### Assumptions

- **Real data files exist**: ``data/news_data_dedup.csv`` and ``data/embeddings.joblib`` must be present.  Tests skip gracefully if they are not.
- **Ollama not needed for tests**: The smoke tests stop before reaching ``llm_call``.  No model server needs to be running.
- **ChromaDB persists to ``data/chroma_db``**: The stateful runner defaults to ``ChromaStore(persist_directory="data/chroma_db")``.  Tests run against ephemeral in-memory stores.
- **No model inference in tests**: The 3 smoke tests avoid ``semantic_search_retrieve`` (which triggers embedding) and ``bm25_retrieve`` (which builds a BM25 index).

### Files

| File | Purpose |
|------|---------|
| ``w3/C1M3_Assignment.py`` | Cell extraction with ``adapted`` guard in every function |
| ``w3/C1M3_Assignment_stateful.py`` | Updated runner with importlib loading + ``adapted`` flag |
| ``w3/utils.py`` | Ollama-based LLM functions (replaces OpenAI/Together) |
| ``test_stateful_cells.py`` | 3 pipeline smoke tests (18 s total) |
