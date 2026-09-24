"""Explicit automation decisions for every Week 1 and Week 2 code cell."""

EXECUTE_REAL = "execute-real"
SKIP_DEMO = "skip-demo"
SKIP_LLM = "skip-llm"
SKIP_WIDGET = "skip-widget"

VALID_DECISIONS = {EXECUTE_REAL, SKIP_DEMO, SKIP_LLM, SKIP_WIDGET}


NOTEBOOK_EXECUTION_PLANS = {
    "w1": {
        5: (EXECUTE_REAL, "Import the canonical local utility module."),
        7: (EXECUTE_REAL, "Load the configured real news dataset."),
        9: (SKIP_DEMO, "Pretty-printing sample rows adds no behavior coverage."),
        13: (EXECUTE_REAL, "Define the notebook's dataset lookup helper."),
        14: (SKIP_DEMO, "Example lookup is covered by selected-cell assertions."),
        16: (SKIP_DEMO, "Standalone retrieval demo is covered by prompt execution."),
        17: (SKIP_DEMO, "Pretty-printing retrieval output is demonstration-only."),
        19: (EXECUTE_REAL, "Define the graded retrieval composition helper."),
        20: (SKIP_DEMO, "Example retrieval is covered by selected-cell assertions."),
        23: (EXECUTE_REAL, "Define the graded document formatter."),
        24: (SKIP_DEMO, "Example slice only supports a print demonstration."),
        26: (SKIP_DEMO, "Formatted-output print is demonstration-only."),
        28: (EXECUTE_REAL, "Define final RAG prompt generation."),
        29: (SKIP_DEMO, "Prompt print is covered by selected-cell assertions."),
        31: (EXECUTE_REAL, "Define the notebook LLM wrapper without calling it."),
        32: (SKIP_DEMO, "Hard-coded demo queries are manual exploration."),
        33: (SKIP_LLM, "Long free-form LLM demo is covered by bounded utility tests."),
        35: (SKIP_WIDGET, "Interactive widget requires manual input and display."),
    },
    "w2": {
        3: (EXECUTE_REAL, "Import canonical utilities and retrieval packages."),
        5: (EXECUTE_REAL, "Load the configured real news dataset."),
        7: (SKIP_DEMO, "Pretty-printing a sample row adds no behavior coverage."),
        10: (EXECUTE_REAL, "Define the notebook's dataset lookup helper."),
        13: (SKIP_DEMO, "Verbose BM25 walkthrough is replaced by compact setup."),
        15: (EXECUTE_REAL, "Build the real BM25 corpus and index."),
        16: (EXECUTE_REAL, "Define the graded BM25 retrieval helper."),
        17: (SKIP_DEMO, "Standalone BM25 demo is covered by assertions."),
        20: (EXECUTE_REAL, "Load the configured precomputed embeddings."),
        22: (EXECUTE_REAL, "Load the configured local embedding model."),
        23: (SKIP_DEMO, "Embedding preview is exploratory output."),
        25: (SKIP_DEMO, "Similarity example setup is not assignment-critical."),
        26: (SKIP_DEMO, "Shape demonstration is exploratory output."),
        27: (SKIP_DEMO, "Similarity print demonstration is exploratory output."),
        30: (SKIP_DEMO, "Manual query embedding is covered by semantic retrieval."),
        31: (SKIP_DEMO, "Intermediate similarity calculation is covered downstream."),
        32: (SKIP_DEMO, "Sorting walkthrough is covered by graded retrieval."),
        33: (SKIP_DEMO, "Retrieved-row display is covered by assertions."),
        35: (EXECUTE_REAL, "Define the graded semantic retrieval helper."),
        36: (SKIP_DEMO, "Standalone semantic demo is covered by assertions."),
        41: (EXECUTE_REAL, "Define the graded reciprocal-rank fusion helper."),
        42: (SKIP_DEMO, "Verbose fusion demo is covered by assertions."),
        45: (EXECUTE_REAL, "Define final RAG prompt generation."),
        46: (EXECUTE_REAL, "Define the notebook LLM wrapper without calling it."),
        47: (SKIP_LLM, "Long free-form LLM demo is covered by bounded utility tests."),
        49: (SKIP_WIDGET, "Interactive widget requires manual input and display."),
    },
}
