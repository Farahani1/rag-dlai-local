"""Contract tests for the local data shape.

These tests verify the agreement between local data files and the assignment's
expected document format *before* any Chroma or database code is written.

The assignment expects documents with: id, title, chunk, pubDate, link.
Primary data: news_data_dedup.csv  →  embeddings.joblib (870, 384)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def transform_news_row(row: dict) -> dict:
    """Map a news_data_dedup.csv row to the assignment's expected document shape."""
    return {
        "id": row["guid"],
        "title": row["title"],
        "chunk": row["description"],
        "pubDate": row["published_at"],
        "link": row["url"],
    }


# ---------------------------------------------------------------------------
# Existence checks
# ---------------------------------------------------------------------------


class TestDataFilesExist:
    def test_news_csv_exists(self):
        assert (DATA_DIR / "news_data_dedup.csv").exists(), (
            "Missing data/news_data_dedup.csv"
        )

    def test_embeddings_exist(self):
        assert (DATA_DIR / "embeddings.joblib").exists(), (
            "Missing data/embeddings.joblib"
        )


# ---------------------------------------------------------------------------
# Row count and embedding alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_row_count_matches_embedding_count(self):
        df = pd.read_csv(DATA_DIR / "news_data_dedup.csv")
        embeddings = joblib.load(DATA_DIR / "embeddings.joblib")
        assert len(df) == embeddings.shape[0], (
            f"news_data_dedup.csv has {len(df)} rows but embeddings have "
            f"{embeddings.shape[0]} rows"
        )

    def test_embeddings_shape_and_dtype(self):
        embeddings = joblib.load(DATA_DIR / "embeddings.joblib")
        assert isinstance(embeddings, np.ndarray), (
            f"Expected numpy array, got {type(embeddings).__name__}"
        )
        assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
        assert embeddings.shape[1] == 384, (
            f"Expected 384-dimensional embeddings, got {embeddings.shape[1]}"
        )
        assert np.issubdtype(embeddings.dtype, np.floating), (
            f"Expected floating dtype, got {embeddings.dtype}"
        )


# ---------------------------------------------------------------------------
# Transformed document shape (assignment contract)
# ---------------------------------------------------------------------------


class TestTransformedDocument:
    REQUIRED_FIELDS = {"id", "title", "chunk", "pubDate", "link"}

    def test_each_row_transforms_to_required_fields(self):
        df = pd.read_csv(DATA_DIR / "news_data_dedup.csv")
        for _, row in df.iterrows():
            doc = transform_news_row(row.to_dict())
            missing = self.REQUIRED_FIELDS - set(doc.keys())
            extra = set(doc.keys()) - self.REQUIRED_FIELDS
            assert not missing, (
                f"Row {row.name} is missing fields: {missing}"
            )
            # Extra fields are allowed as long as required ones exist

    def test_required_fields_have_non_null_values(self):
        df = pd.read_csv(DATA_DIR / "news_data_dedup.csv")
        for _, row in df.iterrows():
            doc = transform_news_row(row.to_dict())
            for field in self.REQUIRED_FIELDS:
                assert doc[field] is not None and not (
                    isinstance(doc[field], str) and doc[field].strip() == ""
                ), f"Row {row.name}: field '{field}' is empty or None"

    def test_single_row_demo(self):
        """Demonstrate one complete document to help visual inspection."""
        df = pd.read_csv(DATA_DIR / "news_data_dedup.csv")
        row = df.iloc[0]
        doc = transform_news_row(row.to_dict())
        assert isinstance(doc["id"], str)
        assert isinstance(doc["title"], str)
        assert isinstance(doc["chunk"], str)
        assert isinstance(doc["pubDate"], str)
        assert isinstance(doc["link"], str)

