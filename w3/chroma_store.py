"""Thin adapter around ChromaDB.

The adapter hides ChromaDB's native types and returns plain Python dictionaries.
Notebook functions should talk to this adapter, not directly to ChromaDB.
"""

from __future__ import annotations

from typing import Any

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError


class ChromaStore:
    """A minimal wrapper around a ChromaDB client.

    Parameters
    ----------
    persist_directory : str or None
        If ``None``, use an ephemeral (in-memory) store.
        If set, use a persistent store at the given directory.
    """

    def __init__(self, persist_directory: str | None = None) -> None:
        if persist_directory is None:
            self._client: ClientAPI = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=persist_directory)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, name: str) -> None:
        """Create a collection if it does not already exist."""
        self._client.get_or_create_collection(name=name)

    def list_collections(self) -> list[str]:
        """Return the names of all existing collections."""
        return [col.name for col in self._client.list_collections()]

    def delete_collection(self, name: str) -> None:
        """Delete a collection by name.

        Raises ``ValueError`` if the collection does not exist.
        """
        try:
            self._client.delete_collection(name)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Collection '{name}' could not be deleted: {exc}"
            ) from exc

    def count(self, collection_name: str) -> int:
        """Return the number of documents in a collection."""
        try:
            collection = self._client.get_collection(collection_name)
        except NotFoundError as exc:
            raise ValueError(
                f"Collection '{collection_name}' not found"
            ) from exc
        return collection.count()

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def add_documents(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Add documents with pre-computed embeddings to a collection.

        Parameters
        ----------
        collection_name :
            Name of the target collection (must already exist).
        documents :
            Each dict must contain ``id``, ``title``, ``chunk``, ``pubDate``,
            and ``link`` keys.
        embeddings :
            A 2D array of shape ``(len(documents), D)``.
        """
        try:
            collection = self._client.get_collection(collection_name)
        except NotFoundError as exc:
            raise ValueError(
                f"Collection '{collection_name}' not found"
            ) from exc

        ids = [doc["id"] for doc in documents]
        metadatas = [
            {
                "title": doc["title"],
                "chunk": doc["chunk"],
                "pubDate": doc["pubDate"],
                "link": doc["link"],
            }
            for doc in documents
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
            metadatas=metadatas,
        )

    def get_all_documents(self, collection_name: str) -> list[dict[str, Any]]:
        """Return all documents from a collection.

        Parameters
        ----------
        collection_name :
            Name of the collection.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``title``, ``chunk``, ``pubDate``,
            and ``link`` keys.  Returns an empty list for an empty collection.
        """
        try:
            collection = self._client.get_collection(collection_name)
        except NotFoundError as exc:
            raise ValueError(
                f"Collection '{collection_name}' not found"
            ) from exc

        if collection.count() == 0:
            return []

        result = collection.get(include=["metadatas"])

        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])

        output: list[dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            output.append(
                {
                    "id": doc_id,
                    "title": meta.get("title", ""),
                    "chunk": meta.get("chunk", ""),
                    "pubDate": meta.get("pubDate", ""),
                    "link": meta.get("link", ""),
                }
            )

        return output

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_with_filter(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a vector similarity search with optional metadata filter.

        Parameters
        ----------
        collection_name :
            Name of the collection to search.
        query_embedding :
            D-dimensional embedding vector.
        top_k :
            Number of nearest neighbours to return.
        metadata_filter :
            Optional Chroma ``where`` filter dict.  If ``None``, no filtering
            is applied.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``title``, ``chunk``, ``pubDate``,
            and ``link`` keys.  Returns an empty list for an empty collection.
        """
        try:
            collection = self._client.get_collection(collection_name)
        except NotFoundError as exc:
            raise ValueError(
                f"Collection '{collection_name}' not found"
            ) from exc

        if collection.count() == 0:
            return []

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter

        result = collection.query(**kwargs)

        ids = result["ids"][0] if result["ids"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []

        output: list[dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            output.append(
                {
                    "id": doc_id,
                    "title": meta.get("title", ""),
                    "chunk": meta.get("chunk", ""),
                    "pubDate": meta.get("pubDate", ""),
                    "link": meta.get("link", ""),
                }
            )

        return output

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Run a vector similarity search.

        Parameters
        ----------
        collection_name :
            Name of the collection to search.
        query_embedding :
            D-dimensional embedding vector.
        top_k :
            Number of nearest neighbours to return.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``title``, ``chunk``, ``pubDate``,
            and ``link`` keys.  Returns an empty list for an empty collection.
        """
        try:
            collection = self._client.get_collection(collection_name)
        except NotFoundError as exc:
            raise ValueError(
                f"Collection '{collection_name}' not found"
            ) from exc

        if collection.count() == 0:
            return []

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        # result is a dict with keys: ids, embeddings, documents, uris, metadatas, distances
        ids = result["ids"][0] if result["ids"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []

        output: list[dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            output.append(
                {
                    "id": doc_id,
                    "title": meta.get("title", ""),
                    "chunk": meta.get("chunk", ""),
                    "pubDate": meta.get("pubDate", ""),
                    "link": meta.get("link", ""),
                }
            )

        return output