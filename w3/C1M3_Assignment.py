"""Mechanical Python extraction of code cells from C1M3_Assignment.ipynb.

Each function contains the source from one notebook code cell. This file is an
adaptation scaffold: it preserves the notebook code for inspection and future
TDD work, but it does not try to make notebook shared state implicit.

When called with ``adapted=True``, the setup and graded cells use Chroma-backed
implementations instead of the original Weaviate-based code.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup so adapted code can import sibling modules ─────────────────
_w3_dir = Path(__file__).resolve().parent
if str(_w3_dir) not in sys.path:
    sys.path.insert(0, str(_w3_dir))

def cell_04(adapted: bool = False):
    """Code cell 4 from the notebook."""
    if adapted:
        import joblib

        import numpy as np
        import pandas as pd

        from chroma_store import ChromaStore
        from embedding import embed_query
        from retrieval import (
            bm25_retrieve,
            clear_bm25_cache,
            filter_by_metadata,
            generate_final_prompt,
            hybrid_retrieve,
            llm_call,
            semantic_search_retrieve,
            semantic_search_with_reranking,
        )
        from utils import print_object_properties
    else:
        import joblib
        import weaviate
        from weaviate.classes.query import (
            Filter,
            Rerank,
        )


def cell_05(adapted: bool = False):
    """Code cell 5 from the notebook."""
    if adapted:
        from utils import (
            generate_with_single_input,
            print_object_properties,
        )
        # flask_app, weaviate_server, unittests not needed
    else:
        import flask_app
        import weaviate_server
        from utils import (
            generate_with_single_input,
            print_object_properties,
            display_widget,
        )
        import unittests


def cell_08(adapted: bool = False):
    """Code cell 8 from the notebook."""
    if adapted:
        from setting import config
        store = ChromaStore(persist_directory=str(config.chromaPath))
    else:
        client = weaviate.connect_to_local(port=8079, grpc_port=50050)


def cell_10(adapted: bool = False):
    """Code cell 10 from the notebook."""
    if adapted:
        # Load from local CSV + pre-computed embeddings instead of bbc_data.joblib
        df = pd.read_csv("data/news_data_dedup.csv")
        embeddings = joblib.load("data/embeddings.joblib")
        bbc_data = [row.to_dict() for _, row in df.iterrows()]
    else:
        bbc_data = joblib.load("data/bbc_data.joblib")


def cell_11(adapted: bool = False):
    """Code cell 11 from the notebook."""
    if adapted:
        print(f"First item: {bbc_data[0]}")
    else:
        from utils import print_object_properties
        print_object_properties(bbc_data[0])


def cell_13(adapted: bool = False):
    """Code cell 13 from the notebook."""
    if adapted:
        # Transform data and populate Chroma collection
        collection_name = "bbc_collection"
        store.create_collection(collection_name)

        # Transform rows into assignment document shape
        documents = []
        for row in bbc_data:
            doc = {
                "id": str(row.get("guid", "")),
                "title": row.get("title", ""),
                "chunk": row.get("description", row.get("chunk", "")),
                "pubDate": str(row.get("published_at", "")),
                "link": row.get("url", row.get("link", "")),
            }
            documents.append(doc)

        store.add_documents(collection_name, documents, embeddings.astype(np.float32))
        collection = collection_name  # alias for compatibility with usage cells
    else:
        collection = client.collections.get("bbc_collection")


def cell_14(adapted: bool = False):
    """Code cell 14 from the notebook."""
    if adapted:
        print(f"The number of elements in the collection is: {store.count(collection)}")
    else:
        print(f"The number of elements in the collection is: {len(collection)}")


def cell_16(adapted: bool = False):
    """Code cell 16 from the notebook."""
    if adapted:
        from utils import print_object_properties
        docs = store.get_all_documents(collection)
        print("Printing the properties (some will be truncated due to size)")
        if docs:
            print_object_properties(docs[0])
        else:
            print("Collection is empty.")
    else:
        object = collection.query.fetch_objects(limit=1, include_vector=True).objects[0]
        print("Printing the properties (some will be truncated due to size)")
        print_object_properties(object.properties)
        print("Vector: (truncated)", object.vector["main_vector"][0:15])
        print("Vector length: ", len(object.vector["main_vector"]))


def cell_19(adapted: bool = False):
    """Code cell 19 from the notebook."""
    # GRADED CELL

    def filter_by_metadata(
        metadata_property: str,
        values: list[str],
        collection,
        limit: int = 5,
    ) -> list:
        """
        Retrieves objects from a collection based on metadata filtering criteria.

        When ``adapted=True`` (Chroma local), ``collection`` is the collection name (``str``).
        When ``adapted=False`` (original Weaviate), ``collection`` is a Weaviate collection object.
        """
        if adapted:
            from retrieval import filter_by_metadata as _adapted_filter

            return _adapted_filter(
                metadata_property=metadata_property,
                values=values,
                store=store,
                collection_name=collection,
                limit=limit,
            )

        ### START CODE HERE ###

        # Retrieve using collection.query.fetch_objects

        response = None

        ### END CODE HERE ###

        response_objects = [x.properties for x in response.objects]

        return response_objects


def cell_20(adapted: bool = False):
    """Code cell 20 from the notebook."""
    if adapted:
        from utils import print_object_properties
    # Let's get an example
    res = filter_by_metadata("title", ["Taylor Swift"], collection, limit=2)
    for x in res:
        print_object_properties(x)


def cell_22(adapted: bool = False):
    """Code cell 22 from the notebook."""
    if adapted:
        # Adapted: skip the original Weaviate-specific test
        print("Skipping unittests.test_filter_by_metadata (adapted mode)")
    else:
        # Test your solution!
        unittests.test_filter_by_metadata(filter_by_metadata, client)


def cell_24(adapted: bool = False):
    """Code cell 24 from the notebook."""
    # GRADED CELL

    def semantic_search_retrieve(
        query: str,
        collection,
        top_k: int = 5,
    ) -> list:
        """
        Performs a semantic search on a collection and retrieves the top relevant chunks.

        When ``adapted=True`` (Chroma local), ``collection`` is the collection name (``str``).
        When ``adapted=False`` (original Weaviate), ``collection`` is a Weaviate collection object.
        """
        if adapted:
            from retrieval import semantic_search_retrieve as _adapted_ss

            return _adapted_ss(
                query=query,
                store=store,
                collection_name=collection,
                embed_function=embed_query,
                top_k=top_k,
            )

        ### START CODE HERE ###

        # Retrieve using collection.query.near_text
        response = None

        ### END CODE HERE ###

        response_objects = [x.properties for x in response.objects]

        return response_objects


def cell_25(adapted: bool = False):
    """Code cell 25 from the notebook."""
    # Let's have an example!
    print_object_properties(
        semantic_search_retrieve(
            query="Tell me about the last Taylor Swift show",
            collection=collection,
            top_k=2,
        )
    )


def cell_27(adapted: bool = False):
    """Code cell 27 from the notebook."""
    if adapted:
        print("Skipping unittests.test_semantic_search_retrieve (adapted mode)")
    else:
        unittests.test_semantic_search_retrieve(semantic_search_retrieve, client)


def cell_29(adapted: bool = False):
    """Code cell 29 from the notebook."""
    # GRADED CELL

    def bm25_retrieve(
        query: str,
        collection,
        top_k: int = 5,
    ) -> list:
        """
        Performs a BM25 search on a collection and retrieves the top relevant chunks.

        When ``adapted=True`` (Chroma local), ``collection`` is the collection name (``str``).
        When ``adapted=False`` (original Weaviate), ``collection`` is a Weaviate collection object.
        """
        if adapted:
            from retrieval import bm25_retrieve as _adapted_bm25

            clear_bm25_cache()
            return _adapted_bm25(
                query=query,
                store=store,
                collection_name=collection,
                top_k=top_k,
            )

        ### START CODE HERE ###

        # Retrieve using collection.query.bm25
        response = None

        ### END CODE HERE ###

        response_objects = [x.properties for x in response.objects]
        return response_objects


def cell_30(adapted: bool = False):
    """Code cell 30 from the notebook."""
    print_object_properties(
        bm25_retrieve(
            "Tell me about the last Taylor Swift show",
            collection,
            top_k=2,
        )
    )


def cell_32(adapted: bool = False):
    """Code cell 32 from the notebook."""
    if adapted:
        print("Skipping unittests.test_bm25_retrieve (adapted mode)")
    else:
        unittests.test_bm25_retrieve(bm25_retrieve, client)


def cell_34(adapted: bool = False):
    """Code cell 34 from the notebook."""
    # GRADED CELL

    def hybrid_retrieve(
        query: str,
        collection,
        alpha: float = 0.5,
        top_k: int = 5,
    ) -> list:
        """
        Performs a hybrid search on a collection and retrieves the top relevant chunks.

        When ``adapted=True`` (Chroma local), ``collection`` is the collection name (``str``).
        When ``adapted=False`` (original Weaviate), ``collection`` is a Weaviate collection object.
        """
        if adapted:
            from retrieval import hybrid_retrieve as _adapted_hybrid

            return _adapted_hybrid(
                query=query,
                store=store,
                collection_name=collection,
                embed_function=embed_query,
                alpha=alpha,
                top_k=top_k,
            )

        ### START CODE HERE ###

        # Retrieve using collection.query.hybrid
        response = None

        ### END CODE HERE ###

        response_objects = [x.properties for x in response.objects]

        return response_objects


def cell_35(adapted: bool = False):
    """Code cell 35 from the notebook."""
    print_object_properties(
        hybrid_retrieve(
            "Tell me about the last Taylor Swift show",
            collection,
            top_k=2,
        )
    )


def cell_37(adapted: bool = False):
    """Code cell 37 from the notebook."""
    if adapted:
        print("Skipping unittests.test_hybrid_retrieve (adapted mode)")
    else:
        unittests.test_hybrid_retrieve(hybrid_retrieve, client)


def cell_39(adapted: bool = False):
    """Code cell 39 from the notebook."""
    # GRADED CELL

    def semantic_search_with_reranking(
        query: str,
        rerank_property: str,
        collection,
        rerank_query: str = None,
        top_k: int = 5,
    ) -> list:
        """
        Performs a semantic search and reranks the results based on a specified property.

        When ``adapted=True`` (Chroma local), ``collection`` is the collection name (``str``).
        When ``adapted=False`` (original Weaviate), ``collection`` is a Weaviate collection object.
        """
        if adapted:
            from retrieval import semantic_search_with_reranking as _adapted_rr

            return _adapted_rr(
                query=query,
                rerank_property=rerank_property,
                store=store,
                collection_name=collection,
                embed_function=embed_query,
                rerank_query=rerank_query,
                top_k=top_k,
            )

        ### START CODE HERE ###

        # Set the rerank_query to be the same as the query if rerank_query is not passed
        if rerank_query is None:
            rerank_query = query

        # Define the reranker with rerank_query and rerank_property
        reranker = None

        # Retrieve using collection.query.near_text with the appropriate parameters
        response = None

        ### END CODE HERE ###

        response_objects = [x.properties for x in response.objects]

        return response_objects


def cell_41(adapted: bool = False):
    """Code cell 41 from the notebook."""
    # Set a query
    query = "Tell me about the conflicts in Latin America"
    # Get the results from a search (in this case the hybrid search)
    results = semantic_search_with_reranking(
        query,
        collection=collection,
        top_k=2,
        rerank_property="chunk",
    )


def cell_42(adapted: bool = False):
    """Code cell 42 from the notebook."""
    print_object_properties(results)


def cell_44(adapted: bool = False):
    """Code cell 44 from the notebook."""
    if adapted:
        print("Skipping unittests.test_semantic_search_with_reranking (adapted mode)")
    else:
        # Test your function!
        unittests.test_semantic_search_with_reranking(
            semantic_search_with_reranking,
            client,
        )


def cell_46(adapted: bool = False):
    """Code cell 46 from the notebook."""

    def generate_final_prompt(
        query: str,
        top_k: int,
        retrieve_function: callable,
        rerank_query: str = None,
        rerank_property: str = None,
        use_rerank: bool = False,
        use_rag: bool = True,
    ) -> str:
        """
        Generates a final prompt by optionally retrieving and formatting relevant documents
        using retrieval-augmented generation (RAG).

        When ``adapted=True``, uses our ``w3.retrieval.generate_final_prompt`` which expects
        ``store``, ``collection_name``, and ``embed_function`` in addition to the original params.
        """
        if adapted:
            from retrieval import generate_final_prompt as _adapted_gfp

            return _adapted_gfp(
                query=query,
                top_k=top_k,
                retrieve_function=retrieve_function,
                store=store,
                collection_name=collection,
                embed_function=embed_query,
                rerank_query=rerank_query,
                rerank_property=rerank_property,
                use_rerank=use_rerank,
                use_rag=use_rag,
            )

        # If no rag, return the query
        if not use_rag:
            return query

        if use_rerank:
            if rerank_property is None:
                raise ValueError("rerank_property must be set if use_rerank = True")
            top_k_documents = retrieve_function(
                query=query,
                top_k=top_k,
                collection=collection,
                rerank_property=rerank_property,
                rerank_query=rerank_query,
            )
        else:
            top_k_documents = retrieve_function(
                query=query,
                top_k=top_k,
                collection=collection,
            )

        # Initialize an empty string to store the formatted data.
        formatted_data = ""

        # Iterate over each retrieved document.
        for document in top_k_documents:
            # Format each document into a structured string.
            document_layout = (
                f"Title: {document['title']}, Chunk: {document['chunk']}, "
                f"Published at: {document['pubDate']}\nURL: {document['link']}"
            )
            # Append the formatted string to the main data string with a newline for separation.
            formatted_data += document_layout + "\n"

        # If use_rag flag is True, construct the enhanced prompt with the augmented data.
        retrieve_data_formatted = formatted_data  # Store formatted data.
        prompt = (
            f"Answer the user query below. There will be provided additional information for you to compose your answer. "
            f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
            f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
            f"The news data is ordered by relevance."
            f"Query: {query}\n"
            f"2024 News: {retrieve_data_formatted}"
        )

        return prompt


def cell_47(adapted: bool = False):
    """Code cell 47 from the notebook."""
    prompt = generate_final_prompt(
        "Tell me the economic situation of the US in 2024.",
        top_k=5,
        retrieve_function=semantic_search_retrieve,
        use_rerank=False,
        rerank_property="title",
    )


def cell_48(adapted: bool = False):
    """Code cell 48 from the notebook."""
    print(prompt)


def cell_50(adapted: bool = False):
    """Code cell 50 from the notebook."""

    def llm_call(
        query: str,
        retrieve_function: callable = None,
        top_k: int = 5,
        use_rag: bool = True,
        use_rerank: bool = False,
        rerank_property: str = None,
        rerank_query: str = None,
    ) -> str:
        """
        Simulates a call to a language model by generating a prompt and using it to produce a response.

        When ``adapted=True``, uses our ``w3.retrieval.llm_call`` with a local Ollama backend.
        When ``adapted=False``, uses the original ``generate_with_single_input`` (Together API).
        """
        if adapted:
            from retrieval import llm_call as _adapted_llm

            return _adapted_llm(
                query=query,
                retrieve_function=retrieve_function,
                store=store,
                collection_name=collection if isinstance(collection, str) else None,
                embed_function=embed_query,
                top_k=top_k,
                use_rag=use_rag,
                use_rerank=use_rerank,
                rerank_property=rerank_property,
                rerank_query=rerank_query,
                llm_backend=None,  # returns prompt; real LLM call would need config
            )

        # Get the prompt
        PROMPT = generate_final_prompt(
            query,
            top_k=top_k,
            retrieve_function=retrieve_function,
            use_rag=use_rag,
            use_rerank=use_rerank,
            rerank_property=rerank_property,
            rerank_query=rerank_query,
        )

        generated_response = generate_with_single_input(PROMPT)

        generated_message = generated_response["content"]

        return generated_message


def cell_51(adapted: bool = False):
    """Code cell 51 from the notebook."""
    query = "Tell me about United States and Brazil's relationship over the course of 2024. Provide links for the resources you use in the answer."


def cell_52(adapted: bool = False):
    """Code cell 52 from the notebook."""
    # Result with reranked results
    print(
        llm_call(
            query=query,
            top_k=5,
            retrieve_function=hybrid_retrieve,
        )
    )


def cell_54(adapted: bool = False):
    """Code cell 54 from the notebook."""
    display_widget(
        llm_call,
        semantic_search_retrieve,
        bm25_retrieve,
        hybrid_retrieve,
        semantic_search_with_reranking,
    )