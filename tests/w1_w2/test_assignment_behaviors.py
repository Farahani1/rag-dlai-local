from types import SimpleNamespace

from .support import (
    PROJECT_ROOT,
    W1_NOTEBOOK,
    W2_NOTEBOOK,
    TinyMatrix,
    TinyScores,
    TinyVector,
    configure_module_retrieval,
    import_from_path,
    install_import_stubs,
    load_notebook_functions,
)


def test_w1_utils_local_behaviors(monkeypatch):
    requests = install_import_stubs(monkeypatch)
    module = import_from_path("w1_behavior_utils", PROJECT_ROOT / "w1" / "utils.py")

    assert module.format_date("April 25, 2024 10:30 PM") == "2024-04-25"
    assert module.concatenate_fields(
        [{"title": "A", "description": "B", "ignored": "C"}],
        ["title", "description"],
    ) == ["A B"]
    assert module.generate_with_single_input("hello") == {
        "role": "assistant",
        "content": "local answer",
    }
    assert requests.last_payload["model"] == "dummy-llm"
    assert requests.last_payload["options"]["num_predict"] == 50

    configure_module_retrieval(module, scores=[0.2, 0.9, 0.4])
    assert module.retrieve("local news", top_k=2) == [1, 2]


def test_w2_utils_local_behaviors(monkeypatch):
    requests = install_import_stubs(monkeypatch)
    module = import_from_path("w2_behavior_utils", PROJECT_ROOT / "w2" / "utils.py")

    assert module.format_date("2024-04-26T09:15:00Z") == "2024-04-26"
    assert module.concatenate_fields(
        [{"title": "Semantic", "description": "BM25"}],
        ["title", "description"],
    ) == ["Semantic BM25"]
    assert module.generate_with_single_input("hello", top_p=0.9)["content"] == " local answer "
    assert requests.last_payload["options"]["temperature"] == 0.2
    assert requests.last_payload["options"]["top_p"] == 0.9

    configure_module_retrieval(module, scores=[0.1, 0.5, 0.4])
    assert module.retrieve("semantic query", top_k=2) == [1, 2]


def test_w1_key_notebook_functions_run_on_tiny_inputs():
    news_data = [
        {
            "title": "Local retrieval",
            "description": "A tiny smoke document",
            "published_at": "2024-04-25",
            "url": "https://example.test/local",
        },
        {
            "title": "Second item",
            "description": "Another tiny smoke document",
            "published_at": "2024-04-26",
            "url": "https://example.test/second",
        },
    ]
    namespace = {
        "NEWS_DATA": news_data,
        "retrieve": lambda query, top_k=5: list(range(top_k)),
        "list": list,
        "dict": dict,
    }

    load_notebook_functions(
        W1_NOTEBOOK,
        ["query_news", "get_relevant_data", "format_relevant_data", "generate_final_prompt"],
        namespace,
    )

    assert namespace["query_news"]([1]) == [news_data[1]]
    assert namespace["get_relevant_data"]("local", top_k=2) == news_data

    formatted = namespace["format_relevant_data"]([news_data[0]])
    for expected_text in ["title", "description", "Published", "URL", "Local retrieval"]:
        assert expected_text.lower() in formatted.lower()

    plain_prompt = namespace["generate_final_prompt"]("What happened?", use_rag=False)
    assert plain_prompt == "What happened?"

    rag_prompt = namespace["generate_final_prompt"](
        "What happened?",
        top_k=1,
        prompt="Question: {query}\nDocs: {documents}",
    )
    assert "Question: What happened?" in rag_prompt
    assert "Local retrieval" in rag_prompt


def test_w2_key_notebook_functions_run_on_tiny_inputs():
    corpus = ["GDP growth report", "music tour revenue", "climate policy"]

    class TinyBM25Retriever:
        def index(self, tokenized_data):
            self.indexed = True

        def retrieve(self, tokenized_query, k=5):
            return [corpus[:k]], [[1.0] * min(k, len(corpus))]

    def tokenize(texts):
        return texts

    def cosine_similarity(query, embeddings):
        return [TinyScores([0.2, 0.9, 0.4])]

    namespace = {
        "BM25_RETRIEVER": TinyBM25Retriever(),
        "TOKENIZED_DATA": ["GDP", "music", "climate"],
        "bm25s": SimpleNamespace(tokenize=tokenize),
        "corpus": corpus,
        "model": SimpleNamespace(encode=lambda query: TinyVector([1.0, 0.0])),
        "EMBEDDINGS": TinyMatrix([[0.0, 0.1], [0.9, 0.1], [0.3, 0.2]]),
        "cosine_similarity": cosine_similarity,
        "np": SimpleNamespace(argsort=lambda values: sorted(range(len(values)), key=values.__getitem__)),
        "list": list,
        "dict": dict,
    }

    load_notebook_functions(
        W2_NOTEBOOK,
        [
            "query_news",
            "bm25_retrieve",
            "semantic_search_retrieve",
            "reciprocal_rank_fusion",
            "generate_final_prompt",
        ],
        namespace,
    )

    assert namespace["bm25_retrieve"]("GDP", top_k=2) == [0, 1]
    assert namespace["semantic_search_retrieve"]("GDP", top_k=2) == [1, 2]
    assert namespace["reciprocal_rank_fusion"]([1, 2, 3], [2, 1, 4], top_k=3) == [1, 2, 3]

    news_data = [
        {
            "title": "GDP growth report",
            "description": "A tiny economic smoke document",
            "published_at": "2024-04-25",
            "url": "https://example.test/gdp",
        },
        {
            "title": "Music tour revenue",
            "description": "A tiny music smoke document",
            "published_at": "2024-04-26",
            "url": "https://example.test/music",
        },
    ]
    namespace["NEWS_DATA"] = news_data

    def fake_retrieve(query, top_k=5):
        return list(range(top_k))

    plain_prompt = namespace["generate_final_prompt"](
        "What happened?",
        top_k=1,
        retrieve_function=fake_retrieve,
        use_rag=False,
    )
    assert plain_prompt == "What happened?"

    rag_prompt = namespace["generate_final_prompt"](
        "What happened?",
        top_k=2,
        retrieve_function=fake_retrieve,
        use_rag=True,
    )
    assert "Query: What happened?" in rag_prompt
    assert "GDP growth report" in rag_prompt
    assert "Music tour revenue" in rag_prompt
