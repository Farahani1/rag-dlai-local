"""Mechanical Python extraction of code cells from C1M5_Assignment.ipynb.

Each function contains the source from one notebook code cell (numbered by
the notebook's actual cell index; markdown-only cells are skipped). This file
is an adaptation scaffold: it preserves the notebook code for inspection and
future TDD work, but it does not try to make notebook shared state implicit
(see ``C1M5_Assignment_stateful.py`` for that).

When called with ``adapted=True``, cells that depend on Weaviate, Together.ai/
OpenAI, Arize Phoenix/OpenTelemetry, or Coursera-lab-only modules
(``unittests``, ``flask_app``, ``weaviate_server``) use local, Chroma- and
Ollama-backed implementations instead. Cells whose original code has no such
dependency (pure functions, or calls into already-adapted helpers) have no
``if adapted`` split at all — same convention as ``w4/C1M4_Assignment.py``.

W5 adds a ``simplified`` mode to every RAG function (trading a little
accuracy for far fewer LLM tokens) plus tracing spans.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup so code can import sibling modules ─────────────────
_w5_dir = Path(__file__).resolve().parent
if str(_w5_dir) not in sys.path:
    sys.path.insert(0, str(_w5_dir))


def cell_03(adapted: bool = False):
    """Code cell 3 from the notebook."""
    if adapted:
        import json
        import joblib
    else:
        import json
        from weaviate.classes.query import Filter
        import weaviate
        import joblib
        import pandas as pd


def cell_04(adapted: bool = False):
    """Code cell 4 from the notebook."""
    if adapted:
        import json
        from chroma_store import ChromaStore
        from embedding import embed_query
        from utils import (
            ChatWidget,
            generate_with_single_input,
            generate_params_dict,
            parse_json_output,
            get_filter_by_metadata,
            generate_filters_from_query,
            filters_to_where,
            _condition_key,
            process_and_print_query,
            print_properties,
            make_url,
            NullTracer,
            Status,
            StatusCode,
        )
    else:
        import json
        import weaviate_server
        import unittests
        import flask_app
        from utils import (
            ChatWidget,
            generate_with_single_input,
            parse_json_output,
            get_filter_by_metadata,
            generate_filters_from_query,
            process_and_print_query,
            print_properties,
            make_url,
        )


def cell_06(adapted: bool = False):
    """Code cell 6 from the notebook."""
    if adapted:
        from setting import config
        from chroma_store import ChromaStore

        store = ChromaStore(persist_directory=str(config.productsChromaPath))
        if "products_w5" not in store.list_collections():
            raise RuntimeError(
                "Collection 'products_w5' not found. Run `python w5/populate_products.py` "
                "once before using this notebook."
            )
    else:
        import weaviate

        client = weaviate.connect_to_local(port=8079, grpc_port=50050)


def cell_08(adapted: bool = False):
    """Code cell 8 from the notebook."""
    if adapted:
        from utils import NullTracer

        tracer = NullTracer()
    else:
        import phoenix as px
        from phoenix.otel import register
        from opentelemetry.trace import Status, StatusCode


def cell_09(adapted: bool = False):
    """Code cell 9 from the notebook."""
    if adapted:
        # No local Phoenix UI/server in adapted mode; make_url() already
        # prints an informational message instead of a dead URL.
        make_url()
    else:
        # Launch the lab and the URL
        make_url()
        session = px.launch_app()


def cell_11(adapted: bool = False):
    """Code cell 11 from the notebook."""
    make_url("/settings/models")


def cell_14(adapted: bool = False):
    """Code cell 14 from the notebook."""
    if adapted:
        from utils import NullTracer

        phoenix_project_name = "chatbot"
        tracer = NullTracer()
    else:
        # Setting up the telemetry
        phoenix_project_name = "chatbot"

        # With phoenix, we just need to register to get the tracer provider with the appropriate endpoint.
        # Different from the ungraded lab, you will NOT use auto_instrument = True, as there are LLM calls not needed to be traced (examples, calls within unittests etc.)

        tracer_provider_phoenix = register(project_name=phoenix_project_name, endpoint="http://127.0.0.1:6006/v1/traces")

        # Retrieve a tracer for manual instrumentation
        tracer = tracer_provider_phoenix.get_tracer(__name__)


def cell_17(adapted: bool = False):
    """Code cell 17 from the notebook."""
    if adapted:
        import joblib
        from setting import config

        # Loading products data
        products_data = joblib.load(str(config.clothesData))
    else:
        # Loading products data
        products_data = joblib.load('dataset/clothes_json.joblib')


def cell_18(adapted: bool = False):
    """Code cell 18 from the notebook."""
    # Let's get one example
    products_data[0]


def cell_21(adapted: bool = False):
    """Code cell 21 from the notebook."""
    if adapted:
        import joblib
        from setting import config

        faq = joblib.load(str(config.faqData))
    else:
        faq = joblib.load("dataset/faq.joblib")


def cell_22(adapted: bool = False):
    """Code cell 22 from the notebook."""
    # Get an example
    faq[:2]


def cell_26(adapted: bool = False):
    """Code cell 26 from the notebook."""
    # The output is a dictionary containing the role and content from the LLM call, as well as the token usage.:
    result = generate_with_single_input("What are the primary colors?")
    print(json.dumps(result, indent=2))


def cell_27(adapted: bool = False):
    """Code cell 27 from the notebook."""
    # To retreive the content, then you should do as follows:
    print(result['choices'][0]['message']['content'])


def cell_28(adapted: bool = False):
    """Code cell 28 from the notebook."""
    # The total tokens count (input + output) for this is:
    print(result['usage']['total_tokens'])


def cell_32(adapted: bool = False):
    """Code cell 32 from the notebook."""
    def generate_params_dict(
        prompt: str,
        temperature: float = 1.0,
        role: str = 'user',
        top_p: float = 1.0,
        max_tokens: int = 500,
        model: str | None = None
    ) -> dict:
        """
        Generates a dictionary of parameters for calling a Language Learning Model (LLM),
        allowing for the customization of several key options that can affect the output from the model.

        Args:
            prompt (str): The input text that will be provided to the model to guide text generation.
            temperature (float): A value between 0 and 1 that controls the randomness of the model's output;
                lower values result in more repetitive and deterministic results, while higher values enhance randomness.
            role (str): The role designation to be used in context, typically identifying the initiator of the interaction.
            top_p (float): A value between 0 and 1 that manages diversity through the technique of nucleus sampling;
                this parameter limits the set of considered words to the smallest possible while maintaining 'top_p' cumulative probability.
            max_tokens (int): The maximum number of tokens that the model is allowed to generate in response, where a token can
                be as short as one character or as long as one word.
            model (str): The specific model identifier to be utilized for processing the request. This typically specifies both
                the version and configuration of the LLM to be employed.

        Returns:
            dict: A dictionary containing all specified parameters which can then be used to configure and execute a call to the LLM.
        """
        kwargs = {
            "prompt": prompt,
            "role": role,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "model": model
        }
        return kwargs


def cell_33(adapted: bool = False):
    """Code cell 33 from the notebook."""
    kwargs = generate_params_dict("Solve 3x^2 + 5 = 0")
    print(kwargs)


def cell_34(adapted: bool = False):
    """Code cell 34 from the notebook."""
    # Now you can call the LLM
    result = generate_with_single_input(**kwargs)
    content = result['choices'][0]['message']['content']
    total_tokens = result['usage']['total_tokens']
    print(f"Content: {content}\n\nTotal Tokens: {total_tokens}")


def cell_37(adapted: bool = False):
    """Code cell 37 from the notebook. GRADED CELL: check_if_faq_or_product."""
    def check_if_faq_or_product(query, simplified=False):
        """
        Determines whether a given instruction prompt is related to a frequently asked question (FAQ) or a product inquiry.

        Parameters:
        - query (str): The instruction or query that needs to be labeled as either FAQ or Product related.
        - simplified (bool): If True, uses a simplified prompt.

        Returns:
        - str: The label 'FAQ' if the prompt is deemed a frequently asked question, 'Product' if it is related to product information, or
          None if the label is inconclusive.
        """
        if not simplified:
            PROMPT = f"""Label the following instruction as an FAQ related answer or a product related answer for a clothing store.
        Product related answers are answers specific about product information or that needs to use the products to give an answer.
        Examples:
                Is there a refund for incorrectly bought clothes? Label: FAQ
                Where are your stores located?: Label: FAQ
                Tell me about the cheapest T-shirts that you have. Label: Product
                Do you have blue T-shirts under 100 dollars? Label: Product
                What are the available sizes for the t-shirts? Label: FAQ
                How can I contact you via phone? Label: FAQ
                How can I find the promotions? Label: FAQ
                Give me ideas for a sunny look. Label: Product
        Return only one of the two labels: FAQ or Product, nothing more.
        Query to classify: {query}
                 """
        else:
            # Simplified: no few-shot examples, minimal instructions -> far fewer tokens.
            PROMPT = (
                f"Classify this clothing-store query as FAQ or Product. "
                f"Product = about items, prices, or stock. FAQ = policies, shipping, contact, general info.\n"
                f"Query: {query}\n"
                f"Answer with exactly one word: FAQ or Product."
            )

        with tracer.start_as_current_span("routing_faq_or_product", openinference_span_kind='tool') as span:
            span.set_input(str({"query": query, "simplified": simplified}))

            kwargs = generate_params_dict(PROMPT, temperature=0, max_tokens=10)

            with tracer.start_as_current_span("router_call", openinference_span_kind='llm') as router_span:
                router_span.set_input(kwargs)
                try:
                    response = generate_with_single_input(**kwargs)
                except Exception as error:
                    router_span.record_exception(error)
                    router_span.set_status(Status(StatusCode.ERROR))
                else:
                    router_span.set_attribute("llm.token_count.prompt", response['usage']['prompt_tokens'])
                    router_span.set_attribute("llm.token_count.completion", response['usage']['completion_tokens'])
                    router_span.set_attribute("llm.token_count.total", response['usage']['total_tokens'])
                    router_span.set_attribute("llm.model_name", response['model'])
                    router_span.set_attribute("llm.provider", 'ollama')
                    router_span.set_output(response)
                    router_span.set_status(Status(StatusCode.OK))

            label = response['choices'][0]['message']['content']
            total_tokens = response['usage']['total_tokens']
            span.set_output(str({"label": label, 'total_tokens': total_tokens}))
            span.set_status(Status(StatusCode.OK))

            if 'faq' in label.lower():
                label = 'FAQ'
            elif 'product' in label.lower():
                label = 'Product'
            else:
                label = 'undefined'

            return label, total_tokens


def cell_38(adapted: bool = False):
    """Code cell 38 from the notebook."""
    if adapted:
        # Equivalent coverage lives in tests/w5/test_graded_cells.py
        pass
    else:
        unittests.test_check_if_faq_or_product(check_if_faq_or_product)


def cell_40(adapted: bool = False):
    """Code cell 40 from the notebook."""
    queries = [
        'What is your return policy?',
        'Give me three examples of blue T-shirts you have available.',
        'How can I contact the user support?',
        'Do you have blue Dresses?',
        'Create a look suitable for a wedding party happening during dawn.'
    ]

    labels = ['FAQ', 'Product', 'FAQ', 'Product', 'Product']

    for query, correct_label in zip(queries, labels):
        response_std, tokens_std = check_if_faq_or_product(query, simplified=False)
        response_simp, tokens_simp = check_if_faq_or_product(query, simplified=True)

        process_and_print_query(query, correct_label, response_std, tokens_std, response_simp, tokens_simp)


def cell_42(adapted: bool = False):
    """Code cell 42 from the notebook."""
    @tracer.tool
    def generate_faq_layout(faq_dict):
        """
        Generates a formatted string layout for a list of FAQs.

        This function iterates through a dictionary of frequently asked questions (FAQs) and constructs
        a string where each question is followed by its corresponding answer and type.

        Parameters:
        - faq_dict (list): A list of dictionaries, each containing keys 'question', 'answer', and 'type'
          representing an FAQ entry.

        Returns:
        - str: A string representing the formatted layout of FAQs, with each entry on a separate line.
        """
        t = ""
        for f in faq_dict:
            t += f"Question: {f['question']} Answer: {f['answer']} Type: {f['type']}\n"
        return t


def cell_43(adapted: bool = False):
    """Code cell 43 from the notebook."""
    faq_layout = generate_faq_layout(faq)
    print(faq_layout[:1000])


def cell_44(adapted: bool = False):
    """Code cell 44 from the notebook."""
    print(generate_faq_layout(faq[1:2]))


def cell_46(adapted: bool = False):
    """Code cell 46 from the notebook."""
    if adapted:
        # `store` already created in cell_06; the "faq_w5" collection is
        # created (and populated, if empty) in cell_48.
        if "faq_w5" not in store.list_collections():
            store.create_collection("faq_w5")
    else:
        faq_collection = client.collections.get("Faq")


def cell_48(adapted: bool = False):
    """Code cell 48 from the notebook."""
    if adapted:
        import numpy as np

        if store.count("faq_w5") == 0:
            questions = [item['question'] for item in faq]
            embeddings = np.array([embed_query(q) for q in questions])
            documents = [
                {"id": str(i), "question": item['question'], "answer": item['answer'], "type": item['type']}
                for i, item in enumerate(faq)
            ]
            store.add_documents("faq_w5", documents, embeddings)
    else:
        from tqdm import tqdm
        from weaviate.util import generate_uuid5
        with faq_collection.batch.fixed_size(batch_size=20, concurrent_requests=5) as batch:
            for document in tqdm(faq):
                uuid = generate_uuid5(document['question'])
                batch.add_object(properties=document, uuid=uuid)


def cell_50(adapted: bool = False):
    """Code cell 50 from the notebook."""
    if adapted:
        res = store.query("faq_w5", embed_query("What is the return policy?"), top_k=5)
    else:
        res = faq_collection.query.near_text("What is the return policy?", limit=5)


def cell_51(adapted: bool = False):
    """Code cell 51 from the notebook."""
    if adapted:
        for obj in res:
            print_properties(obj)
    else:
        for obj in res.objects:
            print_properties(obj)


def cell_53(adapted: bool = False):
    """Code cell 53 from the notebook. GRADED CELL: query_on_faq."""
    def query_on_faq(query, simplified=False, **kwargs):
        """
        Constructs a prompt to query an FAQ system and generates a response.

        This function integrates an FAQ layout into the prompt to help generate a suitable answer to the given query
        using a language model. It supports additional keyword arguments to customize the prompt generation process.

        Parameters:
        - query (str): The query about which the function seeks to provide an answer from the FAQ.
        - simplified (bool): If True, uses semantic search to extract a relevant subset of FAQ questions
        - **kwargs: Optional keyword arguments for extra configuration of prompt parameters.

        Returns:
        - dict: The kwargs dict generated from the prompt, ready for ``generate_with_single_input``.
        """
        if not simplified:
            with tracer.start_as_current_span("query_on_faq", openinference_span_kind="tool") as span:
                span.set_input({"query": query, "simplified": simplified})
                faq_layout = generate_faq_layout(faq)

                PROMPT = f"""You will be provided with an FAQ for a clothing store.
        Answer the instruction based on it. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ.
        <FAQ_ITEMS>
        PROVIDED FAQ: {faq_layout}
        </FAQ_ITEMS>
        Question: {query}
            """
                span.set_attribute("prompt", PROMPT)

                kwargs = generate_params_dict(PROMPT, **kwargs)

                span.set_attribute("output", str(kwargs))
                span.set_status(Status(StatusCode.OK))

                return kwargs

        else:
            with tracer.start_as_current_span("query_on_faq", openinference_span_kind="tool") as span:
                span.set_input({"query": query, "simplified": simplified})
                with tracer.start_as_current_span("retrieve_faq_questions", openinference_span_kind="retriever") as retrieve_span:

                    if adapted:
                        # Get the 5 most relevant FAQ objects via semantic search.
                        results = store.query("faq_w5", embed_query(query), top_k=5)

                        for i, document in enumerate(results):
                            retrieve_span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.get("id")))
                            retrieve_span.set_attribute(f"retrieval.documents.{i}.document.content", str(document))

                        # Reverse the order to add the most relevant objects in the bottom, so it gets closer to the end of the input
                        results = list(reversed(results))
                        faq_layout = generate_faq_layout(results)
                    else:
                        # Get the 5 most relevant FAQ objects
                        results = faq_collection.query.near_text(query, limit=5)

                        for i, document in enumerate(results.objects):
                            retrieve_span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.uuid))
                            retrieve_span.set_attribute(f"retrieval.documents.{i}.document.metadata", str(document.metadata))
                            retrieve_span.set_attribute(
                                f"retrieval.documents.{i}.document.content", str(document.properties)
                            )
                        results = [x.properties for x in results.objects]
                        results.reverse()
                        faq_layout = generate_faq_layout(results)

                PROMPT = (f"You will be provided with a query for a clothing store regarding FAQ. It will be provided relevant FAQ from the clothing store."
            f"Answer the query based on the relevant FAQ provided. They are ordered in decreasing relevance, so the first is the most relevant FAQ and the last is the least relevant."
            f"Answer the instruction based on them. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ.\n"
            f"<FAQ>\n"
            f"RELEVANT FAQ ITEMS:\n{faq_layout}\n"
            f"</FAQ>\n"
            f"Query: {query}")

                span.set_attribute("prompt", PROMPT)

                kwargs = generate_params_dict(PROMPT, **kwargs)

                span.set_attribute("output", str(kwargs))
                span.set_status(Status(StatusCode.OK))

                return kwargs


def cell_54(adapted: bool = False):
    """Code cell 54 from the notebook."""
    if adapted:
        # Equivalent coverage lives in tests/w5/test_graded_cells.py
        pass
    else:
        unittests.test_query_on_faq(query_on_faq)


def cell_55(adapted: bool = False):
    """Code cell 55 from the notebook."""
    kwargs = query_on_faq("I received the dress I ordered but I don't like it. How can I return it?")


def cell_56(adapted: bool = False):
    """Code cell 56 from the notebook."""
    print(len(kwargs['prompt'].split()))


def cell_58(adapted: bool = False):
    """Code cell 58 from the notebook."""
    result = generate_with_single_input(**kwargs)


def cell_60(adapted: bool = False):
    """Code cell 60 from the notebook."""
    print(result['choices'][0]['message']['content'])


def cell_61(adapted: bool = False):
    """Code cell 61 from the notebook."""
    print(result['usage']['total_tokens'])


def cell_63(adapted: bool = False):
    """Code cell 63 from the notebook."""
    kwargs = query_on_faq("I received the dress I ordered but I don't like it. How can I return it?", simplified=True)


def cell_64(adapted: bool = False):
    """Code cell 64 from the notebook."""
    print(len(kwargs['prompt'].split()))


def cell_65(adapted: bool = False):
    """Code cell 65 from the notebook."""
    result = generate_with_single_input(**kwargs)


def cell_66(adapted: bool = False):
    """Code cell 66 from the notebook."""
    print(result['choices'][0]['message']['content'])


def cell_67(adapted: bool = False):
    """Code cell 67 from the notebook."""
    print(result['usage']['total_tokens'])


def cell_70(adapted: bool = False):
    """Code cell 70 from the notebook. GRADED CELL: decide_task_nature."""
    def decide_task_nature(query, simplified=True):
        """
        Determines the nature of a query, labeling it as either creative or technical.

        This function constructs a prompt for a language model to decide if a given query requires a creative response,
        such as making suggestions or composing ideas, or a technical response, like providing product details or prices.

        Parameters:
        - query (str): The query to be evaluated for its nature.
        - simplified (bool): If True, uses a simplified prompt.

        Returns:
        - str: The label 'creative' if the query requires creative input, or 'technical' if it requires technical information.
        """
        if not simplified:
            PROMPT = f"""Decide if the following query is a query that requires creativity (creating, composing, making new things) or technical (information about products, prices etc.). Label it as creative or technical.
          Examples:
          Give me suggestions on a nice look for a nightclub. Label: creative
          What are the blue dresses you have available? Label: technical
          Give me three Tshirts for summer. Label: technical
          Give me a look for attending a wedding party. Label: creative
          Give me suggestions on clothes that match a green Tshirt. Label: creative
          I would like a suggestion on which products match a green Tshirt I already have. Label: creative

          Query to be analyzed: {query}. Only output one token with the label
          """
        else:
            # Simplified: no few-shot examples -> far fewer tokens.
            PROMPT = (
                f"Is this clothing-store query creative (ideas, outfit suggestions, styling) or "
                f"technical (product facts, prices, availability)?\n"
                f"Query: {query}\n"
                f"Answer with exactly one word: creative or technical."
            )

        with tracer.start_as_current_span("decide_task_nature", openinference_span_kind="tool") as span:
            span.set_input({"query": query, "simplified": simplified})
            kwargs = generate_params_dict(PROMPT, temperature=0, max_tokens=1)

            with tracer.start_as_current_span("router_call", openinference_span_kind='llm') as router_span:
                router_span.set_input(kwargs)
                try:
                    response = generate_with_single_input(**kwargs)
                except Exception as error:
                    router_span.record_exception(error)
                    router_span.set_status(Status(StatusCode.ERROR))
                else:
                    router_span.set_attribute("llm.token_count.prompt", response['usage']['prompt_tokens'])
                    router_span.set_attribute("llm.token_count.completion", response['usage']['completion_tokens'])
                    router_span.set_attribute("llm.token_count.total", response['usage']['total_tokens'])
                    router_span.set_attribute("llm.model_name", response['model'])
                    router_span.set_attribute("llm.provider", 'ollama')
                    router_span.set_output(response)
                    router_span.set_status(Status(StatusCode.OK))

            label = response['choices'][0]['message']['content']
            total_tokens = response['usage']['total_tokens']
            span.set_output(str({"label": label, 'total_tokens': total_tokens}))
            span.set_status(Status(StatusCode.OK))

            return label, total_tokens


def cell_71(adapted: bool = False):
    """Code cell 71 from the notebook."""
    if adapted:
        # Equivalent coverage lives in tests/w5/test_graded_cells.py
        pass
    else:
        unittests.test_decide_task_nature(decide_task_nature)


def cell_72(adapted: bool = False):
    """Code cell 72 from the notebook."""
    queries = ["Give me two sneakers with vibrant colors.",
               "What are the most expensive clothes you have in your catalogue?",
               "I have a green Dress and I like a suggestion on an accessory to match with it.",
               "Give me three trousers with vibrant colors you have in your catalogue.",
               "Create a look for a woman walking in a park on a sunny day. It must be fresh due to hot weather."
               ]

    labels = ['technical', 'technical', 'creative', 'technical', 'creative']


def cell_73(adapted: bool = False):
    """Code cell 73 from the notebook."""
    for query, correct_label in zip(queries, labels):
        response, total_tokens = decide_task_nature(query, simplified=True)
        label = response
        if label == correct_label:
            label = "\033[32m" + label + "\033[0m"
        else:
            label = "\033[31m" + label + "\033[0m"
        if total_tokens > 150:
            total_tokens = "\033[31m" + str(total_tokens) + "\033[0m"
        else:
            total_tokens = "\033[32m" + str(total_tokens) + "\033[0m"
        print(f"Query: {query} Label Predicted: {label}. Correct Label: {correct_label} Total Tokens: {total_tokens}")


def cell_75(adapted: bool = False):
    """Code cell 75 from the notebook."""
    @tracer.tool
    def get_params_for_task(task):
        """
        Retrieves specific language model parameters based on the task nature.

        This function provides parameter sets tailored for creative or technical tasks to optimize
        language model behavior. For creative tasks, higher randomness is encouraged, while technical
        tasks are handled with more focus and precision. A default parameter set is provided for unexpected cases.

        Parameters:
        - task (str): The nature of the task ('creative' or 'technical').

        Returns:
        - dict: A dictionary containing 'top_p' and 'temperature' settings for the specified task.
        """
        PARAMETERS_DICT = {"creative": {'top_p': 0.9, 'temperature': 1},
                           "technical": {'top_p': 0.7, 'temperature': 0.3}}

        if task == 'technical':
            param_dict = PARAMETERS_DICT['technical']

        if task == 'creative':
            param_dict = PARAMETERS_DICT['creative']

        else:
            param_dict = {'top_p': 0.5, 'temperature': 1}

        return param_dict


def cell_77(adapted: bool = False):
    """Code cell 77 from the notebook."""
    products_data[0]


def cell_79(adapted: bool = False):
    """Code cell 79 from the notebook."""
    values = {}
    for d in products_data:
        for key, val in d.items():
            if key in ('product_id', 'price', 'productDisplayName', 'subCategory', 'year'):
                continue
            if key not in values.keys():
                values[key] = set()
            values[key].add(val)


def cell_80(adapted: bool = False):
    """Code cell 80 from the notebook."""
    values['season']


def cell_82(adapted: bool = False):
    """Code cell 82 from the notebook."""
    def generate_metadata_from_query(query):
        """
        Generates metadata in JSON format based on a given query to filter clothing items.

        This function constructs a prompt for a language model to create a JSON object that will
        guide the filtering of a vector database query for clothing items. It takes possible values from
        a predefined set and ensures only relevant metadata is included in the output JSON.

        Parameters:
        - query (str): The query describing specific clothing-related needs.

        Returns:
        - str: A JSON string representing metadata with keys like gender, masterCategory, articleType,
          baseColour, price, usage, and season. Each value in the JSON is within a list, with prices specified
          as a dict containing "min" and "max" values. Unrestricted keys should use ["Any"] and unspecified
          prices should default to {"min": 0, "max": "inf"}.
        """
        PROMPT = f"""
        One query will be provided. For the given query, there will be a call on vector database to query relevant clothing items.
        Generate a JSON with useful metadata to filter the products in the query. Possible values for each feature is in the following json: {values}

        Provide a JSON with the features that best fit in the query (can be more than one, write in a list). Also, if present, add a price key, saying if there is a price range (between values, greater than or smaller than some value).
        Only return the JSON, nothing more. price key must be a json with "min" and "max" values (0 if no lower bound and inf if no upper bound).
        Always include gender, masterCategory, articleType, baseColour, price, usage and season as keys. All values must be within lists.
        If there is no price set, add min = 0 and max = inf.
        Only include values that are given in the json above.

        Example of expected JSON:

        {{
        "gender": ["Women"],
        "masterCategory": ["Apparel"],
        "articleType": ["Dresses"],
        "baseColour": ["Blue"],
        "price": {{"min": 0, "max": "inf"}},
        "usage": ["Formal"],
        "season": ["All seasons"]
        }}

        Query: {query}
                 """
        with tracer.start_as_current_span("generate_metadata_from_query", openinference_span_kind="tool") as span:
            span.set_input(query)
            with tracer.start_as_current_span("llm_call", openinference_span_kind="llm") as metadata_span:
                kwargs = {"prompt": PROMPT, 'temperature': 0, "max_tokens": 1500}
                metadata_span.set_input(kwargs)
                try:
                    response = generate_with_single_input(**kwargs)
                except Exception as error:
                    metadata_span.record_exception(error)
                    metadata_span.set_status(Status(StatusCode.ERROR))
                else:
                    metadata_span.set_attribute("llm.token_count.prompt", response['usage']['prompt_tokens'])
                    metadata_span.set_attribute("llm.token_count.completion", response['usage']['completion_tokens'])
                    metadata_span.set_attribute("llm.token_count.total", response['usage']['total_tokens'])
                    metadata_span.set_attribute("llm.model_name", response['model'])
                    metadata_span.set_attribute("llm.provider", 'ollama')
                    metadata_span.set_output(response)
                    metadata_span.set_status(Status(StatusCode.OK))

            content = response['choices'][0]['message']['content']
            total_tokens = response['usage']['total_tokens']
            span.set_output({"content": content, 'total_tokens': total_tokens})
            span.set_status(Status(StatusCode.OK))

        return content, total_tokens


def cell_83(adapted: bool = False):
    """Code cell 83 from the notebook."""
    content, total_tokens = generate_metadata_from_query("Create a look for a man that suits a sunny day in the park. I don't want to spend more than 300 dollars on each piece.")


def cell_84(adapted: bool = False):
    """Code cell 84 from the notebook."""
    print(content)


def cell_85(adapted: bool = False):
    """Code cell 85 from the notebook."""
    print(total_tokens)


def cell_88(adapted: bool = False):
    """Code cell 88 from the notebook."""
    if adapted:
        products_collection_name = "products_w5"
    else:
        products_collection = client.collections.get('products')


def cell_89(adapted: bool = False):
    """Code cell 89 from the notebook."""
    if adapted:
        store.count(products_collection_name)
    else:
        len(products_collection)


def cell_91(adapted: bool = False):
    """Code cell 91 from the notebook."""
    if adapted:
        import json

        @tracer.tool
        def parse_json_output(llm_output):
            """
            Parses a string output from a language model into a JSON object.
            See ``utils.parse_json_output`` for the full docstring; this
            redefinition mirrors the notebook's own structure (cell 4 imports
            a baseline copy from ``utils``, this cell shadows it with a
            traced version, exactly like the original notebook).
            """
            try:
                llm_output = llm_output.replace("\n", '').replace("'", '').replace("}}", "}").replace("{{", "{")
                parsed_json = json.loads(llm_output)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                return None

        @tracer.tool
        def get_filter_by_metadata(json_output: dict | None = None):
            """
            Generate a list of Chroma filter conditions based on a provided metadata dictionary.

            Parameters:
            - json_output (dict) or None: Dictionary containing metadata keys and their values.

            Returns:
            - list[dict] or None: A list of single-key Chroma filter conditions, or None if input is None.
            """
            if json_output is None:
                return None

            valid_keys = (
                'gender',
                'masterCategory',
                'articleType',
                'baseColour',
                'price',
                'usage',
                'season',
            )

            conditions = []
            for key, value in json_output.items():
                if key not in valid_keys:
                    continue

                if key == 'price':
                    if not isinstance(value, dict):
                        continue
                    min_price = value.get('min')
                    max_price = value.get('max')
                    if min_price is None or max_price is None:
                        continue
                    if min_price <= 0 or max_price == 'inf':
                        continue
                    conditions.append({key: {"$gte": min_price}})
                    conditions.append({key: {"$lte": max_price}})
                else:
                    conditions.append({key: {"$in": list(value) if isinstance(value, (list, set)) else [value]}})

            return conditions if conditions else None

        @tracer.tool
        def generate_filters_from_query(query):
            json_string, total_tokens = generate_metadata_from_query(query)
            json_output = parse_json_output(json_string)
            filters = get_filter_by_metadata(json_output)
            return filters, total_tokens
    else:
        @tracer.tool
        def parse_json_output(llm_output):
            """
            Parses a string output from a language model into a JSON object.

            This function attempts to clean and parse a JSON-formatted string produced by a language model (LLM).
            The input string might contain minor formatting issues, such as unnecessary newlines or single quotes
            instead of double quotes. The function attempts to correct such issues before parsing.

            Parameters:
            - llm_output (str): The string output from the language model that is expected to be in JSON format.

            Returns:
            - dict or None: A dictionary if parsing is successful, or None if the input string cannot be parsed into valid JSON.

            Exception Handling:
            - In case of a JSONDecodeError during parsing, an error message is printed, and the function returns None.
            """
            try:
                llm_output = llm_output.replace("\n", '').replace("'", '').replace("}}", "}").replace("{{", "{")
                parsed_json = json.loads(llm_output)
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                return None

        @tracer.tool
        def get_filter_by_metadata(json_output: dict | None = None):
            """
            Generate a list of Weaviate filters based on a provided metadata dictionary.

            Parameters:
            - json_output (dict) or None: Dictionary containing metadata keys and their values.

            Returns:
            - list[Filter] or None: A list of Weaviate filters, or None if input is None.
            """
            if json_output is None:
                return None

            valid_keys = (
                'gender',
                'masterCategory',
                'articleType',
                'baseColour',
                'price',
                'usage',
                'season',
            )

            filters = []
            for key, value in json_output.items():
                if key not in valid_keys:
                    continue

                if key == 'price':
                    if not isinstance(value, dict):
                        continue
                    min_price = value.get('min')
                    max_price = value.get('max')
                    if min_price is None or max_price is None:
                        continue
                    if min_price <= 0 or max_price == 'inf':
                        continue
                    filters.append(Filter.by_property(key).greater_than(min_price))
                    filters.append(Filter.by_property(key).less_than(max_price))
                else:
                    filters.append(Filter.by_property(key).contains_any(value))

            return filters

        @tracer.tool
        def generate_filters_from_query(query):
            json_string, total_tokens = generate_metadata_from_query(query)
            json_output = parse_json_output(json_string)
            filters = get_filter_by_metadata(json_output)
            return filters, total_tokens


def cell_94(adapted: bool = False):
    """Code cell 94 from the notebook. GRADED CELL: get_relevant_products_from_query."""
    def get_relevant_products_from_query(query, simplified=False):
        """
        Retrieve the most relevant products for a given query by applying semantic search and optional filters.

        This function generates metadata filters from the query and uses them to search for products
        that best match the intended criteria. If `simplified` is True, it performs only a basic semantic
        search with no filters. If the filtered search returns too few results, it progressively reduces
        filtering constraints based on the predefined importance of each filter.

        Parameters:
        query (str): The query string used to search for relevant products.
        simplified (bool): If True, only a simple semantic search is performed without any metadata filters.

        Returns:
        list: A list of product objects that are most relevant to the query.
        total_tokens: The number of tokens used in the LLM call. Returns 0 if simplified search is used.
        """
        if adapted:
            if simplified:
                with tracer.start_as_current_span("get_relevant_products_from_query", openinference_span_kind="retriever") as span:
                    span.set_input({'query': query, 'simplified': simplified})

                    query_emb = embed_query(query)
                    results = store.query("products_w5", query_emb, top_k=20)

                    for i, document in enumerate(results):
                        span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.get("id")))
                        span.set_attribute(f"retrieval.documents.{i}.document.content", str(document))

                    span.set_output({"results": results, "total_tokens": 0})
                    span.set_status(Status(StatusCode.OK))

                    return results, 0  # Total tokens in this case is 0 because there was no LLM call!

            with tracer.start_as_current_span("get_relevant_products_from_query", openinference_span_kind="retriever") as span:
                span.set_input({'query': query, 'simplified': simplified})
                filters, total_tokens = generate_filters_from_query(query)  # Generate filters based on the query
                query_emb = embed_query(query)

                if filters is None or len(filters) == 0:
                    span.set_attribute("retrieval.filters", '')
                    results = store.query("products_w5", query_emb, top_k=20)
                    for i, document in enumerate(results):
                        span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.get("id")))
                        span.set_attribute(f"retrieval.documents.{i}.document.content", str(document))
                    span.set_output({"results": results, "total_tokens": total_tokens})
                    span.set_status(Status(StatusCode.OK))
                    return results, total_tokens

                where = filters_to_where(filters)
                span.set_attribute("retrieval.filters", str(filters))
                results = store.query_with_filter("products_w5", query_emb, top_k=20, metadata_filter=where)
                span.set_attribute("retrieval.len", len(results))
                for i, document in enumerate(results):
                    span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.get("id")))
                    span.set_attribute(f"retrieval.documents.{i}.document.content", str(document))

                # If the result set contains fewer than 10 products, try reducing filters to broaden the search
                importance_order = ['baseColour', 'masterCategory', 'usage', 'masterCategory', 'season', 'articleType', 'gender']
                if len(results) < 10:
                    for i in range(len(importance_order)):
                        with tracer.start_as_current_span(f"refilter_{i}", openinference_span_kind="chain") as refilter_span:
                            filtered_filters = [f for f in filters if _condition_key(f) in importance_order[i + 1:]]
                            refilter_span.set_input(str(filtered_filters))

                            reduced_where = filters_to_where(filtered_filters)
                            if reduced_where is None:
                                results = store.query("products_w5", query_emb, top_k=20)
                            else:
                                results = store.query_with_filter("products_w5", query_emb, top_k=20, metadata_filter=reduced_where)

                            for j, document in enumerate(results):
                                refilter_span.set_attribute(f"retrieval.documents.{j}.document.id", str(document.get("id")))
                                refilter_span.set_attribute(f"retrieval.documents.{j}.document.content", str(document))

                            if len(results) >= 5:
                                refilter_span.set_output(results)
                                refilter_span.set_status(Status(StatusCode.OK))
                                span.set_output(results)
                                span.set_status(Status(StatusCode.OK))
                                return results, total_tokens
                span.set_output(results)
                span.set_status(Status(StatusCode.OK))
                return results, total_tokens

        else:
            if simplified:
                with tracer.start_as_current_span("get_relevant_products_from_query", openinference_span_kind="retriever") as span:
                    span.set_input({'query': query, 'simplified': simplified})

                    results = products_collection.query.near_text(query, limit=20)

                    for i, document in enumerate(results.objects):
                        span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.uuid))
                        span.set_attribute(f"retrieval.documents.{i}.document.metadata", str(document.metadata))
                        span.set_attribute(
                            f"retrieval.documents.{i}.document.content", str(document.properties)
                        )

                    span.set_output({"results": results.objects, "total_tokens": 0})
                    span.set_status(Status(StatusCode.OK))

                    return results.objects, 0

            with tracer.start_as_current_span("get_relevant_products_from_query", openinference_span_kind="retriever") as span:
                span.set_input({'query': query, 'simplified': simplified})
                filters, total_tokens = generate_filters_from_query(query)

                if filters is None or len(filters) == 0:
                    span.set_attribute("retrieval.filters", '')
                    results = products_collection.query.near_text(query, limit=20)
                    for i, document in enumerate(results.objects):
                        span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.uuid))
                        span.set_attribute(f"retrieval.documents.{i}.document.metadata", str(document.metadata))
                        span.set_attribute(
                            f"retrieval.documents.{i}.document.content", str(document.properties)
                        )
                    span.set_output({"results": results.objects, "total_tokens": total_tokens})
                    span.set_status(Status(StatusCode.OK))
                    return results.objects, total_tokens

                span.set_attribute("retrieval.filters", str(filters))
                results = products_collection.query.near_text(query, filters=Filter.all_of(filters), limit=20)
                span.set_attribute("retrieval.len", len(results.objects))
                for i, document in enumerate(results.objects):
                    span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.uuid))
                    span.set_attribute(f"retrieval.documents.{i}.document.metadata", str(document.metadata))
                    span.set_attribute(
                        f"retrieval.documents.{i}.document.content", str(document.properties)
                    )

                importance_order = ['baseColour', 'masterCategory', 'usage', 'masterCategory', 'season', 'articleType', 'gender']
                if len(results.objects) < 10:
                    for i in range(len(importance_order)):
                        with tracer.start_as_current_span(f"refilter_{i}", openinference_span_kind="chain") as refilter_span:
                            filtered_filters = [x for x in filters if x.target in importance_order[i + 1:]]
                            refilter_span.set_input(str(filtered_filters))

                            results = products_collection.query.near_text(query, filters=Filter.all_of(filtered_filters), limit=20)
                            for j, document in enumerate(results.objects):
                                refilter_span.set_attribute(f"retrieval.documents.{j}.document.id", str(document.uuid))
                                refilter_span.set_attribute(f"retrieval.documents.{j}.document.metadata", str(document.metadata))
                                refilter_span.set_attribute(
                                    f"retrieval.documents.{j}.document.content", str(document.properties)
                                )
                            if len(results.objects) >= 5:
                                refilter_span.set_output(results.objects)
                                refilter_span.set_status(Status(StatusCode.OK))
                                span.set_output(results.objects)
                                span.set_status(Status(StatusCode.OK))
                                return results.objects, total_tokens
                span.set_output(results.objects)
                span.set_status(Status(StatusCode.OK))
                return results.objects, total_tokens


def cell_95(adapted: bool = False):
    """Code cell 95 from the notebook."""
    query = "Give me three T-shirts to use in sunny days"


def cell_96(adapted: bool = False):
    """Code cell 96 from the notebook."""
    t, total_tokens = get_relevant_products_from_query(query)


def cell_97(adapted: bool = False):
    """Code cell 97 from the notebook."""
    total_tokens


def cell_99(adapted: bool = False):
    """Code cell 99 from the notebook."""
    t, total_tokens = get_relevant_products_from_query(query, simplified=True)


def cell_100(adapted: bool = False):
    """Code cell 100 from the notebook."""
    total_tokens


def cell_102(adapted: bool = False):
    """Code cell 102 from the notebook."""
    if adapted:
        # Equivalent coverage lives in tests/w5/test_graded_cells.py
        pass
    else:
        unittests.test_get_relevant_products_from_query(get_relevant_products_from_query)


def cell_104(adapted: bool = False):
    """Code cell 104 from the notebook."""
    @tracer.tool
    def generate_items_context(results):
        """
        Compile detailed product information from a list of result objects into a formatted string.

        Parameters:
        results (list): A list of result objects. In adapted mode, each is a plain dict with product
                        attributes; in the original, each has a ``properties`` dict with the same keys.

        Returns:
        str: A multi-line string where each line contains the formatted details of a single product.
        """
        t = ""
        for item in results:
            if not adapted:
                item = item.properties

            t += (
                f"Product ID: {item['product_id']}. "
                f"Product name: {item['productDisplayName']}. "
                f"Product Category: {item['masterCategory']}. "
                f"Product usage: {item['usage']}. "
                f"Product gender: {item['gender']}. "
                f"Product Type: {item['articleType']}. "
                f"Product Category: {item['subCategory']} "
                f"Product Color: {item['baseColour']}. "
                f"Product Season: {item['season']}. "
                f"Product Year: {item['year']}.\n"
            )

        return t


def cell_105(adapted: bool = False):
    """Code cell 105 from the notebook."""
    print(generate_items_context(t)[:1000])


def cell_107(adapted: bool = False):
    """Code cell 107 from the notebook."""
    @tracer.tool
    def query_on_products(query, simplified=False):
        """
        Execute a product query process to generate a response based on the nature of the query.

        Parameters:
        query (str): The input query string that needs to be analyzed and answered using product data.
        simplified (bool): If True, does not use LLM to generate metadata for filtering

        Returns:
        dict: A dictionary of keyword arguments (`kwargs`) containing the prompt and additional settings
              for creating a response, suitable for input to an LLM or other processing system.
        int: Number of tokens used in the process to create the kwargs dictionary
        """
        total_tokens = 0

        query_label, tokens = decide_task_nature(query, simplified=simplified)
        total_tokens += tokens

        parameters_dict = get_params_for_task(query_label)

        relevant_products, tokens = get_relevant_products_from_query(query, simplified=simplified)
        total_tokens += tokens

        context = generate_items_context(relevant_products)

        PROMPT = (
            f"Given the available set of clothing products given by: "
            f"CLOTHING PRODUCTS AVAILABLE:\n{context}\n"
            f"Answer the question that follows.\n"
            f"Never use more than 5 clothing products available below to compose your answer.\n"
            f"Provide the item ID in your answers.\n"
            f"The other information might be provided but not necessarily all of them, pick only the relevant ones for the given query.\n"

            f"QUERY: {query}"
        )

        kwargs = generate_params_dict(PROMPT, role='assistant', **parameters_dict)

        return kwargs, total_tokens


def cell_109(adapted: bool = False):
    """Code cell 109 from the notebook."""
    kwargs, total_tokens = query_on_products('Make a wonderful look for a man attending a wedding party happening during night.', simplified=False)


def cell_110(adapted: bool = False):
    """Code cell 110 from the notebook."""
    result = generate_with_single_input(**kwargs)
    print(result['choices'][0]['message']['content'])


def cell_112(adapted: bool = False):
    """Code cell 112 from the notebook."""
    print(f"Total tokens used in the query is: {total_tokens + result['usage']['total_tokens']}")


def cell_114(adapted: bool = False):
    """Code cell 114 from the notebook."""
    kwargs, total_tokens = query_on_products('Make a wonderful look for a man attending a wedding party happening during night.', simplified=True)


def cell_115(adapted: bool = False):
    """Code cell 115 from the notebook."""
    total_tokens


def cell_116(adapted: bool = False):
    """Code cell 116 from the notebook."""
    result = generate_with_single_input(**kwargs)
    print(result['choices'][0]['message']['content'])


def cell_117(adapted: bool = False):
    """Code cell 117 from the notebook."""
    print(f"Total tokens used in the query is: {total_tokens + result['usage']['total_tokens']}")


def cell_120(adapted: bool = False):
    """Code cell 120 from the notebook."""
    @tracer.tool
    def answer_query(query, model=None, simplified=False):
        """
        Processes a user's query to determine its type (FAQ or Product) and executes the appropriate workflow.

        Parameters:
        - query (str): The query string provided by the user.
        - model (str): The model that will answer the question. Defaults to the local Ollama model.
        - simplified (bool): If True, uses a simplified version of the method. Defaults to False.

        Returns:
        - dict: A dictionary containing keyword arguments for further processing.
          If the query is neither FAQ nor Product-related, returns a default response dictionary instructing
          the assistant to answer based on existing context.
        """
        total_tokens = 0

        label, tokens = check_if_faq_or_product(query, simplified=simplified)
        total_tokens += tokens

        if label not in ['FAQ', 'Product']:
            return {
                "role": "assistant",
                "prompt": (f"User provided a question that does not fit FAQ or Product-related categories. "
                           f"Answer it based on the context you already have. Query provided by the user: {query}")
            }

        if label == 'FAQ':
            kwargs = query_on_faq(query, simplified=simplified)
        elif label == 'Product':
            try:
                kwargs, tokens = query_on_products(query, simplified=simplified)
                total_tokens += tokens
            except Exception:
                return {
                    "role": "assistant",
                    "prompt": (f"User provided a question that broke the querying system. "
                               f"Instruct them to rephrase it. Answer it based on the context you already have. "
                               f"Query provided by the user: {query}")
                }, total_tokens

        kwargs['model'] = model
        return kwargs, total_tokens


def cell_121(adapted: bool = False):
    """Code cell 121 from the notebook."""
    kwargs, total_tokens = answer_query("Give me three examples of blue t-shirts available on your catalogue.", simplified=False)


def cell_122(adapted: bool = False):
    """Code cell 122 from the notebook."""
    result = generate_with_single_input(**kwargs)
    print(result['choices'][0]['message']['content'])


def cell_123(adapted: bool = False):
    """Code cell 123 from the notebook."""
    total_tokens + result['usage']['total_tokens']


def cell_124(adapted: bool = False):
    """Code cell 124 from the notebook."""
    kwargs, total_tokens = answer_query("Give me three examples of blue t-shirts available on your catalogue.", simplified=True)


def cell_125(adapted: bool = False):
    """Code cell 125 from the notebook."""
    result = generate_with_single_input(**kwargs)
    print(result['choices'][0]['message']['content'])


def cell_126(adapted: bool = False):
    """Code cell 126 from the notebook."""
    total_tokens + result['usage']['total_tokens']


def cell_128(adapted: bool = False):
    """Code cell 128 from the notebook."""
    chat_widget_standard = ChatWidget(generator_function=lambda x: answer_query(x, simplified=False), tracer=tracer)


def cell_130(adapted: bool = False):
    """Code cell 130 from the notebook."""
    make_url()


def cell_132(adapted: bool = False):
    """Code cell 132 from the notebook."""
    chat_widget_simplified = ChatWidget(generator_function=lambda x: answer_query(x, simplified=True), tracer=tracer)


def cell_133(adapted: bool = False):
    """Code cell 133 from the notebook."""
    make_url()
