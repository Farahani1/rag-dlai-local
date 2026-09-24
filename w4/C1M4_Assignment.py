"""Mechanical Python extraction of code cells from C1M4_Assignment.ipynb.

Each function contains the source from one notebook code cell. This file is an
adaptation scaffold: it preserves the notebook code for inspection and future
TDD work, but it does not try to make notebook shared state implicit.

When called with ``adapted=True``, the setup and data-loading cells use
local Chroma-backed implementations and config-based data paths instead of
Weaviate and hardcoded paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup so code can import sibling modules ─────────────────
_w4_dir = Path(__file__).resolve().parent
if str(_w4_dir) not in sys.path:
    sys.path.insert(0, str(_w4_dir))


def cell_04(adapted: bool = False):
    """Code cell 4 from the notebook."""
    if adapted:
        import json
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
        from utils import (
            generate_with_single_input,
            print_object_properties,
        )
    else:
        import json
        from weaviate.classes.query import Filter
        import weaviate
        import joblib


def cell_05(adapted: bool = False):
    """Code cell 5 from the notebook."""
    if adapted:
        from utils import (
            ChatWidget,
            generate_with_single_input,
            generate_params_dict,
        )
    else:
        import unittests
        import flask_app
        import weaviate_server
        from utils import (
            ChatWidget,
            generate_with_single_input,
            generate_params_dict
        )


def cell_07(adapted: bool = False):
    """Code cell 7 from the notebook."""
    if adapted:
        from setting import config

        from chroma_store import ChromaStore

        store = ChromaStore(persist_directory=str(config.productsChromaPath))
        # Collection should already exist from w4/populate_products.py
        if "products" not in store.list_collections():
            store.create_collection("products")
    else:
        import weaviate

        client = weaviate.connect_to_local(port=8079, grpc_port=50050)


def cell_09(adapted: bool = False):
    """Code cell 9 from the notebook."""
    # An output example is
    kwargs = generate_params_dict("Solve x^2 - 1 = 0", temperature=1.2, top_p=0.2)
    print(kwargs)


def cell_10(adapted: bool = False):
    """Code cell 10 from the notebook."""
    # Generating
    response = generate_with_single_input(**kwargs)
    print(response['content'])


def cell_12(adapted: bool = False):
    """Code cell 12 from the notebook."""
    if adapted:
        import joblib
        from setting import config
        products_data = joblib.load(str(config.clothesData))
    else:
        # Loading products data
        products_data = joblib.load('dataset/clothes_json.joblib')


def cell_13(adapted: bool = False):
    """Code cell 13 from the notebook."""
    # Let's get one example
    products_data[0]


def cell_16(adapted: bool = False):
    """Code cell 16 from the notebook."""
    if adapted:
        import joblib
        from setting import config
        faq = joblib.load(str(config.faqData))
    else:
        faq = joblib.load("dataset/faq.joblib")


def cell_17(adapted: bool = False):
    """Code cell 17 from the notebook."""
    # Get an example
    faq[:2]


def cell_22(adapted: bool = False):
    """Code cell 22 from the notebook."""
    # GRADED CELL

    def check_if_faq_or_product(query: str) -> str:
        """
        Determines whether a given instruction prompt is related to a frequently asked question (FAQ) or a product inquiry.

        Parameters:
        - query (str): The instruction or query to be labeled as either FAQ or product-related.

        Returns:
        - str: The label 'FAQ' if the prompt is classified as a frequently asked question, 'Product' if it relates to product information, or
          None if the label is inconclusive.
        """
        ### START CODE HERE ###

        # Set the hardcoded prompt. Remember to include the query, clear instructions (explicitly tell the LLM to return FAQ or Product)
        # Include examples of question / desired label pairs.

        prompt = f"""Label the following instruction as an FAQ-related query or a product-related query.
Product-related answers are specific to product information or require using product details to answer. Products are clothes from a store.
An FAQ question addresses common inquiries and provides answers to help users find the information they need.
Examples:
        Is there a refund for incorrectly bought clothes? Label: FAQ
        Tell me about the cheapest T-shirts that you have. Label: Product
        Do you have blue T-shirts under 100 dollars? Label: Product
        I bought a T-shirt and I didn't like it. How can I get a refund? Label: FAQ

Return only one of the two labels: FAQ or Product.
Instruction: {query}"""

        # Get the kwargs dictionary to call the LLM, with PROMPT as prompt, low temperature (0.3 - 0.5)
        # The function call is generate_params_dict, pass the PROMPT and the correct temperature
        kwargs = generate_params_dict(prompt, temperature=0.3, max_tokens=1)

        # Call generate_with_single_input with **kwargs
        response = generate_with_single_input(**kwargs)
        # Get the label by accessing the 'content' key of the response dictionary

        label = response['content'].strip()

        if label not in ['FAQ', 'Product']:
            return None

        ### END CODE HERE ###

        return label


def cell_23(adapted: bool = False):
    """Code cell 23 from the notebook."""
    queries = ['What is your return policy?',
               'Give me three examples of blue T-shirts you have available.',
               'How can I contact the user support?',
               'Do you have blue Dresses?',
               'Create a look suitable for a wedding party happening during dawn.']

    for query in queries:
        response = check_if_faq_or_product(query)
        label = response
        print(f"Query: {query} Label: {label}")


def cell_25(adapted: bool = False):
    """Code cell 25 from the notebook."""
    if adapted:
        # unittests not available in adapted mode; skip
        pass
    else:
        unittests.test_check_if_faq_or_product(check_if_faq_or_product)


def cell_27(adapted: bool = False):
    """Code cell 27 from the notebook."""
    # print the structure of the first element
    faq[0]


def cell_29(adapted: bool = False):
    """Code cell 29 from the notebook."""
    def generate_faq_layout(faq_dict: list) -> str:
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
        # Initialize an empty string
        t = ""

        # Iterate over every FAQ question in the FAQ list
        for f in faq_dict:
            # Append the question with formatted string (remember to use f-string and access the values as f['question'], f['answer'] and so on)
            # Also, do not forget to add a new line character (\n) at the end of each line.
            t += f"Question: {f['question']} Answer: {f['answer']} Type: {f['type']}\n"


        return t


def cell_30(adapted: bool = False):
    """Code cell 30 from the notebook."""
    faq_layout = generate_faq_layout(faq)
    print(faq_layout[:1000])


def cell_32(adapted: bool = False):
    """Code cell 32 from the notebook."""
    # GRADED CELL

    def query_on_faq(query: str, **kwargs) -> dict:
        """
        Constructs a prompt to query an FAQ system and generates a response.

        Parameters:
        - query (str): The query about which the function seeks to provide an answer from the FAQ.
        - **kwargs: Optional keyword arguments for extra configuration of prompt parameters.

        Returns:
        - str: The response generated from the LLM based on the input query and FAQ layout.

        """
        ### START CODE HERE ###

        # Make the prompt. Don't forget to add the faq_layout and the query in it!
        prompt = f"""You will be provided with an FAQ for a cloth store.
    Answer the instruction based on it. You might use more than one question and answer to make your answer. Only answer the question and do not mention that you have access to a FAQ.
    <FAQ>
    PROVIDED FAQ: {faq_layout}
    </FAQ>
    Question: {query}"""

        # Generate the parameters dict with PROMPT and **kwargs
        kwargs = generate_params_dict(prompt, **kwargs)

        ### END CODE HERE ###

        return kwargs


def cell_33(adapted: bool = False):
    """Code cell 33 from the notebook."""
    kwargs = query_on_faq("I got my cloth but I didn't like it. How can I return it?")


def cell_34(adapted: bool = False):
    """Code cell 34 from the notebook."""
    content = generate_with_single_input(**kwargs)


def cell_35(adapted: bool = False):
    """Code cell 35 from the notebook."""
    print(content['content'])


def cell_37(adapted: bool = False):
    """Code cell 37 from the notebook."""
    if adapted:
        # unittests not available in adapted mode; skip
        pass
    else:
        unittests.test_query_on_faq(query_on_faq)


def cell_40(adapted: bool = False):
    """Code cell 40 from the notebook."""
    # GRADED CELL

    def decide_task_nature(query: str) -> str:
        """
        Determines whether a query is creative or technical.

        This function constructs a prompt for an LLM to decide if a given query requires a creative response,
        such as making suggestions or composing ideas, or a technical response, such as providing product details or prices.

        Parameters:
        - query (str): The query to be evaluated for its nature.

        Returns:
        - str: The label 'creative' if the query requires creative input, or 'technical' if it requires technical information.
        """

        ### START CODE HERE ###

        # Create the prompt. Remember to include the query, examples, and clear instructions (not necessarily in this order!)
        prompt = f"""Decide if the following query is a query that requires creativity (creating, composing, making new things) or technical (information about products, prices, etc.). Label it as creative or technical.
              Examples:
              Give me suggestions on a nice look for a nightclub. Label: creative
              What are the blue dresses you have available? Label: technical
              Give me three T-shirts for summer. Label: technical
              Give me a look for attending a wedding party. Label: creative
              Query to be analyzed: {query}. Only output one token: the label."""

        # Generate the kwargs dictionary by passing the PROMPT, setting temperature to 0 and max_tokens to 1
        kwargs = generate_params_dict(prompt, temperature=0, max_tokens=1)

        # Generate the response using generate_with_single_input and **kwargs
        response = generate_with_single_input(**kwargs)

        # Get the label
        label = response['content'].strip().lower()

        ### END CODE HERE ###

        return label


def cell_41(adapted: bool = False):
    """Code cell 41 from the notebook."""
    queries = ["Give me two sneakers with vibrant colors.",
               "What are the most expensive clothes you have in your catalogue?",
               "I have a green dress and I like a suggestion on an accessory to match with it.",
               "Give me three trousers with vibrant colors you have in your catalogue.",
               "Create a look for a woman walking in a park on a sunny day. It must be fresh due to hot weather."
               ]


def cell_42(adapted: bool = False):
    """Code cell 42 from the notebook."""
    for query in queries:
        label = decide_task_nature(query)
        print(f"Query: {query} Label: {label}")


def cell_44(adapted: bool = False):
    """Code cell 44 from the notebook."""
    if adapted:
        # unittests not available in adapted mode; skip
        pass
    else:
        unittests.test_decide_task_nature(decide_task_nature)


def cell_46(adapted: bool = False):
    """Code cell 46 from the notebook."""
    # GRADED CELL

    def get_params_for_task(task: str) -> dict:
        """
        Retrieves specific LLM parameters based on the nature of the task.

        This function returns parameter sets optimized for either creative or technical tasks.
        Creative tasks benefit from higher randomness, while technical tasks require more focus and precision.
        A default parameter set is returned for unrecognized task types.

        Parameters:
        - task (str): The nature of the task ('creative' or 'technical').

        Returns:
        - dict: A dictionary containing 'top_p' and 'temperature' settings appropriate for the task.
        """
        ### START CODE HERE ###
        # Define the parameter sets for technical and creative tasks
        PARAMETERS_DICT = {
            "creative": {"top_p": 0.8, 'temperature': 1.0},
            "technical": {'top_p': 0.7, 'temperature': 0.3}
        }

        # Return the corresponding parameter set based on task type
        if task == 'technical':
            param_dict = PARAMETERS_DICT['technical']
        elif task == 'creative':
            param_dict = PARAMETERS_DICT['creative']
        else:
            # Fallback to a default parameter set for unrecognized task types
            param_dict = {'top_p': 0.7, 'temperature': 0.5}
        ### END CODE HERE ###

        return param_dict


def cell_47(adapted: bool = False):
    """Code cell 47 from the notebook."""
    get_params_for_task("technical")


def cell_49(adapted: bool = False):
    """Code cell 49 from the notebook."""
    if adapted:
        # unittests not available in adapted mode; skip
        pass
    else:
        unittests.test_get_params_for_task(get_params_for_task)


def cell_51(adapted: bool = False):
    """Code cell 51 from the notebook."""
    # Let's remember the data structure of a product
    products_data[0]


def cell_52(adapted: bool = False):
    """Code cell 52 from the notebook."""
    # Run this cell to generate the dictionary with the possible values for each key
    values = {}
    for d in products_data:
        for key, val in d.items():
            if key in ('product_id', 'price', 'productDisplayName', 'subCategory', 'year'):
                continue
            if key not in values.keys():
                values[key] = set()
            values[key].add(val)


def cell_53(adapted: bool = False):
    """Code cell 53 from the notebook."""
    # Example of possible values for the feature 'season'
    values['season']


def cell_55(adapted: bool = False):
    """Code cell 55 from the notebook."""
    # GRADED CELL

    def generate_metadata_from_query(query: str) -> str:
        """
        Generates metadata in JSON format based on a given query to filter clothing items.

        This function constructs a prompt for an LLM to produce a JSON object
        that will guide filtering in a vector database query for clothing items.
        It uses possible values from a predefined set and ensures that only relevant metadata
        is included in the output JSON.

        Parameters:
        - query (str): A description of specific clothing-related needs.

        Returns:
        - str: A JSON string representing metadata with keys such as gender, masterCategory,
          articleType, baseColour, price, usage, and season. Each value in the JSON is a list.
          The price is specified as a dictionary with "min" and "max" keys.
          For unrestricted categories, use ["Any"], and if no price is specified,
          default to {"min": 0, "max": "inf"}.
        """
        ### START CODE HERE ###

        # Construct the prompt.
        # Include the query, the desired JSON format, and the possible values (pass {values} where needed).
        # Clearly instruct the LLM to include gender, masterCategory, articleType, baseColour, price, usage, and season as keys.
        # Specify that the price key must be a JSON object with "min" and "max" values (0 if no lower bound, "inf" if no upper bound).
        # If no price is set, default to min = None
        prompt = f"""A query will be provided. Based on this query, a vector database will be searched to find relevant clothing items.
    Generate a JSON object containing useful metadata to filter products for this query.
    The possible values for each feature are given in the following JSON: {values}

    Provide a JSON containing the features that best match the query (values should be in lists, multiple values possible).
    If a price range is mentioned, include a price key specifying the range (between values, greater than, or less than).
    Return only the JSON, nothing else. The price key must be a JSON object with "min" and "max" values (use 0 if no lower bound, and "inf" if no upper bound).
    Always include the following keys: gender, masterCategory, articleType, baseColour, price, usage, and season.
    If no price is specified, set min = 0 and max = inf.
    Include only values present in the JSON above.

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

    Query: {query}"""

        # Generate the response with generate_with_single_input using PROMPT, temperature=0 (low randomness), and max_tokens=1500
        kwargs = generate_params_dict(prompt, temperature=0, max_tokens=1500)
        response = generate_with_single_input(**kwargs)

        # Extract the content from the response
        content = response['content']

        ### END CODE HERE ###

        return content


def cell_56(adapted: bool = False):
    """Code cell 56 from the notebook."""
    print(generate_metadata_from_query("Create a look for a man that suits a sunny day in the park. I don't want to spend more than 300 dollars on each piece."))


def cell_59(adapted: bool = False):
    """Code cell 59 from the notebook."""
    def parse_json_output(llm_output: str) -> dict:
        """
        Parses a string output from an LLM into a JSON object.

        This function attempts to clean and parse a JSON-formatted string produced by an LLM.
        The input string might contain minor formatting issues, such as unnecessary newlines or single quotes
        instead of double quotes. The function attempts to correct such issues before parsing.

        Parameters:
        - llm_output (str): The string output from the LLM that is expected to be in JSON format.

        Returns:
        - dict or None: A dictionary if parsing is successful, or None if the input string cannot be parsed into valid JSON.

        Exception Handling:
        - In case of a JSONDecodeError during parsing, an error message is printed, and the function returns None.
        """
        try:
            # Since the input might be improperly formatted, ensure any single quotes are removed
            llm_output = llm_output.replace("\n", '').replace("'",'').replace("}}", "}").replace("{{", "{")  # Remove any erroneous structures

            # Attempt to parse JSON directly provided it is a properly-structured JSON string
            parsed_json = json.loads(llm_output)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return None


def cell_60(adapted: bool = False):
    """Code cell 60 from the notebook."""
    json_string = generate_metadata_from_query("Give me three blue dresses suitable for a wedding party, less than 200 dollars and at least 50 dollars")
    json_output = parse_json_output(json_string)


def cell_61(adapted: bool = False):
    """Code cell 61 from the notebook."""
    json_output


def cell_63(adapted: bool = False):
    """Code cell 63 from the notebook."""
    if adapted:
        # In adapted mode, the ChromaStore was created in cell_07 as `store`
        from chroma_store import ChromaStore
        # `store` is available from cell_07; use it directly
        pass
    else:
        products_collection = client.collections.get('products')


def cell_64(adapted: bool = False):
    """Code cell 64 from the notebook."""
    if adapted:
        from chroma_store import ChromaStore
        # Use the store to get collection count
        len(store.get_all_documents("products"))
    else:
        len(products_collection)


def cell_66(adapted: bool = False):
    """Code cell 66 from the notebook."""
    if adapted:
        import json

        def get_filter_by_metadata(json_output: dict | None = None):
            """
            Generate a Chroma where-filter dict based on a provided metadata dictionary.

            Parameters:
            - json_output (dict) or None: Dictionary containing metadata keys and their values.

            Returns:
            - dict or None: A Chroma-compatible where filter dict, or None if input is None.
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

            # Build a list of individual filter conditions
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
                    # Price range: use $and with $gte and $lte
                    conditions.append({key: {"$gte": min_price}})
                    conditions.append({key: {"$lte": max_price}})
                else:
                    # For other keys, use $in
                    conditions.append({key: {"$in": list(value) if isinstance(value, (list, set)) else [value]}})

            if not conditions:
                return None
            if len(conditions) == 1:
                return conditions[0]
            return {"$and": conditions}
    else:
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


def cell_68(adapted: bool = False):
    """Code cell 68 from the notebook."""
    def generate_filters_from_query(query: str) -> list:
        json_string = generate_metadata_from_query(query)
        json_output = parse_json_output(json_string)
        filters = get_filter_by_metadata(json_output)
        return filters


def cell_69(adapted: bool = False):
    """Code cell 69 from the notebook."""
    filters = generate_filters_from_query("Give me three T-shirts to use in sunny days")


def cell_70(adapted: bool = False):
    """Code cell 70 from the notebook."""
    filters


def cell_72(adapted: bool = False):
    """Code cell 72 from the notebook."""
    if adapted:
        from embedding import embed_query

        def get_relevant_products_from_query(query: str):
            """
            Retrieve products that are most relevant to a given query by applying filters.

            Parameters:
            query (str): The query string used to search for relevant products.

            Returns:
            list: A list of product objects (plain dicts) that are most relevant to the query.
            """
            where = generate_filters_from_query(query)  # Generate Chroma where-filter dict

            if where is None or (isinstance(where, dict) and len(where) == 0):
                # No filters: plain semantic search
                query_emb = embed_query(query)
                return store.query("products", query_emb, top_k=20)

            # Apply filters via Chroma query_with_filter
            query_emb = embed_query(query)
            results = store.query_with_filter(
                "products",
                query_emb,
                top_k=20,
                metadata_filter=where,
            )
            # If fewer than 10 results, try without filters
            if len(results) < 10:
                results = store.query("products", query_emb, top_k=20)
            return results
    else:
        def get_relevant_products_from_query(query: str):
            """
            Retrieve products that are most relevant to a given query by applying filters.

            This function generates filters based on the provided query and uses them to find
            products that closely match the query criteria. If no filters are applicable or if
            the initial search returns a small number of products, the function dynamically reduces
            the filtering constraints based on a predefined order of filter importance.

            Parameters:
            query (str): The query string used to search for relevant products.

            Returns:
            list: A list of product objects that are most relevant to the query. If filters are not effective,
                  it adjusts them to ensure a minimum return of products.
            """
            filters = generate_filters_from_query(query)  # Generate filters based on query

            # Check if there are no applicable filters
            if filters is None or len(filters) == 0:
                # Query the collection without filters, using the query text for relevance
                res = products_collection.query.near_text(query, limit=20).objects
                return res

            # Query with filters and limit to top 20 relevant objects
            res = products_collection.query.near_text(query, filters=Filter.all_of(filters), limit=20).objects

            # If the result set is fewer than 10 products, try reducing filters to broaden the search
            importance_order = ['baseColour', 'masterCategory', 'usage', 'masterCategory', 'season', 'gender']

            if len(res) < 10:
                # Iterate through the importance order of filters
                for i in range(len(importance_order)):
                    # Create a list of filters that excludes less important ones
                    filtered_filters = [x for x in filters if x.target not in importance_order[i+1:]]

                    # Re-query with the reduced set of filters
                    res = products_collection.query.near_text(query, filters=Filter.all_of(filtered_filters), limit=20).objects

                    # If sufficient products have been found, return early
                    if len(res) >= 5:
                        return res

            return res  # Return the final set of relevant products


def cell_73(adapted: bool = False):
    """Code cell 73 from the notebook."""
    query = "Give me three T-shirts to use in sunny days"


def cell_74(adapted: bool = False):
    """Code cell 74 from the notebook."""
    t = get_relevant_products_from_query("Give me three T-shirts to use in sunny days")


def cell_75(adapted: bool = False):
    """Code cell 75 from the notebook."""
    if adapted:
        # In adapted mode, results are plain dicts, not Weaviate objects
        t[0]
    else:
        t[0].properties


def cell_78(adapted: bool = False):
    """Code cell 78 from the notebook."""
    if adapted:
        # Adapted version works on plain dicts (not Weaviate objects with .properties)
        def generate_items_context(results: list) -> str:
            t = ""
            for item in results:
                # item is a plain dict with keys: id, title, chunk, pubDate, link
                # We stored product metadata in these fields
                t += (
                    f"Product ID: {item['id']}. "
                    f"Product name: {item['title']}. "
                    f"Product Category: {item.get('masterCategory', '')}. "
                    f"Product usage: {item.get('usage', '')}. "
                    f"Product gender: {item.get('gender', '')}. "
                    f"Product Type: {item.get('articleType', '')}. "
                    f"Product Category: {item.get('subCategory', '')} "
                    f"Product Color: {item.get('baseColour', '')}. "
                    f"Product Season: {item.get('season', '')}. "
                    f"Product Year: {item.get('year', '')}.\n"
                )
            return t
    else:
        def generate_items_context(results: list) -> str:
            """
            Compile detailed product information from a list of result objects into a formatted string.

            This function takes a list of results, each containing various product attributes, and constructs
            a human-readable summary for each product. Each product's details, including ID, name, category,
            usage, gender, type, and other characteristics, are concatenated into a string that describes
            all products in the list.

            Parameters:
            results (list): A list of result objects, each having a `properties` attribute that is a dictionary
                            containing product attributes such as 'product_id', 'productDisplayName',
                            'masterCategory', 'usage', 'gender', 'articleType', 'subCategory',
                            'baseColour', 'season', and 'year'.

            Returns:
            str: A multi-line string where each line contains the formatted details of a single product.
                 Each product detail includes the product ID, name, category, usage, gender, type, color,
                 season, and year.
            """
            t = ""  # Initialize an empty string to accumulate product information

            for item in results:  # Iterate through each item in the results list
                item = item.properties  # Access the properties dictionary of the current item

                # Append formatted product details to the output string
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

            return t  # Return the complete formatted string with product details


def cell_79(adapted: bool = False):
    """Code cell 79 from the notebook."""
    print(generate_items_context(t)[:1000])


def cell_81(adapted: bool = False):
    """Code cell 81 from the notebook."""
    def query_on_products(query: str) -> dict:
        """
        Execute a product query process to generate a response based on the nature of the query.

        This function analyzes the type of query — whether it is technical or creative — and retrieves
        relevant product information accordingly. It constructs a prompt that includes product details
        and the original query, and then generates parameters for querying an LLM.
        Finally, it generates a response based on the prompt and returns the content of the response.

        Parameters:
        query (str): The input query string that needs to be analyzed and answered using product data.

        Returns:
        dict: A dictionary of keyword arguments (`kwargs`) containing the prompt and additional settings
              for creating a response, suitable for input to an LLM or other processing system.

        Outputs:
        dict: A dictionary with the parameters to call an LLM
        """


        # Determine if the query is technical or creative in nature
        query_label = decide_task_nature(query)

        # Obtain necessary parameters based on the query type
        parameters_dict = get_params_for_task(query_label)

        # Retrieve products that are relevant to the query
        relevant_products = get_relevant_products_from_query(query)

        # Create a context string from the relevant products
        context = generate_items_context(relevant_products)

        # Construct a prompt including product details and the query. Remember to add the context and the query in the prompt, also, ask the LLM to provide the product ID in the answer
        prompt = (
        f"Given the available set of cloth products, answer the question that follows, providing the item ID in your answers. "
        f"Other information might be provided but not necessarily all of them; pick only the relevant ones for the given query and avoid being too long when describing the items' features. "
        f"If no number of products is mentioned in the query, select at most five to show. "
        f"CLOTH PRODUCTS AVAILABLE: {context} "
        f"QUERY: {query}"
            )

        # Generate kwargs (parameters dict) for parameterized input to the LLM with , Prompt, role = 'assistant' and **parameters_dict
        kwargs = generate_params_dict(prompt, role='assistant', **parameters_dict)


        return kwargs


def cell_82(adapted: bool = False):
    """Code cell 82 from the notebook."""
    kwargs = query_on_products('Make a wonderful look for a man attending a wedding party happening during night.')


def cell_83(adapted: bool = False):
    """Code cell 83 from the notebook."""
    result = generate_with_single_input(**kwargs)
    print(result['content'])


def cell_84(adapted: bool = False):
    """Code cell 84 from the notebook."""
    kwargs = query_on_products('Give me three T-shirts for sunny days')


def cell_85(adapted: bool = False):
    """Code cell 85 from the notebook."""
    result = generate_with_single_input(**kwargs)
    print(result['content'])


def cell_87(adapted: bool = False):
    """Code cell 87 from the notebook."""
    def answer_query(query: str) -> dict:
        """
        Determines the type of a given query (FAQ or Product) and executes the appropriate workflow.

        Parameters:
        - query (str): The user's query string.

        Returns:
        - dict: A dictionary of keyword arguments to be used for further processing.
          If the query is neither FAQ nor Product-related, returns a default response dictionary
          instructing the assistant to answer based on existing context.
        """
        label = check_if_faq_or_product(query)
        if label not in ['FAQ', 'Product']:
            return {
                "role": "assistant",
                "prompt": f"User provided a question that does not fit FAQ or Product related questions. "
                          f"Answer it based on the context you already have so far. Query provided by the user: {query}"
            }
        if label == 'FAQ':
            kwargs = query_on_faq(query)
        if label == 'Product':
            try:
                kwargs = query_on_products(query)
            except:
                return {
                "role": "assistant",
                "prompt": f"User provided a question that broke the querying system. Instruct them to rephrase it."
                          f"Answer it based on the context you already have so far. Query provided by the user: {query}"
            }

        return kwargs


def cell_88(adapted: bool = False):
    """Code cell 88 from the notebook."""
    kwargs = answer_query("What are your working hours?")


def cell_89(adapted: bool = False):
    """Code cell 89 from the notebook."""
    result = generate_with_single_input(**kwargs)
    print(result['content'])


def cell_91(adapted: bool = False):
    """Code cell 91 from the notebook."""
    if adapted:
        # ChatWidget from utils (local Ollama-backed) is available
        from utils import ChatWidget
        chat_widget = ChatWidget(generator_function=answer_query)
    else:
        from utils import ChatWidget
        chat_widget = ChatWidget(generator_function=answer_query)