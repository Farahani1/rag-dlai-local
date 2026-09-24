"""Utility functions for the W5 assignment.

Adapted from the original Coursera version to use
local Ollama for LLM calls instead of Together API / OpenAI proxy, and a
local no-op tracer instead of Arize Phoenix / OpenTelemetry (no such packages
are installed in this project — see ``requirements.txt``).

Unlike ``w4/utils.py``, ``generate_with_single_input``/``generate_with_multiple_input``
here return an OpenAI/Together-compatible response shape (``choices``/``usage``)
because W5's whole point is measuring LLM token usage, and the notebook code
reads ``response['choices'][0]['message']['content']`` and
``response['usage']['total_tokens']`` throughout.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import markdown
import requests
from IPython.display import display

from setting import config


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================
# CONFIG
# =========================
OLLAMA_MODEL_NAME = config.ollama["modelName"]
OLLAMA_URL = config.ollama["url"]
COMPLETION_TEMPERATURE = config.completionOptions.get("temperature")
COMPLETION_MAX_TOKENS = config.completionOptions.get("maxTokens", 500)


# =========================
# TRACING (local no-op stand-in — no arize-phoenix/opentelemetry dependency)
# =========================


class StatusCode:
    """Stand-in for ``opentelemetry.trace.StatusCode``."""

    OK = "OK"
    ERROR = "ERROR"


class Status:
    """Stand-in for ``opentelemetry.trace.Status``."""

    def __init__(self, status_code: str, description: str | None = None):
        self.status_code = status_code
        self.description = description


class NullSpan:
    """No-op stand-in for a Phoenix/OpenTelemetry span context manager."""

    def __enter__(self) -> "NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def set_input(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_output(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass


class NullTracer:
    """Local no-op stand-in for a Phoenix/OpenTelemetry tracer.

    Avoids adding ``arize-phoenix``/``opentelemetry`` as project dependencies
    while preserving every ``@tracer.tool`` / ``tracer.start_as_current_span(...)``
    call site from the original notebook unchanged — only the object backing
    ``tracer`` differs between adapted and non-adapted mode.
    """

    def tool(self, func):
        return func

    def start_as_current_span(self, name: str, **kwargs: Any) -> NullSpan:
        return NullSpan()


# =========================
# LOCAL LLM (OLLAMA) — OpenAI/Together-compatible response shape
# =========================


def _count_words(text: str) -> int:
    return len(text.split()) if text else 0


def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Send a single prompt to the local Ollama model.

    Returns an OpenAI/Together-compatible response dict (``choices``/``usage``),
    with token counts sourced from Ollama's ``prompt_eval_count``/``eval_count``
    fields (present in non-streaming ``/api/generate`` responses). Falls back
    to a whitespace word count if either field is missing, so this never
    raises purely due to token accounting.
    """
    model = model or OLLAMA_MODEL_NAME
    if max_tokens is None:
        max_tokens = COMPLETION_MAX_TOKENS
    if temperature is None and COMPLETION_TEMPERATURE is not None:
        temperature = COMPLETION_TEMPERATURE

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature
    if top_p is not None:
        payload["options"]["top_p"] = top_p

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        raise Exception(f"Ollama call failed: {e}")

    return _to_openai_shape(result, model, prompt)


def generate_with_multiple_input(
    messages: list[dict[str, str]],
    top_p: float = 1,
    temperature: float = 1,
    max_tokens: int = 500,
    model: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Send a chat-format conversation to Ollama; same response shape as above.

    Ollama's ``/api/generate`` endpoint is single-prompt, so the conversation
    is flattened into one prompt string before sending.
    """
    model = model or OLLAMA_MODEL_NAME
    prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature
    if top_p is not None:
        payload["options"]["top_p"] = top_p

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        raise Exception(f"Ollama call failed: {e}")

    return _to_openai_shape(result, model, prompt)


def _to_openai_shape(result: dict[str, Any], model: str, prompt: str) -> dict[str, Any]:
    content = result.get("response", "")
    prompt_tokens = result.get("prompt_eval_count")
    completion_tokens = result.get("eval_count")
    if prompt_tokens is None:
        prompt_tokens = _count_words(prompt)
    if completion_tokens is None:
        completion_tokens = _count_words(content)

    return {
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def generate_params_dict(
    prompt: str,
    temperature: float = None,
    role: str = "user",
    top_p: float = None,
    max_tokens: int = 500,
    model: str = None,
):
    """
    Build a dictionary of parameters for calling the local LLM.

    Args:
        prompt: The text prompt to send to the model
        temperature: Controls randomness (lower = more deterministic)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate
        model: The model to use (defaults to OLLAMA_MODEL_NAME)

    Returns:
        A dict suitable for ``**kwargs`` into ``generate_with_single_input``.
    """
    if model is None:
        model = OLLAMA_MODEL_NAME

    kwargs = {
        "prompt": prompt,
        "role": role,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "model": model,
    }

    return kwargs


def call_llm_with_context(prompt: str, context: list, role: str = "user", **kwargs):
    """
    Calls the LLM with the given prompt appended to a running conversation.

    Parameters:
    - prompt (str): The input text prompt provided by the user.
    - context (list): Conversation history, mutated in place with the new turn.
    - role (str): The role of the participant, e.g. "user" or "assistant".
    - **kwargs: Additional keyword arguments (temperature, top_p, etc.).

    Returns:
    - dict: The raw OpenAI-shaped response from the LLM.
    """
    context.append({"role": role, "content": prompt})
    response = generate_with_multiple_input(context, **kwargs)
    return response


# =========================
# JSON / FILTER HELPERS (used by cell 91's redefinitions too)
# =========================


def parse_json_output(llm_output: str) -> dict | None:
    """
    Parses a string output from an LLM into a JSON object.

    Cleans minor formatting issues (stray newlines, single quotes, doubled
    braces) before parsing. Returns ``None`` on failure.
    """
    try:
        llm_output = (
            llm_output.replace("\n", "").replace("'", "").replace("}}", "}").replace("{{", "{")
        )
        parsed_json = json.loads(llm_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None


VALID_FILTER_KEYS = (
    "gender",
    "masterCategory",
    "articleType",
    "baseColour",
    "price",
    "usage",
    "season",
)


def get_filter_by_metadata(json_output: dict | None = None) -> list[dict] | None:
    """Build a list of Chroma single-key filter conditions from LLM-extracted metadata.

    Returns a **list** of conditions (e.g. ``[{"baseColour": {"$in": [...]}}]``)
    rather than a merged ``$and`` dict, so ``get_relevant_products_from_query``
    can progressively drop the least-important conditions by key — mirroring
    the original Weaviate ``Filter`` list (dropped/kept via ``.target``) that
    this replaces. Use :func:`filters_to_where` to merge the list into a
    single Chroma ``where`` clause right before querying.
    """
    if json_output is None:
        return None

    conditions: list[dict] = []
    for key, value in json_output.items():
        if key not in VALID_FILTER_KEYS:
            continue

        if key == "price":
            if not isinstance(value, dict):
                continue
            min_price = value.get("min")
            max_price = value.get("max")
            if min_price is None or max_price is None:
                continue
            if min_price <= 0 or max_price == "inf":
                continue
            conditions.append({key: {"$gte": min_price}})
            conditions.append({key: {"$lte": max_price}})
        else:
            conditions.append(
                {key: {"$in": list(value) if isinstance(value, (list, set)) else [value]}}
            )

    return conditions if conditions else None


def _condition_key(condition: dict) -> str:
    """Return the single metadata key a Chroma filter condition targets."""
    return next(iter(condition))


def filters_to_where(filters: list[dict] | None) -> dict | None:
    """Merge a list of single-key filter conditions into one Chroma ``where`` clause."""
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _load_products_data():
    import joblib

    return joblib.load(str(config.clothesData))


def _build_values_dict(products_data) -> dict[str, set]:
    values: dict[str, set] = {}
    for d in products_data:
        for key, val in d.items():
            if key in ("product_id", "price", "productDisplayName", "subCategory", "year"):
                continue
            values.setdefault(key, set()).add(val)
    return values


def generate_metadata_from_query(query: str, values: dict | None = None) -> tuple[str, int]:
    """Generate JSON metadata (for Chroma filtering) from a natural-language query.

    Standalone version used by :func:`generate_filters_from_query` below, so
    ``utils.py`` is importable and usable on its own. The notebook additionally
    defines its own ``generate_metadata_from_query`` in cell 82 (with tracing,
    reusing the notebook's already-loaded ``values``); that copy shadows this
    one for the rest of the notebook, matching the layering of the original
    assignment (cell 91 redefines ``generate_filters_from_query`` too, and at
    that point it resolves ``generate_metadata_from_query`` from the notebook's
    own globals, not this module).
    """
    if values is None:
        values = _build_values_dict(_load_products_data())

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
    kwargs = generate_params_dict(PROMPT, temperature=0, max_tokens=1500)
    response = generate_with_single_input(**kwargs)
    content = response["choices"][0]["message"]["content"]
    total_tokens = response["usage"]["total_tokens"]
    return content, total_tokens


def generate_filters_from_query(query: str) -> tuple[list[dict] | None, int]:
    json_string, total_tokens = generate_metadata_from_query(query)
    json_output = parse_json_output(json_string)
    filters = get_filter_by_metadata(json_output)
    return filters, total_tokens


# =========================
# DISPLAY / FORMATTING
# =========================


def print_properties(item: Any) -> None:
    """Adapted: Chroma results are already plain dicts (no ``.properties``)."""
    print(json.dumps(item, indent=2, sort_keys=True, default=str))


def process_and_print_query(query, correct_label, response_std, tokens_std, response_simp, tokens_simp):
    """Pretty-print a standard-vs-simplified comparison for one query. Pure formatting."""
    max_tokens = 180
    label_std_colored = (
        "\033[32m" + response_std + "\033[0m"
        if response_std == correct_label
        else "\033[31m" + response_std + "\033[0m"
    )
    tokens_std_colored = (
        "\033[32m" + str(tokens_std) + "\033[0m"
        if tokens_std <= 130
        else "\033[31m" + str(tokens_std) + "\033[0m"
    )
    label_simp_colored = (
        "\033[32m" + response_simp + "\033[0m"
        if response_simp == correct_label
        else "\033[31m" + response_simp + "\033[0m"
    )
    tokens_simp_colored = (
        "\033[32m" + str(tokens_simp) + "\033[0m"
        if tokens_simp <= max_tokens
        else "\033[31m" + str(tokens_simp) + "\033[0m"
    )
    print(f"Query: {query}")
    print(f"  Standard    -> Label: {label_std_colored} | Tokens: {tokens_std_colored}")
    print(f"  Simplified  -> Label: {label_simp_colored} | Tokens: {tokens_simp_colored}\n")


def make_url(endpoint: str | None = None) -> None:
    """Adapted: no Coursera lab / local UI server backs this notebook.

    Prints an informational message instead of a URL that would not resolve
    to anything running locally.
    """
    BOLD = "\033[1m"
    RESET = "\033[0m"
    label = endpoint or "the chat UI"
    print(
        f"{BOLD}[adapted mode] No local UI server is running for {label}. "
        f"Use ChatWidget directly in this notebook instead.{RESET}"
    )


# =========================
# CHAT WIDGET
# =========================


class ChatBot:
    """
    A simple chatbot class for handling user interactions using the local LLM.

    Maintains a conversation context and expects ``generator_function`` to
    return ``(params_dict, total_tokens)`` for a given prompt, matching
    ``answer_query``'s signature.
    """

    def __init__(
        self,
        generator_function,
        tracer: "NullTracer | None" = None,
        model: str | None = None,
        context_window: int = 20,
    ):
        self.system_prompt = {
            "role": "system",
            "content": "You are a friendly assistant from Fashion Forward Hub. It is a cloth store selling a variety of items. Your job is to answer questions related to FAQ or Products.",
        }
        self.initial_message = {
            "role": "assistant",
            "content": "Hi! How can I help you?",
        }
        self.generator_function = generator_function
        self.tracer = tracer if tracer is not None else NullTracer()
        self.conversation: list[dict[str, str]] = [self.system_prompt, self.initial_message]
        self.context_window = context_window
        self.model = model or OLLAMA_MODEL_NAME
        self.kwargs_list: list[dict[str, Any]] = []

    def chat(self, prompt: str, role: str = "user", return_stats: bool = False):
        """
        Handles a single round of user interaction and updates the conversation context.
        """
        with self.tracer.start_as_current_span("agent_call", openinference_span_kind="agent") as span:
            span.set_input({"prompt": prompt, "role": role})
            start_time = time.time()
            recent_context = self.conversation[-self.context_window :]
            span.set_attribute("agent.recent_context", str(recent_context))
            params_dict, total_tokens = self.generator_function(prompt)
            self.kwargs_list.append(params_dict)
            with self.tracer.start_as_current_span("llm_call", openinference_span_kind="llm") as llm_span:
                try:
                    response = call_llm_with_context(context=recent_context, **params_dict)
                    llm_span.set_input({"messages": recent_context, **params_dict})
                    content = response["choices"][0]["message"]["content"]
                    total_tokens = response["usage"]["total_tokens"]
                except Exception as error:
                    llm_span.record_exception(error)
                    llm_span.set_status(Status(StatusCode.ERROR))
                    raise
                else:
                    llm_span.set_attribute("llm.token_count.prompt", response["usage"]["prompt_tokens"])
                    llm_span.set_attribute("llm.token_count.completion", response["usage"]["completion_tokens"])
                    llm_span.set_attribute("llm.token_count.total", response["usage"]["total_tokens"])
                    llm_span.set_attribute("llm.model_name", response["model"])
                    llm_span.set_attribute("llm.provider", "ollama")
                    llm_span.set_output(response)
                    llm_span.set_status(Status(StatusCode.OK))

                self.conversation.append({"role": "user", "content": prompt})
                self.conversation.append({"role": "assistant", "content": content})
                end_time = time.time()
                total_time = end_time - start_time
                if return_stats:
                    return content, total_tokens, total_time
                span.set_output({"content": content, "total_tokens": total_tokens})
                span.set_status(Status(StatusCode.OK))
                return {"content": content, "role": "assistant"}

    def start_conversation(self) -> None:
        print(self.initial_message["content"])
        while True:
            prompt = input("You: ")
            if prompt.lower() == "end conversation":
                break
            print(f"User: {prompt}")
            response = self.chat(prompt)
            print(response["content"])

    def clear_conversation(self) -> None:
        self.conversation = [self.system_prompt, self.initial_message]


class ChatWidget:
    """
    A widget-based UI for interacting with the ChatBot using ipywidgets.
    """

    def __init__(self, generator_function, tracer: "NullTracer | None" = None):
        self.chat_bot = ChatBot(generator_function, tracer=tracer)
        self.output_area = widgets.HTML()
        self.image_area = widgets.HBox()
        self.text_input = widgets.Text(
            placeholder="Type your message...", layout=widgets.Layout(width="90%")
        )
        self.send_button = widgets.Button(description="Send", layout=widgets.Layout(width="10%"))
        self.send_button.on_click(self.send_message)
        self.unique_ids: set[str] = set()
        self.display()
        self.refresh_messages()

    def send_message(self, _):
        user_message = self.text_input.value
        if user_message.strip() == "":
            return
        self.display_user_message(user_message)
        self.show_thinking()
        self.text_input.value = ""
        self.image_area.children = ()
        threading.Thread(target=self.process_bot_response, args=(user_message,)).start()

    def process_bot_response(self, user_message):
        response = self.chat_bot.chat(user_message)
        response_content = response["content"]
        self.extract_and_process_ids(response_content)
        self.refresh_messages()

    def extract_and_process_ids(self, message: str):
        pattern = re.compile(r"ID:\s*(\d+(?:,\s*\d+)*)", re.IGNORECASE)
        matches = pattern.findall(message)
        found_ids = [id.strip() for match in matches for id in match.split(",")]
        for id in found_ids:
            if id not in self.unique_ids:
                self.unique_ids.add(id)
                self.load_image(id)

    def load_image(self, id: str):
        """Adapted: image assets (if any) live under ``data/``, not a Coursera lab path."""
        image_path = _PROJECT_ROOT / "data" / "product_images" / f"{id}.jpg"
        if image_path.exists():
            with open(image_path, "rb") as f:
                img_data = f.read()
            img_widget = widgets.Image(
                value=img_data,
                format="jpg",
                layout=widgets.Layout(width="150px", height="auto", margin="5px"),
            )
            id_label = widgets.Label(value=f"ID: {id}", layout=widgets.Layout(width="150px"))
            vbox = widgets.VBox([img_widget, id_label])
            self.image_area.children += (vbox,)

    def display_user_message(self, message: str):
        escaped_message = markdown.markdown(message)
        html_content = self.output_area.value
        html_content += (
            f"<div style='background-color: #f1f1f1; padding: 10px; margin: 5px 0; "
            f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>User:</strong>"
            f"<div style='margin: 0; white-space: normal; color: black !important;'>{escaped_message}</div></div>"
        )
        self.output_area.value = html_content

    def show_thinking(self):
        html_content = self.output_area.value
        html_content += (
            f"<div style='background-color: #fff3cd; padding: 10px; margin: 5px 0; "
            f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>Assistant:</strong>"
            f"<div style='margin: 0; white-space: normal; color: black !important;'>Thinking...</div></div>"
        )
        self.output_area.value = html_content

    def refresh_messages(self):
        html_content = "<div style='font-family: Arial; max-width: 600px;'>"
        for message in self.chat_bot.conversation:
            if message["role"] == "user":
                escaped_content = markdown.markdown(message["content"])
                html_content += (
                    f"<div style='background-color: #f1f1f1; padding: 10px; margin: 5px 0; "
                    f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>User:</strong>"
                    f"<div style='margin: 0; white-space: normal; color: black !important;'>{escaped_content}</div></div>"
                )
            elif message["role"] == "assistant":
                escaped_content = markdown.markdown(message["content"])
                html_content += (
                    f"<div style='background-color: #e2f7d5; padding: 10px; margin: 5px 0; "
                    f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>Assistant:</strong>"
                    f"<div style='margin: 0; white-space: normal; color: black !important;'>{escaped_content}</div></div>"
                )
        html_content += "</div>"
        self.output_area.value = html_content

    def display(self):
        input_area = widgets.HBox([self.text_input, self.send_button])
        chat_ui = widgets.VBox(
            [self.output_area, self.image_area, input_area],
            layout=widgets.Layout(margin="10px"),
        )
        display(chat_ui)
