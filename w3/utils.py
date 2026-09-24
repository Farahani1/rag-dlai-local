"""Utility functions for the W3 assignment.

Adapted from the original Coursera version to use local Ollama for LLM calls
instead of Together API / OpenAI proxy, matching the pattern established in
w1/utils.py and w2/utils.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

# Import config from setting.py
PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from setting import config


# =========================
# CONFIG
# =========================
OLLAMA_MODEL_NAME = config.ollama["modelName"]
OLLAMA_URL = config.ollama["url"]
COMPLETION_TEMPERATURE = config.completionOptions.get("temperature")
COMPLETION_MAX_TOKENS = config.completionOptions.get("maxTokens", 500)


# =========================
# DISPLAY (must be importable without triggering heavy deps)
# =========================


def print_object_properties(obj: dict | list) -> None:
    t = ""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        keys.sort()
        for x in keys:
            y = obj[x]
            if x == "article_content":
                t += f"{x}: {y[:100]}...(truncated)\n"
            elif x == "main_vector":
                t += f"{x}: {y[:30]}...(truncated)\n"
            elif x == "chunk":
                t += f"{x}: {y[:100]}...(truncated)\n"
            else:
                t += f"{x}: {y}\n"
    else:
        for l in obj:
            print_object_properties(l)
    print(t)


def print_properties(item: Any) -> None:
    print(json.dumps(item.properties, indent=2, sort_keys=True, default=str))


# =========================
# LOCAL LLM (OLLAMA)
# =========================


def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Send a single prompt to the local Ollama model and return the response."""
    model = model or OLLAMA_MODEL_NAME
    if max_tokens is None:
        max_tokens = COMPLETION_MAX_TOKENS
    if temperature is None and COMPLETION_TEMPERATURE is not None:
        temperature = COMPLETION_TEMPERATURE

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        },
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

    return {
        "role": "assistant",
        "content": result.get("response", ""),
    }


def ollama_llm_backend(prompt: str) -> str:
    """Send a prompt to Ollama and return just the response string.

    Designed to be passed as ``llm_backend`` to ``llm_call`` in
    ``w3/retrieval.py``.  This adapter matches the
    ``Callable[[str], str]`` signature that ``llm_call`` expects.
    """
    result = generate_with_single_input(prompt)
    return result["content"]


def generate_with_multiple_input(
    messages: list[dict[str, str]],
    top_p: float = 1,
    temperature: float = 1,
    max_tokens: int = 500,
    model: str | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Send a chat-format conversation to Ollama and return the response."""
    model = model or OLLAMA_MODEL_NAME
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        },
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

    return {
        "role": "assistant",
        "content": result.get("response", ""),
    }


# =========================
# EMBEDDING (Ollama-based, following W2 pattern)
# =========================


def generate_embedding(prompt: str, model: str = None, **kwargs: Any) -> list[float]:
    """Generate an embedding using the local Ollama embedding endpoint."""
    model = model or OLLAMA_MODEL_NAME
    payload = {
        "model": model,
        "prompt": prompt,
        **kwargs,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("embedding", [])
    except Exception as e:
        raise Exception(f"Ollama embedding call failed: {e}")