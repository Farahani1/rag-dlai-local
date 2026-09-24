"""Utility functions for the W4 assignment.

Adapted from the original Coursera version to use local Ollama for LLM calls
instead of Together API / OpenAI proxy, matching the pattern established in
w3/utils.py.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import markdown
import requests
from IPython.display import display

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
# EMBEDDING (Ollama-based)
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


# =========================
# W4-SPECIFIC UTILITIES
# =========================


def generate_params_dict(
    prompt: str,
    temperature: float = None,
    role: str = "user",
    top_p: float = None,
    max_tokens: int = 500,
    model: str = None,
):
    """
    Call an LLM with different sampling parameters to observe their effects.

    Args:
        prompt: The text prompt to send to the model
        temperature: Controls randomness (lower = more deterministic)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate
        model: The model to use (defaults to OLLAMA_MODEL_NAME)

    Returns:
        The LLM response
    """

    if model is None:
        model = OLLAMA_MODEL_NAME

    # Create the dictionary with the necessary parameters
    kwargs = {
        "prompt": prompt,
        "role": role,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "model": model,
    }

    return kwargs


def call_llm_with_context(
    prompt: str, context: list, role: str = "user", **kwargs
):
    """
    Calls a language model with the given prompt and context to generate a response.
    Parameters:
    - prompt (str): The input text prompt provided by the user.
    - role (str): The role of the participant in the conversation, e.g., "user" or "assistant".
    - context (list): A list representing the conversation history, to which the new input is added.
    - **kwargs: Additional keyword arguments for configuring the language model call (e.g., top_k, temperature).
    Returns:
    - response (str): The generated response from the language model based on the provided prompt and context.
    """
    context.append({"role": role, "content": prompt})
    response = generate_with_multiple_input(context, **kwargs)
    context.append(response)
    return response


class ChatBot:
    """
    A simple chatbot class for handling user interactions using an LLM.

    This class maintains a conversation context and interfaces with a language model API
    to generate responses related to a clothing store.
    """

    def __init__(
        self,
        generator_function,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        context_window: int = 20,
    ):
        # Initialize system and greeting messages
        self.system_prompt = {
            "role": "system",
            "content": "You are a friendly assistant from Fashion Forward Hub. It is a cloth store selling a variety of items. Your job is to answer questions related to FAQ or Products.",
        }
        self.initial_message = {
            "role": "assistant",
            "content": "Hi! How can I help you?",
        }
        self.generator_function = generator_function
        # Initialize conversation with system and assistant message
        self.conversation: list[dict[str, str]] = [
            self.system_prompt,
            self.initial_message,
        ]
        self.context_window = context_window  # Limit of past messages to consider
        self.model = model  # Model name to use for inference

    def chat(self, prompt: str, role: str = "user") -> dict[str, str]:
        """
        Handles a single round of user interaction and updates the conversation context.
        """
        recent_context = self.conversation[
            -self.context_window :
        ]  # Get recent messages
        params_dict = self.generator_function(prompt)
        params_dict["model"] = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        content = call_llm_with_context(
            context=recent_context, **params_dict
        )  # Get response from model
        self.conversation.append(
            {"role": "user", "content": prompt}
        )  # Append user message
        self.conversation.append(
            recent_context[-1]
        )  # Append latest assistant message
        return content

    def start_conversation(self) -> None:
        """
        Starts a terminal-based conversation loop with the user.
        """
        print(self.format_message(self.initial_message))
        while True:
            prompt = input("You: ")
            if prompt.lower() == "end conversation":
                break
            print(f"User: {prompt}")
            response = self.chat(prompt)
            print(self.format_message(response))

    def clear_conversation(self) -> None:
        """
        Resets the conversation to its initial state.
        """
        self.conversation = [self.system_prompt, self.initial_message]


class ChatWidget:
    """
    A widget-based UI for interacting with the ChatBot using ipywidgets.

    Displays messages, handles user input, and dynamically loads related images.
    """

    def __init__(self, generator_function):
        self.chat_bot = ChatBot(generator_function)  # Initialize chatbot
        self.output_area = widgets.HTML()  # Area to show conversation text
        self.image_area = widgets.HBox()  # Area to display images
        self.text_input = widgets.Text(
            placeholder="Type your message...", layout=widgets.Layout(width="90%")
        )
        self.send_button = widgets.Button(
            description="Send", layout=widgets.Layout(width="10%")
        )
        # Attach click event to send button
        self.send_button.on_click(self.send_message)
        self.unique_ids = set()  # Track which image IDs were already shown
        self.display()  # Display the UI
        self.refresh_messages()  # Show initial messages

    def send_message(self, _):
        """
        Handles sending a user message from the text input field.
        """
        user_message = self.text_input.value
        if user_message.strip() == "":
            return
        self.display_user_message(user_message)  # Show user message
        self.show_thinking()  # Show 'thinking...' message
        self.text_input.value = ""  # Clear input box
        self.image_area.children = ()  # Reset image display
        # Handle bot response in a separate thread
        threading.Thread(
            target=self.process_bot_response, args=(user_message,)
        ).start()

    def process_bot_response(self, user_message):
        """
        Gets the response from the bot and updates the UI accordingly.
        """
        response = self.chat_bot.chat(user_message)
        response_content = response["content"]
        self.extract_and_process_ids(response_content)  # Process any mentioned IDs
        self.refresh_messages()  # Refresh message display

    def extract_and_process_ids(self, message: str):
        """
        Finds and processes any 'ID: <number>' entries in the message to load images.
        """
        pattern = re.compile(r"ID:\s*(\d+(?:,\s*\d+)*)", re.IGNORECASE)
        matches = pattern.findall(message)
        found_ids = [id.strip() for match in matches for id in match.split(",")]
        for id in found_ids:
            if id not in self.unique_ids:
                self.unique_ids.add(id)
                self.load_image(id)

    def load_image(self, id: str):
        """
        Loads an image associated with the given ID and displays it in the image area.
        """
        image_path = (
            f"/home/jovyan/data/collections/collections_assignment_4/images/{id}.jpg"
        )
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_data = f.read()
            img_widget = widgets.Image(
                value=img_data,
                format="jpg",
                layout=widgets.Layout(width="150px", height="auto", margin="5px"),
            )
            id_label = widgets.Label(
                value=f"ID: {id}", layout=widgets.Layout(width="150px")
            )
            vbox = widgets.VBox([img_widget, id_label])
            self.image_area.children += (vbox,)

    def display_user_message(self, message: str):
        """
        Displays the user's message in the output HTML area.
        """
        escaped_message = markdown.markdown(message)
        html_content = self.output_area.value
        html_content += (
            f"<div style='background-color: #f1f1f1; padding: 10px; margin: 5px 0; "
            f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>User:</strong>"
            f"<div style='margin: 0; white-space: normal; color: black !important;'>{escaped_message}</div></div>"
        )
        self.output_area.value = html_content

    def show_thinking(self):
        """
        Temporarily shows a 'Thinking...' placeholder from the assistant.
        """
        html_content = self.output_area.value
        html_content += (
            f"<div style='background-color: #fff3cd; padding: 10px; margin: 5px 0; "
            f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>Assistant:</strong>"
            f"<div style='margin: 0; white-space: normal; color: black !important;'>Thinking...</div></div>"
        )
        self.output_area.value = html_content

    def refresh_messages(self):
        """
        Re-renders the entire conversation history in the output area.
        """
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
                formatted_message = message["content"]
                escaped_content = markdown.markdown(formatted_message)
                html_content += (
                    f"<div style='background-color: #e2f7d5; padding: 10px; margin: 5px 0; "
                    f"border-radius: 8px; max-width: 90%; color: black !important;'><strong>Assistant:</strong>"
                    f"<div style='margin: 0; white-space: normal; color: black !important;'>{escaped_content}</div></div>"
                )
        html_content += "</div>"
        self.output_area.value = html_content

    def display(self):
        """
        Sets up and displays the chat widget user interface with images aligned at the bottom right.
        """
        # Set up the input area with text input and send button.
        input_area = widgets.HBox([self.text_input, self.send_button])
        chat_ui = widgets.VBox(
            [self.output_area, self.image_area, input_area],
            layout=widgets.Layout(margin="10px"),
        )
        display(chat_ui)