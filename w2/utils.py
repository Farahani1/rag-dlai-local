import json
import numpy as np
import pandas as pd
from dateutil import parser
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
import sys
from pathlib import Path

# Import config from setting.py (assumes same directory or parent in sys.path)
PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from setting import config


# =========================
# CONFIG
# =========================
# Set HuggingFace cache directory from config
os.environ["HF_HOME"] = str(config.hfLocalHub)

# Read model and API settings from config
EMBEDDING_MODEL_NAME = config.embeddingModel
OLLAMA_MODEL_NAME = config.ollama["modelName"]
OLLAMA_URL = config.ollama["url"]

# Embeddings file path
EMBEDDINGS_PATH = config.newsEmbeddings  # Path object

# Completion options (with fallback defaults)
COMPLETION_TEMPERATURE = config.completionOptions.get("temperature")
COMPLETION_MAX_TOKENS = config.completionOptions.get("maxTokens", 500)


# =========================
# UTILS
# =========================
def pprint(*args, **kwargs):
    print(json.dumps(*args, indent=2))


def format_date(date_string):
    date_object = parser.parse(date_string)
    return date_object.strftime("%Y-%m-%d")


def read_dataframe(path):
    df = pd.read_csv(path)
    df['published_at'] = df['published_at'].apply(format_date)
    df['updated_at'] = df['updated_at'].apply(format_date)
    return df.to_dict(orient='records')


def concatenate_fields(dataset, fields):
    concatenated_data = []

    for data in dataset:
        text = ""
        for field in fields:
            context = data.get(field, '')
            if context:
                text += f"{context} "

        text = text.strip()[:493]
        concatenated_data.append(text)

    return concatenated_data


# =========================
# RETRIEVAL
# =========================
def retrieve(query, top_k=5):
    # Load embedding model (uses config for model name)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

    # Load precomputed embeddings from config-defined path
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    EMBEDDINGS = joblib.load(str(EMBEDDINGS_PATH))

    query_embedding = embedding_model.encode(query)

    similarity_scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        EMBEDDINGS
    )[0]

    top_k_indices = np.argsort(-similarity_scores)[:top_k]
    return top_k_indices


def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)


# =========================
# LOCAL LLM (OLLAMA)
# =========================
def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float = None,
    temperature: float = None,
    max_tokens: int = None,
    model: str = None,
    **kwargs
):
    # Use config defaults if not explicitly provided
    model = model or OLLAMA_MODEL_NAME
    if max_tokens is None:
        max_tokens = COMPLETION_MAX_TOKENS
    if temperature is None and COMPLETION_TEMPERATURE is not None:
        temperature = COMPLETION_TEMPERATURE

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        }
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
        "content": result.get("response", "")
    }


# =========================
# LOAD DATA
# =========================
# NEWS_DATA can be loaded similarly using config.newsCSV if needed
# NEWS_DATA = pd.read_csv(str(config.newsCSV)).to_dict(orient='records')


# =========================
# UI (UNCHANGED)
# =========================
import ipywidgets as widgets
from IPython.display import display, Markdown

def display_widget(llm_call_func):
    def on_button_click(b):
        output1.clear_output()
        output2.clear_output()
        status_output.clear_output()

        status_output.append_stdout("Generating...\n")

        query = query_input.value
        top_k = slider.value
        prompt = prompt_input.value.strip() if prompt_input.value.strip() else None

        response1 = llm_call_func(query, use_rag=True, top_k=top_k, prompt=prompt)
        response2 = llm_call_func(query, use_rag=False, top_k=top_k, prompt=prompt)

        with output1:
            display(Markdown(response1))

        with output2:
            display(Markdown(response2))

        status_output.clear_output()

    query_input = widgets.Text(
        description='Query:',
        placeholder='Type your query here',
        layout=widgets.Layout(width='100%')
    )

    prompt_input = widgets.Textarea(
        description='Augmented prompt layout:',
        placeholder=("Use {query} and {documents}"),
        layout=widgets.Layout(width='100%', height='100px'),
        style={'description_width': 'initial'}
    )

    slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description='Top K:',
        style={'description_width': 'initial'}
    )

    output1 = widgets.Output(layout={'border': '1px solid #ccc', 'width': '45%'})
    output2 = widgets.Output(layout={'border': '1px solid #ccc', 'width': '45%'})
    status_output = widgets.Output()

    submit_button = widgets.Button(description="Get Responses")
    submit_button.on_click(on_button_click)

    display(query_input, prompt_input, slider, submit_button, status_output)

    display(widgets.HBox([
        widgets.Label("With RAG"),
        widgets.Label("Without RAG")
    ]))

    display(widgets.HBox([output1, output2]))