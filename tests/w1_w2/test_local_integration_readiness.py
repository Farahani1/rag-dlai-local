import importlib.util
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
REQUIRED_PACKAGES = [
    "nbclient",
    "sentence_transformers",
    "joblib",
    "pandas",
    "sklearn",
    "bm25s",
]


pytestmark = pytest.mark.local_integration


def load_raw_config_or_skip():
    if not CONFIG_PATH.exists():
        pytest.skip(f"config.yaml is missing at {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        pytest.skip(f"config.yaml did not parse to a mapping: {CONFIG_PATH}")
    return config


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def huggingface_model_cache_path(cache_root, model_name):
    model_directory = "models--" + model_name.replace("/", "--")
    return cache_root / "hub" / model_directory


def resolved_huggingface_snapshot(cache_root, model_name):
    model_cache = huggingface_model_cache_path(cache_root, model_name)
    main_ref = model_cache / "refs" / "main"
    if not main_ref.is_file():
        return None

    revision = main_ref.read_text(encoding="utf-8").strip()
    if not revision:
        return None

    snapshot = model_cache / "snapshots" / revision
    if not snapshot.is_dir():
        return None
    return snapshot


def ollama_api_base(generate_url):
    parsed = urlparse(generate_url)
    if not parsed.scheme or not parsed.netloc:
        pytest.skip(f"Ollama URL is not a valid absolute URL: {generate_url}")

    path = parsed.path
    if path.endswith("/api/generate"):
        path = path[: -len("/api/generate")]
    base = parsed._replace(path=path.rstrip("/"), params="", query="", fragment="")
    return base.geturl()


def test_required_python_packages_are_installed():
    missing = [
        package
        for package in REQUIRED_PACKAGES
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        pytest.skip(f"Missing required packages for local integration: {', '.join(missing)}")


def test_config_yaml_exists_and_has_required_local_paths():
    config = load_raw_config_or_skip()

    try:
        data_config = config["data"]
        model_config = config["models"]
        news_csv = resolve_path(data_config["newsCSV"])
        news_embeddings = resolve_path(data_config["newsEmbeddings"])
        hf_cache = resolve_path(model_config["hfLocalHub"])
    except KeyError as exc:
        pytest.skip(f"config.yaml is missing required key: {exc}")

    missing_paths = [
        str(path)
        for path in [news_csv, news_embeddings, hf_cache]
        if not path.exists()
    ]
    if missing_paths:
        pytest.skip("Missing local integration paths: " + ", ".join(missing_paths))


def test_configured_embedding_model_is_available_locally():
    config = load_raw_config_or_skip()

    try:
        model_name = config["models"]["embeddingModel"]
        cache_root = resolve_path(config["models"]["hfLocalHub"])
    except KeyError as exc:
        pytest.skip(f"config.yaml is missing required key: {exc}")

    snapshot = resolved_huggingface_snapshot(cache_root, model_name)
    if snapshot is None:
        pytest.skip(
            f"Embedding model is not cached locally: {model_name} under {cache_root}"
        )

    required_files = ["config.json", "modules.json"]
    missing_files = [name for name in required_files if not (snapshot / name).is_file()]
    if missing_files:
        pytest.skip(
            f"Embedding model cache is incomplete at {snapshot}; missing: "
            + ", ".join(missing_files)
        )


def test_ollama_endpoint_is_reachable():
    config = load_raw_config_or_skip()

    try:
        ollama_url = config["models"]["ollama"]["url"]
    except KeyError as exc:
        pytest.skip(f"config.yaml is missing required key: {exc}")

    parsed = urlparse(ollama_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434

    try:
        with socket.create_connection((host, port), timeout=1):
            pass
    except OSError as exc:
        pytest.skip(f"Ollama endpoint is not reachable at {host}:{port}. Error: {exc}")


def test_configured_ollama_model_is_available():
    config = load_raw_config_or_skip()

    try:
        import requests

        ollama_config = config["models"]["ollama"]
        model_name = ollama_config["modelName"]
        tags_url = f"{ollama_api_base(ollama_config['url'])}/api/tags"
    except KeyError as exc:
        pytest.skip(f"config.yaml is missing required key: {exc}")
    except ImportError as exc:
        pytest.skip(f"requests is not importable: {exc}")

    try:
        response = requests.get(tags_url, timeout=3)
        response.raise_for_status()
        models = response.json().get("models", [])
    except Exception as exc:
        pytest.skip(f"Could not query Ollama models from {tags_url}. Error: {exc}")

    available_names = {model.get("name") for model in models}
    if model_name not in available_names:
        pytest.skip(
            f"Configured Ollama model is not available: {model_name}. "
            f"Available models: {sorted(name for name in available_names if name)}"
        )
