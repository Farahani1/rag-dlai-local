import importlib.util
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

pytestmark = pytest.mark.local_integration


def load_config_or_skip():
    if not CONFIG_PATH.is_file():
        pytest.skip(f"config.yaml is missing at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def require_ollama_or_skip(config):
    url = config["models"]["ollama"]["url"]
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=1):
            pass
    except OSError as exc:
        pytest.skip(f"Ollama is not reachable at {host}:{port}: {exc}")


def import_real_utils(stage):
    path = PROJECT_ROOT / stage / "utils.py"
    spec = importlib.util.spec_from_file_location(f"real_{stage}_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real_config():
    config = load_config_or_skip()
    require_ollama_or_skip(config)
    return config


@pytest.fixture(scope="module")
def w1_utils(real_config):
    return import_real_utils("w1")


@pytest.fixture(scope="module")
def w2_utils(real_config):
    return import_real_utils("w2")


@pytest.fixture(scope="module")
def w2_embedding_model(w2_utils):
    return w2_utils.get_embedding_model()


def assert_valid_indices(indices, expected_count, row_count):
    values = [int(index) for index in indices]
    assert len(values) == expected_count
    assert len(set(values)) == expected_count
    assert all(0 <= index < row_count for index in values)


def test_w1_real_data_and_retrieval(w1_utils):
    assert w1_utils.NEWS_DATA
    assert w1_utils.EMBEDDINGS is not None

    indices = w1_utils.retrieve("economic growth and GDP", top_k=2)

    assert_valid_indices(indices, expected_count=2, row_count=len(w1_utils.NEWS_DATA))


def test_w2_real_embedding_model_and_retrieval(
    w2_utils, w2_embedding_model, monkeypatch
):
    embeddings = w2_utils.joblib.load(str(w2_utils.EMBEDDINGS_PATH))
    monkeypatch.setattr(
        w2_utils,
        "SentenceTransformer",
        lambda *args, **kwargs: w2_embedding_model,
    )

    indices = w2_utils.retrieve("economic growth and GDP", top_k=2)

    assert w2_embedding_model.encode("embedding smoke check").shape[-1] == embeddings.shape[1]
    assert_valid_indices(indices, expected_count=2, row_count=len(embeddings))


@pytest.mark.parametrize("stage_fixture", ["w1_utils", "w2_utils"])
def test_real_short_ollama_generation(stage_fixture, request, real_config):
    module = request.getfixturevalue(stage_fixture)
    response = module.generate_with_single_input(
        "Say hello in one short sentence.",
        model=real_config["models"]["ollama"]["modelName"],
        temperature=0,
        max_tokens=64,
    )

    assert response["role"] == "assistant"
    assert response["content"].strip()
