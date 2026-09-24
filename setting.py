from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict
import subprocess
import time
import socket
from urllib.parse import urlparse
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class AppConfig:
    name: str
    version: str
    schema: str
    hfLocalHub: Path
    ollama: Dict[str, Any]
    embeddingModel: str
    ollamaEmbeddingModel: str | None
    completionOptions: Dict[str, Any]
    newsEmbeddings: Path
    newsCSV: Path
    chromaPath: Path
    clothesData: Path | None = None
    faqData: Path | None = None
    productsChromaPath: Path | None = None


def resolve_path(path_str: str, base_dir: Path = PROJECT_ROOT) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data_section = data["data"]
    clothes_data_path = (
        resolve_path(data_section["clothesData"])
        if "clothesData" in data_section
        else None
    )
    faq_data_path = (
        resolve_path(data_section["faqData"])
        if "faqData" in data_section
        else None
    )
    products_chroma_path = (
        resolve_path(data_section["productsChromaPath"])
        if "productsChromaPath" in data_section
        else None
    )

    return AppConfig(
        name=data["name"],
        version=data["version"],
        schema=data["schema"],
        hfLocalHub=resolve_path(data["models"]["hfLocalHub"]),
        ollama=data["models"]["ollama"],
        embeddingModel=data["models"]["embeddingModel"],
        ollamaEmbeddingModel=data["models"].get("ollamaEmbeddingModel"),
        completionOptions=data["completionOptions"],
        newsEmbeddings=resolve_path(data_section["newsEmbeddings"]),
        newsCSV=resolve_path(data_section["newsCSV"]),
        chromaPath=resolve_path(data_section["chromaPath"]),
        clothesData=clothes_data_path,
        faqData=faq_data_path,
        productsChromaPath=products_chroma_path,
    )


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def parse_ollama_host(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    return host, port


def ollama_is_running(cfg: AppConfig) -> bool:
    host, port = parse_ollama_host(cfg.ollama["url"])
    return is_port_open(host, port)


def start_ollama_windows() -> None:
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start Ollama: {e}")


def ensure_ollama_running(
    cfg: AppConfig,
    try_start: bool = False,
    wait_seconds: int = 10
) -> None:
    if ollama_is_running(cfg):
        return

    if try_start:
        start_ollama_windows()
        for _ in range(wait_seconds):
            if ollama_is_running(cfg):
                return
            time.sleep(1)

    raise RuntimeError(
        f"Ollama is not running or not reachable at {cfg.ollama['url']}"
    )


config = load_config()
ensure_ollama_running(config, try_start=False)
