"""One-time script: populate Chroma DB with FAQ embeddings for W5.

Stores each FAQ entry's ``question``/``answer``/``type`` as metadata in a
collection named ``"faq_w5"``, in the same Chroma directory already used for
products (``config.productsChromaPath``). Given the small size of the FAQ
dataset (~25 entries), the W5 notebook's setup cell also self-populates this
collection inline if it is missing — this script is provided as a standalone
alternative, consistent with the W4 population-script pattern.

Run this once, manually, before using the W5 notebook:
    python w5/populate_faq.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

from setting import config

COLLECTION_NAME = "faq_w5"

print("Loading FAQ data...")
faq_data = joblib.load(str(config.faqData))
print(f"Count: {len(faq_data)}")

questions = [item.get("question", "") for item in faq_data]
documents = [
    {
        "id": str(i),
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "type": item.get("type", ""),
    }
    for i, item in enumerate(faq_data)
]

print("Embedding FAQ questions...")
model = SentenceTransformer(config.embeddingModel, local_files_only=True)
embeddings = model.encode(questions, normalize_embeddings=False, show_progress_bar=True)
print(f"Embeddings shape: {np.array(embeddings).shape}")

print("Populating Chroma DB...")
store = PersistentClient(path=str(config.productsChromaPath))
collection = store.get_or_create_collection(COLLECTION_NAME)

ids = [d["id"] for d in documents]
metadatas = [{k: v for k, v in d.items() if k != "id"} for d in documents]
collection.upsert(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)

print(f"Done! Collection '{COLLECTION_NAME}' count: {collection.count()}")
