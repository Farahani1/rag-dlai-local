"""One-time script: populate Chroma DB with product embeddings."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

from setting import config

print("Loading products data...")
products_data = joblib.load(str(config.clothesData))
print(f"Count: {len(products_data)}")

# Build descriptions
descriptions = []
documents = []
for i, prod in enumerate(products_data):
    doc_id = str(prod.get("product_id", i))
    desc = (
        f"{prod.get('productDisplayName', '')} "
        f"{prod.get('masterCategory', '')} "
        f"{prod.get('subCategory', '')} "
        f"{prod.get('articleType', '')} "
        f"{prod.get('baseColour', '')} "
        f"{prod.get('season', '')} "
        f"{prod.get('usage', '')} "
        f"{prod.get('gender', '')}"
    ).strip()
    descriptions.append(desc)
    documents.append(
        {
            "id": doc_id,
            "title": prod.get("productDisplayName", ""),
            "chunk": desc,
            "pubDate": str(prod.get("year", "")),
            "link": "",
        }
    )

print("Batch embedding all products...")
model = SentenceTransformer(config.embeddingModel, local_files_only=True)
embeddings = model.encode(descriptions, normalize_embeddings=False, show_progress_bar=True)
print(f"Embeddings shape: {np.array(embeddings).shape}")

print("Populating Chroma DB...")
store = PersistentClient(path=str(config.productsChromaPath))
collection = store.get_or_create_collection("products")

# Add in chunks
chunk_size = 5000
ids = [d["id"] for d in documents]
metadatas = [
    {"title": d["title"], "chunk": d["chunk"], "pubDate": d["pubDate"], "link": d["link"]}
    for d in documents
]
for start_idx in range(0, len(documents), chunk_size):
    end_idx = min(start_idx + chunk_size, len(documents))
    collection.add(
        ids=ids[start_idx:end_idx],
        embeddings=embeddings[start_idx:end_idx].tolist(),
        metadatas=metadatas[start_idx:end_idx],
    )
    print(f"  Added {end_idx}/{len(documents)}...")

print(f"Done! Collection count: {collection.count()}")