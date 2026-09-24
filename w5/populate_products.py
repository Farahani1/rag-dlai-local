"""One-time script: populate Chroma DB with product embeddings for W5.

Unlike ``w4/populate_products.py`` (which stores a narrow
``title/chunk/pubDate/link`` metadata schema into a collection named
``"products"``), this script stores the **full** scalar product metadata
(``gender``, ``masterCategory``, ``subCategory``, ``articleType``,
``baseColour``, ``season``, ``usage``, ``year``, ``price``,
``productDisplayName``, ``product_id``) into a separate collection named
``"products_w5"``, in the same Chroma directory used by W4
(``config.productsChromaPath``). W5's metadata-filtering exercises need these
fields; W4's ``"products"`` collection is left untouched.

Run this once, manually, before using the W5 notebook:
    python w5/populate_products.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

from setting import config

COLLECTION_NAME = "products_w5"

print("Loading products data...")
products_data = joblib.load(str(config.clothesData))
print(f"Count: {len(products_data)}")

# Build descriptions and full metadata
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
            "product_id": prod.get("product_id", i),
            "productDisplayName": prod.get("productDisplayName", ""),
            "masterCategory": prod.get("masterCategory", ""),
            "subCategory": prod.get("subCategory", ""),
            "articleType": prod.get("articleType", ""),
            "baseColour": prod.get("baseColour", ""),
            "season": prod.get("season", ""),
            "usage": prod.get("usage", ""),
            "gender": prod.get("gender", ""),
            "year": prod.get("year", ""),
            "price": prod.get("price", 0),
        }
    )

print("Batch embedding all products...")
model = SentenceTransformer(config.embeddingModel, local_files_only=True)
embeddings = model.encode(descriptions, normalize_embeddings=False, show_progress_bar=True)
print(f"Embeddings shape: {np.array(embeddings).shape}")

print("Populating Chroma DB...")
store = PersistentClient(path=str(config.productsChromaPath))
collection = store.get_or_create_collection(COLLECTION_NAME)

# Add in chunks
chunk_size = 5000
ids = [d["id"] for d in documents]
metadatas = [{k: v for k, v in d.items() if k != "id"} for d in documents]
for start_idx in range(0, len(documents), chunk_size):
    end_idx = min(start_idx + chunk_size, len(documents))
    collection.add(
        ids=ids[start_idx:end_idx],
        embeddings=embeddings[start_idx:end_idx].tolist(),
        metadatas=metadatas[start_idx:end_idx],
    )
    print(f"  Added {end_idx}/{len(documents)}...")

print(f"Done! Collection '{COLLECTION_NAME}' count: {collection.count()}")
