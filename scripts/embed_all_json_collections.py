import os
import json
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import math

# ====================== CONFIG ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "local_models", "e5-base-v2")  # local path
QDRANT_URL = "http://localhost:6333"
REMOTE_MODEL = "intfloat/e5-base-v2"
VECTOR_SIZE = 768
MAX_BATCH_SIZE = 1000  # smaller batches to avoid payload too large errors

# ====================== INIT ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Check if model exists locally, else download once
if os.path.exists(LOCAL_MODEL_DIR):
    print(f"📦 Loading model from local path: {LOCAL_MODEL_DIR}")
    model = SentenceTransformer(LOCAL_MODEL_DIR, device=device)
else:
    print(f"🌐 Downloading model from Hugging Face: {REMOTE_MODEL}")
    model = SentenceTransformer(REMOTE_MODEL, device=device)
    os.makedirs(os.path.dirname(LOCAL_MODEL_DIR), exist_ok=True)
    model.save(LOCAL_MODEL_DIR)
    print(f"✅ Model saved locally at: {LOCAL_MODEL_DIR}")

print(f"🚀 Using device: {device}")
client = QdrantClient(url=QDRANT_URL)

# ====================== HELPER ======================
def extract_text(record):
    """Extract all text-like values from a JSON record."""
    if isinstance(record, dict):
        values = []
        for key, val in record.items():
            key = key.strip().replace("\ufeff", "")
            if isinstance(val, str):
                values.append(val.strip())
            elif isinstance(val, (int, float, bool)):
                values.append(str(val))
        return " | ".join(values)
    return ""

# ====================== MAIN ======================
for file_name in os.listdir(DATA_DIR):
    if not file_name.lower().endswith(".json"):
        print(f"⏩ Skipping non-JSON file: {file_name}")
        continue

    collection_name = file_name.replace(".json", "")
    file_path = os.path.join(DATA_DIR, file_name)

    print(f"\n📂 Processing file: {collection_name}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read {file_name}: {e}")
        continue

    if not isinstance(data, list) or len(data) == 0:
        print(f"⚠️ Empty or invalid data in {file_name}")
        continue

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE)
    )

    texts = [extract_text(item) for item in data if extract_text(item)]
    print(f"🧩 Found {len(texts)} valid records to embed in {collection_name}")

    # Batch encode and upsert
    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i:i + MAX_BATCH_SIZE]
        embeddings = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
        points = [
            qmodels.PointStruct(
                id=i + j + 1,
                vector=emb.tolist(),
                payload={"text": batch[j]}
            )
            for j, emb in enumerate(embeddings)
        ]
        client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Uploaded batch {i // MAX_BATCH_SIZE + 1}/{math.ceil(len(texts)/MAX_BATCH_SIZE)} ({len(points)} vectors)")

print("\n🎉 All JSON files processed and stored successfully in Qdrant!")
