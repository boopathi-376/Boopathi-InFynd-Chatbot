import time
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
import uvicorn
import ollama
import json
import re

# ====================== CONFIG ======================
QDRANT_URL = "http://localhost:6333"
MODEL_NAME = r"C:\OPENBOT\local_models\e5-base-v2"
OLLAMA_MODEL = "qwen2.5:7b"
TOP_K = 5

# ====================== INIT ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

model = SentenceTransformer(MODEL_NAME, device=device)
client = QdrantClient(url=QDRANT_URL)
app = FastAPI(title="OpenBot Validation API", version="1.4")


# ====================== REQUEST MODEL ======================
class QueryRequest(BaseModel):
    query: str


# ====================== QDRANT SEARCH FUNCTION ======================
def get_qdrant_results(query: str):
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    collections = [col.name for col in client.get_collections().collections]

    semantic_results = {}
    for col in collections:
        try:
            search_res = client.search(
                collection_name=col,
                query_vector=query_vector,
                limit=TOP_K,
                with_payload=True,
            )
            if search_res:
                semantic_results[col] = [hit.payload.get("text", "") for hit in search_res]
        except Exception as e:
            print(f"⚠️ Error searching {col}: {e}")
    return semantic_results


# ====================== LLM VALIDATION FUNCTION ======================
def validate_with_llm(query: str, qdrant_results: dict):
    llm_input = {"query": query, "qdrant_results": qdrant_results, "mode": "live"}

    prompt = f"""
You are a **strict JSON validator**.
Your ONLY source of truth is the data inside "qdrant_results".
Never use external knowledge or make assumptions.

### Task
1. Understand the user's query.
2. Match it ONLY with information available in "qdrant_results".
3. Extract the most relevant filters (keys and values) that align with the query intent.
4. If data is missing, clearly say that instead of guessing.

### Output Format (strict JSON only)
{{
  "intent": "<exact query>",
  "validated_filters": {{
    "<filter_name>": ["<relevant_value1>", "<relevant_value2>"]
  }},
  "reasoning": "<brief factual reason>"
}}

### Rules
- Do NOT fabricate or hallucinate filters.
- Use only keys and values exactly present in qdrant_results.
- Always output valid JSON (no markdown, no commentary).

### Input
{json.dumps(llm_input, indent=2)}
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 512}
    )

    text_response = response["message"]["content"].strip()

    # Safe JSON parsing
    try:
        parsed = json.loads(text_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text_response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except:
                parsed = {
                    "intent": "Error",
                    "validated_filters": {},
                    "reasoning": "Invalid JSON format from LLM."
                }
        else:
            parsed = {
                "intent": "Error",
                "validated_filters": {},
                "reasoning": text_response
            }

    return parsed


# ====================== SUGGESTION EXTRACTION FUNCTION ======================
import numpy as np

def get_semantic_suggestions(query: str, qdrant_results: dict, validated_filters: dict):
    """
    Suggest only missing or semantically related filters 
    that were NOT selected by the LLM.
    """
    query_emb = model.encode(query, normalize_embeddings=True)
    suggestions = {}
    similarity_threshold = 0.8  # Ignore weak matches

    # Normalize validated filter keys
    validated_keys = set(validated_filters.keys())

    for key, values in qdrant_results.items():
        # Skip already used keys
        if key in validated_keys:
            continue

        scored = []
        for val in values:
            try:
                val_emb = model.encode(val, normalize_embeddings=True)
                similarity = np.dot(query_emb, val_emb)
                scored.append((val, similarity))
            except Exception:
                continue

        # Keep top 3 suggestions above similarity threshold
        top_values = [
            v for v, s in sorted(scored, key=lambda x: x[1], reverse=True)
            if s >= similarity_threshold][:3]

        if top_values:
            suggestions[key] = top_values

    return suggestions



# ====================== MAIN ROUTE ======================
@app.post("/validate")
def validate(req: QueryRequest):
    query = req.query.strip()
    print(f"\n🔍 Received query: {query}")

    qdrant_results = get_qdrant_results(query)
    start_time = time.time()
    llm_output = validate_with_llm(query, qdrant_results)
    end_time = time.time()

    # 🧩 Add suggestions dynamically
    suggestions = get_semantic_suggestions(query, qdrant_results, llm_output)

    processing_time = round(end_time - start_time, 2)

    return {
        "query": query,
        "qdrant_result":qdrant_results,
        "llm_validated_output": llm_output,
        "suggestions": suggestions,
        "processing_time_seconds": processing_time,
        "mode": "live"
    }


# ====================== RUN SERVER ======================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
