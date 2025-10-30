Here’s a **clean, professional, and submission-ready** version of your `README.md` for **OpenBot Validation & Semantic Retrieval System** — polished for company or project submission with clear structure, Markdown formatting, and concise explanations.

---

# 🤖 OpenBot: Validation & Semantic Retrieval System

**OpenBot** is an intelligent **FastAPI + Qdrant-based semantic query understanding and validation system**.
It enables natural language query interpretation, contextual retrieval, and validation using **local embeddings** and **LLMs**.

---

## 🧠 Overview

OpenBot combines **semantic vector search** and **LLM reasoning** to extract relevant filters and validate user intents from natural language queries.

### 🔹 Core Components

| Component                            | Purpose                                     | Notes                                               |
| ------------------------------------ | ------------------------------------------- | --------------------------------------------------- |
| **FastAPI**                          | REST API layer for validation and reasoning | Fully async-capable                                 |
| **SentenceTransformer (E5-Base-V2)** | Local embedding model                       | Used for semantic understanding and Qdrant matching |
| **Qdrant**                           | Vector database for fast semantic retrieval | Runs inside Docker                                  |
| **Torch + NumPy**                    | Vector math and GPU acceleration            | Uses CUDA if available                              |
| **Qwen2.5:7B (via Ollama)**          | Local LLM for reasoning and validation      | Optional but enhances accuracy                      |

---

## 📁 Project Structure

```
OPENBOT/
├── scripts/
│   ├── api_server.py             # FastAPI-based validation API
│   ├── embed_to_qdrant.py        # JSON → embeddings → Qdrant ingestion
│
├── data/                         # Source JSON data for Qdrant
│   ├── company_type.json
│   ├── job_function.json
│   └── ...
│
├── local_models/
│   └── e5-base-v2/               # Local embedding model (downloaded once)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ System Requirements

* **Python 3.10+**
* **Docker** (for Qdrant)
* **RAM:** 4GB+ (Recommended: 8GB)
* **Optional GPU:** CUDA for faster embedding generation

---

## 🧩 1️⃣ Setup & Installation

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
fastapi==0.115.0
uvicorn==0.30.0
qdrant-client==1.10.1
sentence-transformers==3.0.1
torch>=2.2.0
numpy>=1.26.0
pydantic>=2.8.0
cachetools
regex
json5
```

---

## 🐳 2️⃣ Run Qdrant Database (via Docker)

Start Qdrant locally using Docker:

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage:z \
  qdrant/qdrant
```

Check the Qdrant Dashboard:

👉 [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

## 🧠 3️⃣ Embed Data into Qdrant

Place your `.json` data files inside `/data`, then run:

```bash
python scripts/embed_to_qdrant.py
```

This script will:

* Load the **E5-Base-V2** embedding model
* Convert all JSON fields into dense vector embeddings
* Upload them to **Qdrant** in batch mode

---

## 🚀 4️⃣ Start the API Server

Run the main FastAPI server:

```bash
python scripts/api_server.py
```

The server starts at:
👉 [http://localhost:8000](http://localhost:8000)

Explore interactive API docs at:
👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💡 Example Query Flow

### 🧾 Input

```json
{
  "query": "List all Fintech companies in the UK"
}
```

### ⚙️ Processing Steps

1. Query is embedded using **E5-Base-V2**
2. Qdrant returns **top semantic matches**
3. LLM (**Qwen2.5:7B**) validates and explains reasoning
4. Suggestions are returned for structured filtering

### ✅ Output

```json
{
  "query": "Top 10 companies in UK",
  "qdrant_result": { ... },
  "llm_validated_output": {
    "intent": "Top 10 companies in UK",
    "validated_filters": {
      "cd_geographyCountries": [
        "United Kingdom | united_kingdom"
      ]
    },
    "reasoning": "The query asks for top companies in the UK, and 'cd_geographyCountries' contains 'United Kingdom | united_kingdom', which matches the intent."
  },
  "suggestions": { ... },
  "processing_time_seconds": 16.93,
  "mode": "live"
}
```

---

## 🧩 Features

✅ Local **embedding-based semantic retrieval**
✅ **Qdrant integration** for high-speed vector search
✅ **LLM-powered reasoning** using Qwen2.5 or other Ollama models
✅ Modular scripts for data ingestion & validation
✅ Supports **offline and GPU-accelerated** workflows

---
