# ===== Base Image =====
FROM python:3.10-slim

# ===== Set Working Directory =====
WORKDIR /app

# ===== Copy Files =====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== Copy Project Files =====
COPY scripts ./scripts
COPY docker-compose.yml .
COPY . .

# ===== Expose FastAPI Port =====
EXPOSE 8000

# ===== Environment Variables =====
ENV QDRANT_URL=http://qdrant:6333
ENV MODEL_NAME=/app/local_models/e5-base-v2
ENV OLLAMA_MODEL=qwen2.5:7b
ENV PYTHONUNBUFFERED=1

# ===== Run FastAPI Server =====
CMD ["python", "scripts/embed_all_json_collections.py","scripts/api_server.py"]
