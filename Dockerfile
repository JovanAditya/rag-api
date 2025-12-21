# ================================
# RAG API - FastAPI Python Backend
# ================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rag-model submodule and install its dependencies
COPY rag-model/ ./rag-model/
# Requirements file has invalid lines (shell commands), filter only valid package lines
RUN head -n 75 ./rag-model/requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Set PYTHONPATH agar rag_model bisa diimport
ENV PYTHONPATH="${PYTHONPATH}:/app/rag-model"

# Copy application code
COPY api/ ./api/

# Create data directories
RUN mkdir -p /app/data/chroma_db /app/data/bm25_cache /app/logs

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["python", "-m", "api.main"]
