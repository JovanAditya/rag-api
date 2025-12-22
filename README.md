# RAG API

REST API untuk sistem RAG Akademik Universitas Mercu Buana.

## 📋 Deskripsi

Repository ini menyediakan REST API untuk:
- **Query RAG**: Tanya-jawab dengan retrieval + generation
- **Manajemen Dokumen**: Upload, hapus, proses dokumen
- **Knowledge Base**: Indexing, reindexing, clear KB
- **Health Check**: Status sistem dan index

## 📁 Struktur Direktori

```
rag-api/
├── rag-model/              # Git Submodule (core RAG)
├── api/
│   ├── main.py             # FastAPI entry point
│   ├── routes/
│   │   ├── query.py        # /api/query endpoints
│   │   ├── documents.py    # /api/documents endpoints
│   │   ├── chunking.py     # /api/chunking endpoints
│   │   ├── knowledge_base.py # /api/kb endpoints
│   │   └── health.py       # /health endpoints
│   ├── services/
│   │   ├── rag_service.py
│   │   ├── document_service.py
│   │   ├── chunking_service.py
│   │   └── kb_service.py
│   └── models/
│       └── schemas.py      # Pydantic schemas
├── data/                   # Shared data
├── .env.example
├── requirements.txt
└── README.md
```

## ⚙️ Instalasi

```bash
# Clone dengan submodule
git clone --recurse-submodules https://github.com/JovanAditya/rag-api.git
cd rag-api

# Atau jika sudah clone
git submodule update --init --recursive

# Install environment dari rag-model (RECOMMENDED)
conda env create -f rag-model/environment.yml
conda activate academic-rag

# (Opsional) Install GPU support untuk PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Konfigurasi
cp .env.example .env
# Edit .env dengan API key
```

## 🔧 Konfigurasi

Edit file `.env`:

```env
HOST=0.0.0.0
PORT=5001

# Pilih salah satu provider LLM:
# - openrouter (RECOMMENDED - banyak model gratis!)
# - gemini
# - openai
# - anthropic
# - ollama (local)

# === OpenRouter (Recommended) ===
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-your-key
# Model gratis:
# - google/gemini-2.0-flash-exp:free  (Recommended)
# - google/gemma-2-9b-it:free
# - meta-llama/llama-3.2-3b-instruct:free
# - qwen/qwen-2.5-7b-instruct:free
OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free

# === Gemini (alternatif) ===
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your-api-key
# GEMINI_MODEL=gemini-2.0-flash

RAG_MODEL_PATH=./rag-model
```

## 🚀 Menjalankan Server

```bash
# Opsi A: Simple / Production (Default Port 5001)
python -m api.main

# Opsi B: Development (Auto-Reload)
# Note: Harus set --port 5001 karena default uvicorn adalah 8000
uvicorn api.main:app --reload --host 0.0.0.0 --port 5001

# Opsi C: Production (Uvicorn Workers)
uvicorn api.main:app --host 0.0.0.0 --port 5001 --workers 4
```

Server berjalan di `http://localhost:5001`

**Dokumentasi OpenAPI**: `http://localhost:5001/docs`

## 📡 API Endpoints

### Query
| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/api/query` | POST | Query RAG dengan pertanyaan |
| `/api/query/batch` | POST | Batch query multiple pertanyaan |

### Dokumen
| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/api/documents` | GET | List semua dokumen |
| `/api/documents/{id}` | GET | Detail dokumen |
| `/api/documents/upload` | POST | Upload dokumen (multipart) |
| `/api/documents/{id}` | DELETE | Hapus dokumen |

### Chunking
| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/api/chunking/process` | POST | Proses dokumen spesifik |
| `/api/chunking/process-all` | POST | Proses semua dokumen |
| `/api/chunking/config` | GET/PUT | Konfigurasi chunking |

### Knowledge Base
| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/api/kb/stats` | GET | Statistik knowledge base |
| `/api/kb/search` | GET/POST | Search chunks |
| `/api/kb/reindex` | POST | Rebuild index |
| `/api/kb/clear` | DELETE | Hapus semua (confirm=true) |

### Health
| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/health` | GET | Quick health check |
| `/health/detailed` | GET | Health check detail |

## 🔗 Git Submodule

```bash
# Clone dengan submodules
git clone --recurse-submodules https://github.com/JovanAditya/rag-api.git

# Update submodule ke commit terbaru
git submodule update --remote

# Commit perubahan reference
git add rag-model
git commit -m "Update rag-model submodule"
```

## 📦 Dependencies

| Package | Deskripsi |
|---------|-----------|
| fastapi | Web framework |
| uvicorn | ASGI server |
| pydantic | Data validation |
| pdfplumber | PDF extraction |
| python-multipart | File upload |
| rag-model | Core RAG (submodule) |

## 🔗 Repository Terkait

| Repository | Deskripsi |
|------------|-----------|
| [rag-model](https://github.com/JovanAditya/rag-model) | Core RAG Model |
| [rag-web](https://github.com/JovanAditya/rag-web) | Laravel Frontend |
| [rag-deploy](https://github.com/JovanAditya/rag-deploy) | Docker Orchestration |

---

*Bagian dari proyek skripsi Sistem RAG Akademik - Universitas Mercu Buana*
