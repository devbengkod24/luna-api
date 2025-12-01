# Quick Start Guide

Panduan cepat untuk menjalankan aplikasi VSA Backend API.

## 🚀 Opsi 1: Menggunakan Docker (Recommended)

### Prerequisites
- Docker Desktop atau Docker Engine
- Docker Compose

### Langkah-langkah

1. **Clone repository**
```bash
git clone <repository-url>
cd VSA-BE-FastAPIv2
```

2. **Setup environment variables**
```bash
cp env.template .env
# Edit .env sesuai kebutuhan (minimal update API keys)
```

3. **Start semua services**
```bash
docker compose up -d
```

4. **Cek status services**
```bash
docker compose ps
```

5. **Run database migrations**
```bash
docker compose exec app alembic upgrade head
```

6. **Akses aplikasi**
- API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs
- API Docs (ReDoc): http://localhost:8000/redoc

7. **View logs**
```bash
# Semua services
docker compose logs -f

# Specific service
docker compose logs -f app
```

### Menghentikan services

```bash
# Stop semua
docker compose down

# Stop dan hapus volumes (WARNING: data akan hilang)
docker compose down -v
```

---

## 💻 Opsi 2: Local Development (Tanpa Docker)

### Prerequisites
- Python 3.11+
- PostgreSQL 16
- Redis 7
- Qdrant (running locally)

### Langkah-langkah

1. **Clone repository**
```bash
git clone <repository-url>
cd VSA-BE-FastAPIv2
```

2. **Create virtual environment**
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup PostgreSQL**
```bash
# Create database
createdb vsa_db

# Create user (opsional)
psql -c "CREATE USER vsa_user WITH PASSWORD 'vsa_password';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE vsa_db TO vsa_user;"
```

5. **Start Redis**
```bash
redis-server
```

6. **Start Qdrant**
```bash
# Dengan Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

7. **Setup environment**
```bash
cp env.template .env
# Edit .env dengan connection strings untuk localhost
```

8. **Run migrations**
```bash
alembic upgrade head
```

9. **Start development server**
```bash
uvicorn src.drivers.rest.main:app --reload
```

10. **Akses aplikasi**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🧪 Testing API

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Chat Text (dengan curl)

```bash
curl -X POST http://localhost:8000/v1/chat/text \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Halo, apa kabar?",
    "context": {
      "emotion": "happy",
      "name": "Test User"
    }
  }'
```

### 3. Get Chat History

```bash
curl http://localhost:8000/v1/chat/history
```

### 4. WebSocket Vision (dengan Python)

```python
import asyncio
import websockets
import json
import base64

async def test_vision():
    uri = "ws://localhost:8000/v1/vision/stream"
    
    async with websockets.connect(uri) as websocket:
        # Kirim frame (contoh dengan file image)
        with open("test_frame.jpg", "rb") as f:
            frame_data = base64.b64encode(f.read()).decode()
        
        await websocket.send(json.dumps({
            "event": "webcam_frame",
            "data": frame_data
        }))
        
        # Terima response
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(test_vision())
```

---

## 📝 Next Steps

### 1. Implementasi Adapters

Adapters berikut masih template dan perlu implementasi:

- **LLM Adapter** (`src/adapters/llm/llm_adapter.py`)
  - Tambahkan integrasi dengan OpenAI, Anthropic, atau local model
  
- **Vision Adapter** (`src/adapters/vision/vision_adapter.py`)
  - Implementasi model emotion detection
  - Implementasi model face recognition
  
- **Speech Adapters** (`src/adapters/speech/`)
  - STT: Integrasi Whisper atau service lain
  - TTS: Integrasi TTS service

- **RAG Adapter** (`src/adapters/qdrant/rag_adapter.py`)
  - Implementasi embedding generation
  - Implementasi document chunking

### 2. Konfigurasi API Keys

Edit file `.env`:
```bash
# LLM (e.g., OpenAI)
LLM_API_KEY=sk-...

# STT (e.g., OpenAI Whisper)
STT_API_KEY=sk-...
```

### 3. Upload Knowledge Base (untuk RAG)

Buat endpoint atau script untuk upload dokumen ke Qdrant:

```python
# Example script
from src.adapters.qdrant.rag_adapter import QdrantRAGAdapter

adapter = QdrantRAGAdapter(url="http://localhost:6333")
await adapter.initialize()
await adapter.add_document(
    text="Your knowledge base content...",
    metadata={"source": "manual", "topic": "faq"}
)
```

### 4. Setup Face Recognition Database

Tambahkan face encodings untuk user yang dikenal:

```python
# Simpan face encodings ke Redis atau database
# Implementasi di vision_adapter.py
```

---

## 🔧 Troubleshooting

### Database connection error

```bash
# Cek apakah PostgreSQL berjalan
docker compose ps postgres

# Cek logs
docker compose logs postgres

# Restart service
docker compose restart postgres
```

### Redis connection error

```bash
# Cek Redis
docker compose ps redis

# Test connection
docker compose exec redis redis-cli ping
```

### Port already in use

```bash
# Cek port usage
sudo lsof -i :8000

# Atau ubah port di docker-compose.yml
ports:
  - "8001:8000"  # host:container
```

### Migration errors

```bash
# Reset database (WARNING: deletes data)
docker compose exec app alembic downgrade base
docker compose exec app alembic upgrade head
```

---

## 📚 Resources

- [Main README](../README.md) - Full documentation
- [Database Schema](DATABASE_SCHEMA.md) - Database design
- [API Docs](http://localhost:8000/docs) - Interactive API documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)

---

## 💡 Tips

1. **Development Mode**: Set `DEBUG=true` di `.env` untuk detailed error messages
2. **Hot Reload**: Uvicorn dengan `--reload` akan restart otomatis saat code berubah
3. **Database GUI**: Gunakan tools seperti pgAdmin, DBeaver, atau TablePlus untuk manage database
4. **API Testing**: Gunakan Postman atau Thunder Client untuk test API endpoints
5. **Logs**: Selalu cek logs saat debugging: `docker compose logs -f app`

