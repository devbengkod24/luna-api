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

5. **Start Qdrant**
```bash
# Dengan Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

6. **Setup environment**
```bash
cp env.template .env
# Edit .env dengan connection strings untuk localhost
```

7. **Run migrations**
```bash
alembic upgrade head
```

8. **Start development server**
```bash
uvicorn src.drivers.main:app --reload
```

9. **Akses aplikasi**
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
curl -N -X POST http://localhost:8000/v1/chat/text \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Halo, apa kabar?",
    "context": {
      "emotion": "happy",
      "name": "Test User"
    }
  }'
```

**Response** (Server-Sent Events):
```
data: {"type": "log", "content": "Membuat sesi baru..."}

data: {"type": "log", "content": "Sesi dibuat: Percakapan dengan Test User"}

data: {"type": "log", "content": "Menganalisis konteks..."}

data: {"id": "chatcmpl-abc123", "object": "chat.chunk", "created": 1701234567, "model": "vsa-assistant", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Halo"}, "finish_reason": null}]}

data: {"id": "chatcmpl-abc123", "object": "chat.chunk", "created": 1701234567, "model": "vsa-assistant", "choices": [{"index": 0, "delta": {"content": " Test"}, "finish_reason": null}]}

data: {"id": "chatcmpl-abc123", "object": "chat.chunk", "created": 1701234567, "model": "vsa-assistant", "choices": [{"index": 0, "delta": {"content": " User!"}, "finish_reason": null}]}

data: {"id": "chatcmpl-abc123", "object": "chat.chunk", "created": 1701234567, "model": "vsa-assistant", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: {"type": "metadata", "session_uuid": "a1b2c3d4-5678-90ab-cdef-1234567890ab"}

data: [DONE]

```

**Note**: 
- Flag `-N` / `--no-buffer` diperlukan untuk streaming SSE
- Response menggunakan format OpenAI-compatible
- Setiap chunk adalah JSON terpisah dengan prefix `data: `
- Stream diakhiri dengan `data: [DONE]`

### 3. Get Chat History

```bash
curl http://localhost:8000/v1/chat/history
```

**Response**:
```json
[
  {
    "id": 1,
    "uuid": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
    "title": "Percakapan dengan Test User",
    "created_at": "2025-12-05T10:30:00Z",
    "updated_at": "2025-12-05T10:35:00Z"
  }
]
```

### 4. Chat Text SSE (dengan Python)

```python
import requests
import json

def test_chat_stream():
    url = "http://localhost:8000/v1/chat/text"
    
    payload = {
        "prompt": "Ceritakan tentang cuaca hari ini",
        "context": {
            "emotion": "happy",
            "name": "Test User"
        }
    }
    
    # Stream response
    response = requests.post(url, json=payload, stream=True)
    
    for line in response.iter_lines():
        if line:
            # Decode line
            line_str = line.decode('utf-8')
            
            # Skip empty lines
            if not line_str.strip():
                continue
                
            # Parse SSE format (remove "data: " prefix)
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove "data: "
                
                # Check for [DONE] marker
                if data_str == "[DONE]":
                    print("\nStream selesai!")
                    break
                
                # Parse JSON
                try:
                    data = json.loads(data_str)
                    
                    # Handle different event types
                    if data.get("type") == "log":
                        print(f"[LOG] {data['content']}")
                    
                    elif data.get("type") == "metadata":
                        if "session_uuid" in data:
                            print(f"[SESSION] UUID: {data['session_uuid']}")
                        if "transcript" in data:
                            print(f"[TRANSCRIPT] {data['transcript']}")
                    
                    elif data.get("object") == "chat.chunk":
                        # Extract content from choices
                        for choice in data.get("choices", []):
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                            
                            # Check finish reason
                            if choice.get("finish_reason"):
                                print()  # New line
                                print(f"[FINISH] {choice['finish_reason']}")
                
                except json.JSONDecodeError:
                    print(f"Failed to parse: {data_str}")

if __name__ == "__main__":
    test_chat_stream()
```

### 5. WebSocket Vision (dengan Python)

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
# Simpan face encodings ke database
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

