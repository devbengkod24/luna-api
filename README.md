# VSA Backend API v2

Backend API untuk sistem Virtual Smart Assistant (VSA) dengan fitur:
- **Emotion Detection** - Deteksi emosi dari video frame
- **Face Recognition** - Pengenalan wajah pengguna
- **LLM + RAG** - Chatbot dengan knowledge base
- **TTS** - Text-to-Speech
- **STT** - Speech-to-Text

Dibangun dengan **Clean Architecture** dan teknologi modern.

## 🏗️ Tech Stack

### Backend Framework
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Database
- **PostgreSQL** - Relational database
- **Qdrant** - Vector database untuk RAG

### ORM & Migrations
- **SQLAlchemy 2.0** - Async ORM
- **Alembic** - Database migrations

### Containerization
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

## 📁 Struktur Proyek

Proyek ini mengikuti **Clean Architecture** dengan pemisahan layer yang jelas:

```
VSA-BE-FastAPIv2/
├── alembic/                    # Database migrations
│   ├── versions/               # Migration files
│   ├── env.py                  # Alembic environment
│   └── script.py.mako          # Migration template
│
├── src/
│   ├── domain/                 # Domain Layer (Business Entities)
│   │   └── entities/
│   │       ├── chat.py         # Chat session & message entities
│   │       ├── vision.py       # Vision result entities
│   │       └── context.py      # User context entities
│   │
│   ├── ports/                  # Port Layer (Interfaces)
│   │   ├── repository.py       # Repository interfaces
│   │   ├── llm.py              # LLM adapter interface
│   │   ├── rag.py              # RAG adapter interface
│   │   ├── vision.py           # Vision adapter interface
│   │   ├── audio.py            # STT/TTS adapter interfaces
│   │   └── cache.py            # Cache adapter interface
│   │
│   ├── adapters/               # Adapter Layer (Implementations)
│   │   ├── database/           # PostgreSQL adapter
│   │   │   ├── models.py       # SQLAlchemy models
│   │   │   ├── connection.py  # DB connection manager
│   │   │   └── repository.py  # Repository implementation
│   │   ├── qdrant/             # Qdrant vector DB adapter
│   │   │   └── rag_adapter.py
│   │   ├── llm/                # LLM adapter (OpenAI, etc.)
│   │   │   └── llm_adapter.py
│   │   ├── vision/             # Vision AI adapter
│   │   │   └── vision_adapter.py
│   │   └── audio/              # Audio adapters
│   │       ├── stt_adapter.py  # Speech-to-Text
│   │       └── tts_adapter.py  # Text-to-Speech
│   │
│   ├── use_cases/              # Use Case Layer (Business Logic)
│   │   ├── chat_text.py        # Text chat use case
│   │   ├── chat_speech.py      # Speech chat use case
│   │   ├── chat_history.py     # Chat history use cases
│   │   └── vision.py    # Vision use case
│   │
│   ├── drivers/                # Driver Layer (External Interfaces)
│   │   ├── rest/               # HTTP REST API
│   │   │   ├── main.py         # FastAPI app entry point
│   │   │   ├── dependencies.py # Dependency injection
│   │   │   └── routers/
│   │   │       └── chat.py     # Chat HTTP endpoints
│   │   └── realtime/           # WebSocket API
│   │       └── routers/
│   │           └── vision.py   # Vision WebSocket endpoint
│   │
│   └── config.py               # Application configuration
│
├── tests/                      # Unit & integration tests
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Docker image definition
├── alembic.ini                 # Alembic configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📂 Penjelasan Layer

### 1. Domain Layer (`src/domain/`)
**Aturan bisnis inti yang independen dari teknologi.**

- **entities/**: Objek bisnis murni tanpa dependencies eksternal
  - `chat.py`: Entity untuk chat session dan messages
  - `vision.py`: Entity untuk hasil analisis vision
  - `context.py`: Entity untuk konteks pengguna

### 2. Ports Layer (`src/ports/`)
**Interface/kontrak yang mendefinisikan komunikasi antar layer.**

- Mendefinisikan interface untuk repository dan adapters
- Tidak ada implementasi, hanya kontrak
- Memungkinkan dependency inversion

### 3. Adapters Layer (`src/adapters/`)
**Implementasi konkrit dari ports untuk teknologi spesifik.**

- **database/**: PostgreSQL dengan SQLAlchemy
- **qdrant/**: Qdrant vector database untuk RAG
- **llm/**: LLM integration (OpenAI, Anthropic, dll)
- **vision/**: Computer vision models (emotion, face recognition)
- **audio/**: STT/TTS services

### 4. Use Cases Layer (`src/use_cases/`)
**Orchestration logic yang menggunakan ports.**

- Business logic independen dari framework
- Menggunakan dependency injection
- Fokus pada alur bisnis, bukan implementasi teknis

### 5. Drivers Layer (`src/drivers/`)
**Entry points untuk aplikasi (HTTP, WebSocket, CLI, dll).**

- **rest/**: FastAPI HTTP endpoints
- **realtime/**: WebSocket endpoints
- Menghubungkan external requests ke use cases

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (untuk development tanpa Docker)

### 1. Clone Repository
```bash
git clone <repository-url>
cd VSA-BE-FastAPIv2
```

### 2. Setup Environment Variables
```bash
cp .env.example .env
# Edit .env dengan konfigurasi Anda
```

### 3. Run with Docker Compose
```bash
docker compose up --build
```

Aplikasi akan berjalan di:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **Qdrant**: http://localhost:6333

### 4. Run Database Migrations
```bash
# Di container
docker compose exec app alembic upgrade head

# Atau lokal
alembic upgrade head
```

### 5. Create New Migration (jika ada perubahan model)
```bash
# Di container
docker compose exec app alembic revision --autogenerate -m "description"

# Atau lokal
alembic revision --autogenerate -m "description"
```

## 📡 API Specification

### Base URL
```
http://localhost:8000
```

### 1. Vision Stream (WebSocket)

**Endpoint**: `ws://localhost:8000/v1/vision/stream`

**Protokol**: WebSocket untuk streaming video frames real-time

**Request**:
```json
{
  "event": "webcam_frame",
  "data": "base64-encoded-string-dari-satu-frame-jpeg"
}
```

**Response**:
```json
{
  "event": "vision_update",
  "payload": {
    "faces": [
      {
        "name": "User_ABC",
        "box": [10, 10, 50, 50],
        "confidence": 0.95
      },
      {
        "name": "User_XYZ",
        "box": [100, 15, 50, 50],
        "confidence": 0.88
      }
    ],
    "emotions": [
      {
        "emotion": "happy",
        "box": [10, 10, 50, 50],
        "confidence": 0.8
      },
      {
        "emotion": "neutral",
        "box": [100, 15, 50, 50],
        "confidence": 0.9
      }
    ]
  }
}
```

**Response (no face detected)**:
```json
{
  "event": "vision_update",
  "payload": {
    "faces": [],
    "emotions": []
  }
}
```

---

### 2. Chat Text (HTTP POST)

**Endpoint**: `POST /v1/chat/text`

**Protokol**: RESTful HTTP dengan Server-Sent Events (SSE) streaming

**Request Body**:
```json
{
  "uuid": "optional-session-uuid",
  "prompt": "Apa pendapatmu tentang ini?",
  "context": {
    "emotion": "happy",
    "name": "User_ABC"
  }
}
```

**Response**: `Content-Type: text/event-stream`

Format mengikuti standar OpenAI-compatible streaming:

```
data: {"type": "log", "content": "Membuat sesi baru..."}

data: {"type": "log", "content": "Sesi dibuat: Diskusi tentang..."}

data: {"type": "log", "content": "Menganalisis konteks..."}

data: {"id": "chatcmpl-abc123", "object": "chat.chunk", "created": 1701234567, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Tentu,"}, "finish_reason": null}]}

data: {"id": "chatcmpl-abc123", "object": "chat.chunk", "created": 1701234567, "choices": [{"index": 0, "delta": {"content": " saya"}, "finish_reason": null}]}

data: {"id": "chatcmpl-abc123", "object": "chat.chunk", "created": 1701234567, "choices": [{"index": 0, "delta": {"content": " akan"}, "finish_reason": null}]}

data: {"id": "chatcmpl-abc123", "object": "chat.chunk", "created": 1701234567, "choices": [{"index": 0, "delta": {"content": " membantu..."}, "finish_reason": null}]}

data: {"id": "chatcmpl-abc123", "object": "chat.chunk", "created": 1701234567, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: {"type": "metadata", "session_uuid": "a1b2c3d4-5678-90ab-cdef-1234567890ab"}

data: [DONE]

```

**Event Types**:
- **Log events**: `{"type": "log", "content": "..."}` - Status/progress messages
- **Chat chunks**: `{"object": "chat.chunk", "choices": [...]}` - Streaming LLM tokens (OpenAI-compatible)
  - First chunk includes `"role": "assistant"` in delta
  - Subsequent chunks only have `"content"` in delta
  - Final chunk has empty delta with `"finish_reason": "stop"`
- **Metadata**: `{"type": "metadata", "session_uuid": "..."}` - Session information
- **End marker**: `[DONE]` - Indicates stream completion

---

### 3. Chat Audio (HTTP POST)

**Endpoint**: `POST /v1/chat/audio`

**Protokol**: RESTful HTTP dengan SSE streaming

**Request Body**:
```json
{
  "uuid": "optional-session-uuid",
  "audio": "base64-encoded-audio-string",
  "context": {
    "emotion": "neutral",
    "name": "User_ABC"
  }
}
```

**Response**: `Content-Type: text/event-stream`

```
data: {"type": "log", "content": "Mentranskripsi speech..."}

data: {"type": "metadata", "transcript": "Apa pendapatmu tentang cuaca hari ini?"}

data: {"type": "log", "content": "Menganalisis konteks..."}

data: {"id": "chatcmpl-def456", "object": "chat.chunk", "created": 1701234567, "model": "vsa-assistant", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Cuaca"}, "finish_reason": null}]}

data: {"id": "chatcmpl-def456", "object": "chat.chunk", "created": 1701234567, "model": "vsa-assistant", "choices": [{"index": 0, "delta": {"content": " hari"}, "finish_reason": null}]}

data: {"id": "chatcmpl-def456", "object": "chat.chunk", "created": 1701234567, "model": "vsa-assistant", "choices": [{"index": 0, "delta": {"content": " ini..."}, "finish_reason": null}]}

data: {"id": "chatcmpl-def456", "object": "chat.chunk", "created": 1701234567, "model": "vsa-assistant", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: {"type": "metadata", "session_uuid": "e5f6g7h8-9012-34ij-klmn-5678901234op"}

data: [DONE]

```

**Event Types**:
- **Log events**: Status messages during processing
- **Metadata (transcript)**: `{"type": "metadata", "transcript": "..."}` - Transcribed text from audio
- **Chat chunks**: OpenAI-compatible streaming tokens
- **Metadata (session)**: Session UUID after completion
- **End marker**: `[DONE]`

---

### 4. Get Chat History (HTTP GET)

**Endpoint**: `GET /v1/chat/history`

**Protokol**: RESTful HTTP

**Response**: `Content-Type: application/json`
```json
[
  {
    "id": 1,
    "uuid": "a1b2c3d4-...",
    "title": "Obrolan tentang cuaca",
    "created_at": "2025-11-01T15:30:00Z",
    "updated_at": "2025-11-01T15:35:00Z"
  },
  {
    "id": 2,
    "uuid": "e5f6g7h8-...",
    "title": "Analisis Visi",
    "created_at": "2025-11-01T16:00:00Z",
    "updated_at": "2025-11-01T16:05:00Z"
  }
]
```

---

### 5. Get Chat Session Detail (HTTP GET)

**Endpoint**: `GET /v1/chat/history/{uuid}`

**Protokol**: RESTful HTTP

**Path Parameters**:
- `uuid`: Session UUID

**Response**: `Content-Type: application/json`
```json
{
  "id": 1,
  "uuid": "a1b2c3d4-...",
  "title": "Obrolan tentang cuaca",
  "messages": [
    {
      "role": "user",
      "content": "Bagaimana cuaca hari ini?",
      "context": {
        "emotion": "neutral",
        "name": "User_ABC"
      }
    },
    {
      "role": "assistant",
      "content": "Cuaca hari ini cerah.",
      "context": null
    },
    {
      "role": "user",
      "content": "Terima kasih!",
      "context": {
        "emotion": "happy",
        "name": "User_ABC"
      }
    }
  ]
}
```

---

### 6. Health Check (HTTP GET)

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "app": "VSA-BE-FastAPI",
  "version": "2.0.0"
}
```

## 📖 Documentation

### Detailed Guides

- **[Quick Start Guide](docs/QUICKSTART.md)** - Setup dan running aplikasi
- **[SSE API Format](docs/SSE_API_FORMAT.md)** - Format lengkap untuk Server-Sent Events streaming
- **[Database Schema](docs/DATABASE_SCHEMA.md)** - Database design dan relationships
- **[RAG Setup](docs/RAG_SETUP.md)** - Setup Retrieval-Augmented Generation

---

## 🧪 Development

### Install Dependencies Locally
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Run Development Server
```bash
uvicorn src.drivers.main:app --reload
```

### Create Database Migration
```bash
alembic revision --autogenerate -m "Add new table"
alembic upgrade head
```

### Run Tests (Coming Soon)
```bash
pytest
```

## 🔧 Configuration

Konfigurasi aplikasi menggunakan environment variables. Lihat `env.template` untuk template.

**Key Variables**:
- `DATABASE_URL`: PostgreSQL connection string
- `QDRANT_URL`: Qdrant server URL
- `LLM_API_KEY`: API key untuk LLM provider
- `STT_API_KEY`: API key untuk STT service

## 🔒 Security Notes

### Development vs Production

**⚠️ PENTING untuk Production:**

1. **Ganti Database Credentials**
   - Credentials di `docker-compose.yml` hanya untuk development
   - Production harus menggunakan strong passwords dan secrets management

2. **Environment Variables**
   - Jangan commit file `.env` (sudah di .gitignore)
   - Gunakan secrets management untuk production (AWS Secrets Manager, HashiCorp Vault, dll)

3. **API Keys**
   - Simpan API keys di environment variables
   - Rotate keys secara berkala
   - Gunakan key dengan least privilege

4. **CORS Configuration**
   - Development menggunakan `CORS_ORIGINS=["*"]`
   - Production harus specify exact domains: `CORS_ORIGINS=["https://yourdomain.com"]`

5. **Database Access**
   - Jangan expose PostgreSQL port ke public internet
   - Gunakan SSL/TLS untuk database connections
   - Implement proper firewall rules

## 🎯 TODO: Implementasi Adapters

Beberapa adapters masih berupa template dan perlu diimplementasikan:

1. **LLM Adapter** (`src/adapters/llm/llm_adapter.py`)
   - Integrasi dengan OpenAI/Anthropic/Local LLM
   - Implementasi streaming generation

2. **Vision Adapter** (`src/adapters/vision/vision_adapter.py`)
   - Integrasi model emotion detection
   - Integrasi model face recognition

3. **Speech Adapters** (`src/adapters/speech/`)
   - STT: Integrasi Whisper atau service lain
   - TTS: Integrasi TTS service

4. **RAG Adapter** (`src/adapters/qdrant/rag_adapter.py`)
   - Implementasi embedding generation
   - Implementasi document chunking

## 📝 License

[Specify your license]

## 👥 Contributors

[Add contributors]
