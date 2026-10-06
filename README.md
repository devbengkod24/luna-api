# Luna Backend API

Backend API untuk sistem Luna (Linked Unified Neural Assistant) dengan fitur:

- **Face Recognition** — Pendaftaran wajah dan indeks pencocokan (deteksi wajah live dan emosi berjalan di kiosk, luna-edge)
- **Agent RAG** — Agent dengan knowledge base dan web search
- **Turn WebSocket** — `/turn/stream`, satu giliran bicara dari kiosk dijawab dengan transkrip, token, dan audio
- **TTS** — Text-to-Speech
- **STT** — Speech-to-Text
- **Photobooth** — Penyimpanan foto dari kiosk di MinIO

Dibangun dengan **Clean Architecture**.

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn, Pydantic, Pydantic-Settings
- **Database:** PostgreSQL, SQLAlchemy 2, Alembic
- **Vector DB:** Qdrant
- **RAG & Agent:** LlamaIndex, LangChain, Ollama
- **Web search:** Exa
- **Object Storage:** MinIO
- **Face:** OpenCV YuNet + SFace
- **STT:** Deepgram (nova-3)
- **TTS**: Cartesia (MMS, Chatterbox, F5-TTS tersedia lewat `TTS_PROVIDER`)
- **Container:** Docker, Docker Compose

---

## Struktur Proyek

```
luna-api/
├── alembic/
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
├── src/
│   ├── config.py
│   ├── domain/
│   │   └── entities/          # conversation, face, file, photo, response, vision
│   ├── ports/                 # identifiers, prompts, repositories, services, vision
│   ├── adapters/
│   │   ├── agent/
│   │   ├── persistence/       # database, models, repositories, mappers
│   │   ├── rag/
│   │   ├── speech/            # STT, TTS
│   │   ├── storage/           # MinIO
│   │   └── vision/            # face store, recognition (YuNet), SFace embedder
│   ├── use_cases/             # response, conversation, file, face, photo
│   ├── drivers/
│   │   ├── main.py            # FastAPI app & lifespan
│   │   ├── dependencies.py
│   │   ├── rest/              # routers, schemas
│   │   └── realtime/          # WebSocket router /turn/stream
│   ├── seeder/                # face_seeder, rag_seeder, __main__
│   ├── shared/                # constants, exceptions, utils
│   └── prompts/
├── storage/                   # faces, rag, models (di-ignore git)
├── tests/unit/
├── requirements/              # base.txt, cpu.txt, gpu.txt
├── docker-compose.yml
├── docker-compose.gpu.yml
├── Dockerfile.cpu
├── Dockerfile.gpu
├── alembic.ini
└── README.md
```

---

## Penjelasan Layer


| Layer         | Path             | Deskripsi                                                                        |
| ------------- | ---------------- | -------------------------------------------------------------------------------- |
| **Domain**    | `src/domain/`    | Entitas bisnis dan tipe data inti; bebas dari framework dan infra.               |
| **Ports**     | `src/ports/`     | Interface (repository, service, vision) yang mendefinisikan kontrak antar layer. |
| **Adapters**  | `src/adapters/`  | Implementasi konkret dari ports (DB, Qdrant, MinIO, Ollama, OpenCV, dll).        |
| **Use cases** | `src/use_cases/` | Orkestrasi logika bisnis yang memakai ports.                                     |
| **Drivers**   | `src/drivers/`   | Entry point aplikasi: HTTP (REST), WebSocket, dan dependency injection.          |


---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (opsional, untuk development tanpa Docker)

### 1. Clone

```bash
git clone <repository-url>
cd luna-api
```

### 2. Konfigurasi

Konfigurasi ada di blok `environment` pada `docker-compose.yml` (database, MinIO, Ollama, dll.). API key dan `API_ACCESS_TOKEN` ada di `.env` (salin dari `.env.example`), bukan di compose.

### 3. Jalankan dengan Docker Compose

**CPU:**

```bash
docker compose up --build -d
```

**GPU:**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

---

- **API:** [http://localhost:8000](http://localhost:8000)
- **Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **PostgreSQL:** localhost:5432
- **Qdrant:** [http://localhost:6333](http://localhost:6333)
- **MinIO Console:** [http://localhost:9001](http://localhost:9001)

### 4. Migration

Upgrade database ke versi terbaru:

```bash
docker compose exec fastapi alembic upgrade head
```

### 5. Seeder

Isi data awal dari folder `storage/` (wajah dari `storage/faces/`, dokumen RAG dari `storage/rag/`).

```bash
# Di container
docker compose exec fastapi python -m src.seeder all    # face + rag
docker compose exec fastapi python -m src.seeder face    # wajah saja
docker compose exec fastapi python -m src.seeder rag     # RAG saja
```

Format nama file wajah: `Nama_Nama__NIM.jpg` (nama dan NIM dipakai untuk metadata).