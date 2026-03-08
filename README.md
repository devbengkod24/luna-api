# VSA Backend API v2

Backend API untuk sistem Virtual Smart Assistant (VSA) dengan fitur:

- **Emotion Detection** — Deteksi emosi dari video frame
- **Face Recognition** — Pengenalan wajah pengguna
- **Agent RAG** — Agent dengan knowledge base
- **TTS** — Text-to-Speech
- **STT** — Speech-to-Text

Dibangun dengan **Clean Architecture**.

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn, Pydantic, Pydantic-Settings
- **Database:** PostgreSQL, SQLAlchemy 2, Alembic
- **Vector DB:** Qdrant
- **RAG & Agent:** LlamaIndex, LangChain, Ollama
- **Object Storage:** MinIO
- **Face:** InsightFace
- **STT:** Faster-Whisper
- **TTS**: MMS
- **Container:** Docker, Docker Compose

---

## Struktur Proyek

```
VSA-BE-FastAPIv2/
├── alembic/
│   ├── versions/
│   ├── env.py
│   └── script.py.mako
├── src/
│   ├── config.py
│   ├── domain/
│   │   └── entities/          # conversation, face, file, response, vision
│   ├── ports/                 # repositories, services, vision
│   ├── adapters/
│   │   ├── agent/
│   │   ├── persistence/       # database, models, repositories, mappers
│   │   ├── rag/
│   │   ├── speech/            # STT, TTS
│   │   ├── storage/           # MinIO
│   │   └── vision/            # face store, recognition, tracker, emotion
│   ├── use_cases/             # response, conversation, file, face
│   ├── drivers/
│   │   ├── main.py            # FastAPI app & lifespan
│   │   ├── dependencies.py
│   │   ├── rest/              # routers, schemas
│   │   └── realtime/          # WebSocket routers (e.g. vision)
│   ├── seed/                  # face_seeder, rag_seeder, __main__
│   ├── shared/                # constants, exceptions, utils
│   └── prompts/
├── storage/                   # faces, rag (isi di-ignore; folder tetap via .gitkeep)
├── tests/
├── requirements/              # base.txt, cpu.txt, gpu.txt
├── docker-compose.yml
├── Dockerfile.cpu
├── Dockerfile.gpu
├── alembic.ini
├── .env.example
└── README.md
```

---

## Penjelasan Layer


| Layer         | Path             | Deskripsi                                                                        |
| ------------- | ---------------- | -------------------------------------------------------------------------------- |
| **Domain**    | `src/domain/`    | Entitas bisnis dan tipe data inti; bebas dari framework dan infra.               |
| **Ports**     | `src/ports/`     | Interface (repository, service, vision) yang mendefinisikan kontrak antar layer. |
| **Adapters**  | `src/adapters/`  | Implementasi konkret dari ports (DB, Qdrant, MinIO, Ollama, InsightFace, dll).   |
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
cd VSA-BE-FastAPIv2
```

### 2. Environment

```bash
cp .env.example .env
# Sesuaikan .env (database, MinIO, Ollama, dll.)
```

### 3. Jalankan dengan Docker Compose

**CPU (default):**

```bash
docker compose up --build -d
```

**GPU:**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

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
docker compose run --rm fastapi python -m src.seed all    # face + rag
docker compose run --rm fastapi python -m src.seed face    # wajah saja
docker compose run --rm fastapi python -m src.seed rag     # RAG saja

# Lokal (setelah dependencies & env siap)
python -m src.seed all
python -m src.seed face
python -m src.seed rag
```

Format nama file wajah: `Nama_Nama__NIM.jpg` (nama dan NIM dipakai untuk metadata).