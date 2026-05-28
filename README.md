# Luna Backend API

Backend API untuk sistem Luna (Linked Unified Neural Assistant) dengan fitur:

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
luna-api/
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
cd luna-api
```

### 2. Environment

```bash
cp .env.example .env
# Sesuaikan .env (database, MinIO, Ollama, dll.)
```

### 3. Jalankan dengan Docker Compose

#### Standard (Windows & Linux native ext4/xfs):

**CPU:**

```bash
docker compose up --build -d
```

**GPU:**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

#### FUSE-based Linux systems (e.g., `/mnt/adata`, WSL2):

If PostgreSQL fails with `wrong ownership` error, use the FUSE workaround:

**CPU:**

```bash
docker compose -f docker-compose.yml -f docker-compose.fuse.yml up --build -d
```

**GPU:**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.fuse.yml up --build -d
```

> **Note:** `docker-compose.fuse.yml` uses `tmpfs` for PostgreSQL data. Data does **not** persist across container restarts but avoids ownership errors on FUSE mounts.

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