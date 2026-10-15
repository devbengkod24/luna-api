# Luna Backend API

Backend untuk sistem **Luna** (Linked Unified Neural Assistant) — asisten virtual
kampus UDINUS. Dibangun dengan **Clean Architecture** di atas FastAPI.

## Fitur

| Fitur | Ringkas |
|---|---|
| **Chat Agent** | LLM + RAG (knowledge base) + web search, jawaban streaming |
| **STT** | Speech-to-Text (Deepgram nova-3) — suara dari kiosk jadi teks |
| **TTS** | Text-to-Speech (Cartesia) — jawaban jadi audio |
| **Turn WebSocket** | `/turn/stream` — satu giliran bicara dari kiosk, dijawab transkrip + token + audio |
| **Face Recognition** | Pendaftaran & pencocokan wajah (indeks cloud) |
| **Photobooth** | Foto dari kiosk disimpan di MinIO |

> **Persepsi bukan di sini.** Kamera, tracking, emosi, VAD dan active-speaker
> semuanya berjalan di kiosk (**luna-edge**). Service ini tidak pernah menerima
> frame video. Yang menyeberang hanya **audio** satu giliran bicara. Pengecualian:
> **indeks wajah** ada di cloud, jadi enrolment & pencocokan ada di sini.

---

## Quick Start

Enam langkah, berurutan. **Langkah 2 dan 4 tidak boleh dibalik** — kalau
`fastapi` dinyalakan sebelum model embedding ada, container crash dan
gejalanya menyesatkan (lihat [Troubleshooting](#troubleshooting)).

**Alur singkat:**

1. Clone — 2. Container pendukung + tarik model Ollama — 3. Buat `.env` —
4. Nyalakan `fastapi` — 5. Migration — 6. Seeder (opsional) — 7. Verifikasi

### Prerequisites

- **Docker & Docker Compose**
- Ruang disk lega — image ~14 GB + build cache ~8 GB, di luar model Ollama
- Python 3.11+ (opsional, untuk development tanpa Docker)

### 1. Clone

```bash
git clone <repository-url>
cd luna-api
```

### 2. Container pendukung + tarik model Ollama

Model LLM dan embedding **tidak ikut di image** — harus di-*pull* ke volume
`ollama_models`. Ini **wajib sebelum `fastapi` dinyalakan**: `rag_service`
meng-embed satu teks saat startup, dan kalau model embedding belum ada,
`fastapi` gagal start.

```bash
docker compose up -d postgres minio qdrant ollama

# Embedding — WAJIB. Tanpa ini fastapi tidak bisa start.
docker compose exec ollama ollama pull embeddinggemma:latest

# LLM — pilih salah satu:
#   a) model cloud (butuh akun Ollama, login sekali saja)
docker compose exec ollama ollama signin
docker compose exec ollama ollama pull gpt-oss:120b-cloud

#   b) model lokal (tanpa akun, lebih berat, harus mendukung tool-calling)
# docker compose exec ollama ollama pull qwen3:8b
# lalu set OLLAMA_LLM_MODEL: "qwen3:8b" di docker-compose.yml
```

Verifikasi keduanya sudah ada:

```bash
docker compose exec ollama ollama list
```

> Login Ollama tersimpan di volume `ollama_models` — **cukup sekali**, tidak
> perlu `signin` lagi setelah restart atau rebuild.

### 3. Buat `.env`

`.env` tidak ada di git dan **tidak** dibaca lewat `env_file`; `config.py`
membacanya dari bind mount `- .:/app`. Salin dari template:

```bash
cp .env.example .env
```

Isi empat variabel berikut sesuai kebutuhanmu:

| Variabel | Wajib untuk | Untuk apa | Kalau kosong |
|---|---|---|---|
| `DEEPGRAM_API_KEY` | fitur voice | STT (nova-3) | `/turn/stream` **menolak kiosk** (tidak ada fallback lokal di branch ini) |
| `CARTESIA_API_KEY` | fitur voice | TTS | TTS mati — Luna tidak bersuara |
| `EXA_API_KEY` | opsional | tool `search_web` | Agent kehilangan tool (lihat catatan di bawah) |
| `API_ACCESS_TOKEN` | ekspos publik | kunci `/api/v1` + turn socket | Semua route terbuka — aman **hanya** di localhost |

**Sudah terisi, tidak perlu diubah:**

- `CARTESIA_VOICE_ID` → ada di `docker-compose.yml`
- Kredensial Postgres & MinIO → nilai lokal, ada di `docker-compose.yml`

**`API_ACCESS_TOKEN`** kamu karang sendiri. Kalau luna-edge dijalankan dari mesin
lain, ia harus membawa nilai yang **sama** (`export LUNA_API_TOKEN=<nilai sama>`):

```bash
openssl rand -hex 32
```

> ⚠️ `.env` sudah masuk `.gitignore`. **Jangan** menaruh nilainya di
> `docker-compose.yml` — file itu di-commit.

### 4. Nyalakan `fastapi`

Setelah model Ollama ada **dan** `.env` terisi:

```bash
# CPU
docker compose up --build -d

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

### 5. Migration

Membuat tabel database. **Wajib di mesin baru** — tabel tidak dibuat otomatis
oleh aplikasi.

```bash
docker compose exec fastapi alembic upgrade head
```

> Jangan mengedit migration yang sudah pernah diterapkan — buat migration baru.

### 6. Seeder (opsional — isi data awal)

`storage/` tidak ada di git (data lokal, bukan bagian repo). Kalau mau mengisi,
siapkan dulu:

- `storage/faces/` — foto wajah, format `Nama_Nama__NIM.jpg`
  (dua underscore antara nama dan NIM; keduanya jadi metadata)
- `storage/rag/` — dokumen knowledge base

Lalu jalankan:

```bash
docker compose exec fastapi python -m src.seeder all    # face + rag
docker compose exec fastapi python -m src.seeder face    # wajah saja
docker compose exec fastapi python -m src.seeder rag     # RAG saja
```

Folder yang belum ada hanya dilewati dengan WARNING — bukan error, jadi langkah
ini aman dijalankan walau belum ada data:

```
WARNING  Face seed skipped: directory not found (storage/faces)
WARNING  RAG seed skipped: directory not found (storage/rag)
```

Setelah knowledge base terisi, aktifkan RAG sebagai tool di `docker-compose.yml`:

```yaml
ENABLE_RAG_TOOL: true
```

### 7. Verifikasi

```bash
curl -s http://localhost:8000/health   # {"status":"ok"}
curl -s http://localhost:8000/ready    # {"status":"ok","db_ready":true,"rag_ready":true}
```

Pastikan tidak ada `ERROR` di log startup:

```bash
docker compose logs fastapi | grep -iE "initialized|disabled|not set"
```

Uji chat end-to-end (SSE, tanpa TTS):

```bash
docker compose exec fastapi curl -sN -X POST \
  http://localhost:8000/api/v1/response/text \
  -H "Content-Type: application/json" \
  -d '{"user_input":"Halo","character":0,"enable_tts":false}'
```

Harusnya mengalir `{"type":"token",...}` berturut-turut lalu `{"type":"done"}`.

---

## Endpoint & Port

| Layanan | Alamat |
|---|---|
| **API** | [http://localhost:8000](http://localhost:8000) |
| **Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **PostgreSQL** | `localhost:5432` |
| **Qdrant** | [http://localhost:6333](http://localhost:6333) |
| **MinIO Console** | [http://localhost:9001](http://localhost:9001) |

Ollama (`11434`) dan Qdrant gRPC (`6334`) serta MinIO S3 (`9000`) hanya di
jaringan internal Docker. Untuk mengaksesnya dari host, buka komentar port yang
relevan di `docker-compose.yml`.

---

## Arsitektur

### Struktur Proyek

```
luna-api/
├── alembic/                   # migration (versions/, env.py)
├── src/
│   ├── config.py              # Settings (Pydantic) — sumber kebenaran konfigurasi
│   ├── domain/                # entitas bisnis murni, tanpa framework
│   │   └── entities/          # conversation, face, file, photo, response, vision
│   ├── ports/                 # kontrak antar layer (identifiers, prompts,
│   │                          #   repositories, services, vision)
│   ├── adapters/              # implementasi konkret dari ports
│   │   ├── agent/             # LangChain + Ollama
│   │   ├── persistence/       # SQLAlchemy: database, models, repositories, mappers
│   │   ├── rag/               # LlamaIndex + Qdrant
│   │   ├── speech/            # STT (Deepgram), TTS (Cartesia/MMS/Chatterbox/F5)
│   │   ├── storage/           # MinIO
│   │   └── vision/            # YuNet + SFace, face store (Qdrant)
│   ├── use_cases/             # orkestrasi bisnis — response, conversation,
│   │                          #   file, face, photo
│   ├── drivers/               # entry point
│   │   ├── main.py            # app FastAPI + lifespan
│   │   ├── dependencies.py    # DI container
│   │   ├── rest/              # router + schema (Pydantic)
│   │   └── realtime/          # WebSocket /turn/stream
│   ├── seeder/                # CLI: face_seeder, rag_seeder
│   ├── shared/                # constants, exceptions, utils
│   └── prompts/               # template prompt (base, luna, tool-policy-*)
├── storage/                   # faces, rag, models (git-ignored)
├── tests/unit/                # uji use-case dengan fake in-memory
├── requirements/              # base.txt, cpu.txt, gpu.txt
├── docker-compose.yml         # + docker-compose.gpu.yml
├── Dockerfile.cpu             # + Dockerfile.gpu
└── alembic.ini
```

### Layer & Arah Dependensi

Dependensi hanya boleh mengarah **ke dalam**. `use_cases` tidak boleh mengimpor
`fastapi`, `sqlalchemy`, `qdrant_client`, `minio`, `langchain*`, `llama_index*`,
atau `cv2`.

```
   drivers ──▶ use_cases ──▶ ports ◀── adapters
                               │
                               ▼
                             domain
```

| Layer | Path | Peran |
|---|---|---|
| **Domain** | `src/domain/` | Entitas & tipe inti. Bebas framework dan infra. |
| **Ports** | `src/ports/` | Interface (repository, service, vision) — kontrak antar layer. |
| **Adapters** | `src/adapters/` | Implementasi konkret: DB, Qdrant, MinIO, Ollama, OpenCV, dll. |
| **Use cases** | `src/use_cases/` | Orkestrasi logika bisnis, memakai ports. |
| **Drivers** | `src/drivers/` | Entry point: REST, WebSocket, DI, lifespan. |

**Wiring & lifecycle.** `AppContainer` dibangun oleh `initialize_container()` di
`src/drivers/dependencies.py`. Lifespan di `main.py` menyimpannya di
`app.state.container`, lalu meng-*await* `rag_service.initialize()` dan
`face_service.initialize()`. Shutdown menutup `db_connection`, `rag_service` dan
`face_service` — adapter baru yang punya `initialize()` **wajib** punya `close()`
dan didaftarkan di lifespan.

**Session DB.** `get_db_session()` memberi session per-request, **commit saat
sukses dan rollback saat exception**. Use case tidak boleh `commit`/`rollback`
sendiri.

**Async.** I/O DB sudah async (asyncpg). Semua yang CPU/GPU-bound (MinIO, TTS
lokal, face, STT) harus lewat `asyncio.to_thread()` atau executor — kalau
inline, satu sintesis membekukan turn socket dan seluruh request lain.

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn, Pydantic, Pydantic-Settings
- **Database:** PostgreSQL, SQLAlchemy 2, Alembic
- **Vector DB:** Qdrant
- **LLM / RAG / Agent:** LlamaIndex, LangChain, Ollama
- **Web search:** Exa
- **Object storage:** MinIO
- **Face:** OpenCV YuNet + SFace
- **STT:** Deepgram (nova-3)
- **TTS:** Cartesia — alternatif lokal: MMS, Chatterbox, F5-TTS lewat `TTS_PROVIDER`
- **Container:** Docker, Docker Compose

---

## Troubleshooting

### `Application startup failed. Exiting.` — `/health` tidak merespons

**Penyebab paling umum: model embedding Ollama belum di-pull.** `rag_service`
meng-embed `"test"` saat startup; kalau model tidak ada, seluruh app gagal
start. Karena app tidak pernah listen, `/health` membalas **kosong**
(connection reset), bukan pesan error — makanya gejalanya membingungkan.

```bash
docker compose logs fastapi | tail -20   # cari: ResponseError: model "..." not found
docker compose exec ollama ollama list   # pastikan embeddinggemma ada
docker compose up -d --force-recreate fastapi
```

### Error `model "embeddinggemma:latest" not found` saat seeder

Race: seeder dijalankan sebelum `ollama pull` selesai. Tunggu pull selesai, lalu
jalankan ulang seeder.

### Setelah mengubah `.env`, adapter masih mati

`config.py` membaca `.env` saat proses start, bukan saat runtime. Recreate:

```bash
docker compose up -d --force-recreate fastapi
```

### `search_web unavailable` / agent tanpa tool

`EXA_API_KEY` kosong. Sementara itu `ENABLE_RAG_TOOL: false`, jadi agent
**tidak punya tool sama sekali** dan menjawab dari bobot LLM saja. Perbaiki
dengan mengisi `EXA_API_KEY`, atau isi `storage/rag/` lalu set
`ENABLE_RAG_TOOL: true`.

### Warning `API_ACCESS_TOKEN is not set`

Bukan error, tapi berarti semua route terbuka. Aman selama port 8000 hanya
loopback (`127.0.0.1`). **Wajib diisi** sebelum diakses dari mesin lain atau
lewat tunnel.

### Warning `Face seed skipped` / `RAG seed skipped`

Wajar kalau `storage/faces/` atau `storage/rag/` belum diisi. Lihat langkah 6.