# Keadaan Guardrails luna-api

> **Keadaan sekarang.** Dokumen ini memetakan semua penjagaan yang **sudah aktif**.
> Untuk peta tiga tahap dan tool yang direncanakan, lihat
> [`THREE_STAGES.md`](./THREE_STAGES.md). Untuk kriteria hasil akhir, lihat
> [`TARGET_OUTCOMES.md`](../product/TARGET_OUTCOMES.md). Untuk rencana pengerjaan, lihat
> [`IMPLEMENTATION_PLAN.md`](../engineering/IMPLEMENTATION_PLAN.md). Untuk kontrak kode, lihat
> [`IGARDRAIL_SPEC.md`](../engineering/IGARDRAIL_SPEC.md).

Dokumen ini memetakan semua penjagaan yang aktif dari dua pintu masuk — REST dan
WebSocket — sampai balasan diucapkan. Setiap klaim menunjuk modul dan tes yang
mengunci perilakunya, supaya perubahan pada satu layer tidak menghapus layer lain
tanpa sadar.

> **Bukan** daftar lengkap regex. Yang didokumentasikan adalah kategori pola,
> urutan eksekusi, biaya, dan batasnya. Daftar pola lengkap ada di
> `src/adapters/agent/input_guard.py`.

---

## 1. Alur end-to-end

### REST — `POST /api/v1/response/text`

```
request
  │
  ├─ BodySizeLimitMiddleware ......... tolak > MAX_REQUEST_BODY_BYTES (51 MB)
  │                                    sebelum CORS, routing, atau parsing
  ├─ CORSMiddleware .................. origin menurut CORS_ORIGINS
  ├─ request_id_middleware ........... X-Request-ID
  ├─ Depends(require_api_token) ...... 401 bila token salah/kosong
  ├─ Pydantic ResponseTextRequest .... user_input ≤ 10.000 char,
  │                                    conversation_id harus UUID,
  │                                    context.name/nim/emotion dibatasi
  ├─ require_response_services ....... 503 bila agent/DB tidak siap
  │
  ├─ ResponseTextUseCase.execute()
  │    ├─ FilePromptRepository ....... sanitasi UserContext → system prompt
  │    │                              (1 baris, tanpa control char, dipotong,
  │    │                               dibuang bila terbaca sebagai instruksi)
  │    └─ _stream_conversation()
  │         ├─ persist user message
  │         ├─ AgentService.generate_stream()
  │         │    ├─ refusal_for(user_input) ... jailbreak → harmful → tolak
  │         │    │                              (TokenEvent + AssistantEvent)
  │         │    ├─ create_agent (LangChain)
  │         │    │    ├─ wrap_model_call: strip_pseudo_tools_middleware
  │         │    │    └─ tools: search_web / rag_search
  │         │    │         └─ asyncio.timeout(AGENT_TOOL_TIMEOUT_SECONDS = 20 s)
  │         │    └─ idle timeout per event (AGENT_IDLE_TIMEOUT_SECONDS = 45 s)
  │         ├─ TTS worker (opsional)
  │         └─ DoneEvent setelah transaksi commit
  │
  └─ SSE ............................. SSE_IDLE_TIMEOUT_SECONDS = 60 s
                                       → ErrorEvent("Stream timeout")
```

### WebSocket — `/api/v1/turn/stream`

```
connect
  │
  ├─ token_accepted .................. close 1008 sebelum accept
  ├─ ENABLE_VOICE_WS / ENABLE_STT .... close 1011
  ├─ service container ............... close 1011 bila STT/agent belum siap
  ├─ accept
  │
  ├─ turn_begin ...................... turn_id baru; audio buffer di-reset
  ├─ binary frame .................... [type:1][ts:uint64 BE][pcm]
  │    ├─ hanya type 0x01
  │    └─ buffer ≤ VOICE_MAX_UTTERANCE_SECONDS (30 s) × sample_rate × 2
  │         (kelebihan dibuang; hanya log warning)
  ├─ text question (opsional) ........ ≤ CHAT_MAX_PROMPT_LENGTH (10.000 char)
  ├─ turn_end ........................ transkripsi → pipeline
  ├─ receive loop .................... FRAME_TIMEOUT_SECONDS = 5 s → `ping`
  │
  ├─ pipeline ........................ batas 2 × AGENT_IDLE_TIMEOUT_SECONDS (90 s)
  │    └─ timeout → kirim `error` bertanda turn_id (bukan diam)
  │
  └─ done ............................ setelah commit transaksi
```

Catatan: `BodySizeLimitMiddleware` hanya menangani `scope["type"] == "http"`.
Ukuran frame WebSocket tunggal tidak dibatasi middleware ini (lihat §6).

### Urutan layer per turn

| # | Layer | Berlaku di | Sifat |
|---|---|---|---|
| 1 | Body size limit | REST | Hard reject (413) |
| 2 | API token | REST + WS | Hard reject (401 / close 1008) |
| 3 | Feature flag & service readiness | WS | Hard reject (close 1011) |
| 4 | Schema validation (Pydantic) | REST + WS (sebagian) | Hard reject (422 / `error`) |
| 5 | Prompt context sanitization | REST + WS | Drop field, lanjut |
| 6 | Input guard (jailbreak/harmful) | REST + WS | Refusal sebagai jawaban |
| 7 | Tool timeout | REST + WS | Tool gagal, jawaban lanjut |
| 8 | Agent idle timeout | REST + WS | `ErrorEvent` / `error` |
| 9 | Stream idle timeout | REST (SSE) | `ErrorEvent` |
| 10 | Pipeline ceiling | WS | `error` bertanda turn |
| 11 | Pseudo-tool mitigation | REST + WS | Retry tanpa tools |
| 12 | Output moderation | — | **Tidak ada** |

---

## 2. Tabel layer guard

| Layer | Modul | Mekanisme | Tech stack | Biaya / limit | Tes pengunci |
|---|---|---|---|---|---|
| **Body size** | `src/drivers/middleware.py` (`BodySizeLimitMiddleware`) | ASGI middleware; cek `Content-Length`, lalu hitung byte saat body tiba | Starlette ASGI, `json` | Default 51 MB; tolak 413 sebelum parsing | `test_api_boundary.py::test_an_oversized_body_is_refused_before_it_is_parsed`, `::test_an_oversized_body_without_a_length_is_refused_too` |
| **API token** | `src/drivers/security.py` (`token_accepted`, `require_api_token`) | Satu token bersama; `Authorization: Bearer` atau `?token=`; perbandingan constant-time | `secrets.compare_digest` | O(1); kosong = API terbuka (warning saat startup) | `test_api_boundary.py::test_every_api_route_refuses_a_caller_without_the_token`, `::test_the_token_works_as_a_header_and_as_a_query_parameter`; `test_turn_stream_events.py::test_the_turn_socket_refuses_a_kiosk_without_the_token` |
| **CORS** | `src/drivers/main.py` | `CORSMiddleware`; `allow_credentials` dimatikan bila origin `*` | Starlette | Default `*` | Tidak ada tes khusus |
| **Schema limits** | `src/drivers/rest/schemas/*.py` | Pydantic `Field(max_length=...)`, validator ukuran & format | Pydantic v2 | Prompt 10.000 char; audio 10 MB; face 10 MB & 100 item; file 50 MB + ekstensi; foto 10 MB + magic JPEG | `test_api_boundary.py` (UUID & body), validasi schema tidak punya tes khusus per-field |
| **Context sanitization** | `src/adapters/prompts/file_prompt_repository.py` (`_context_value`) | Buang control char, jadikan satu baris, potong, lalu jalankan `is_jailbreak`/`is_harmful`; field yang gagal dibuang | `re`, stdlib; memakai ulang input guard | name ≤ 100, nim ≤ 20, emotion ≤ 20 | `test_agent_guard.py::test_a_name_from_the_client_cannot_carry_instructions_into_the_prompt`, `::test_an_ordinary_name_still_reaches_the_prompt` |
| **Input guard — jailbreak** | `src/adapters/agent/input_guard.py` (`_JAILBREAK`) | Regex atas teks lowercased + satu aturan case-sensitive (`DAN`) | `re` (stdlib) | ~0,006 ms panggilan (300 iterasi, mesin dev); 0 model call | `test_agent_guard.py::test_the_guard_refuses_a_jailbreak_and_harmful_requests`, `::test_the_narrowed_patterns_still_catch_the_attacks_they_were_for` |
| **Input guard — harmful** | `src/adapters/agent/input_guard.py` (`_HARMFUL`) | Regex kategori: senjata/kekerasan, siber, konten dewasa/CSAM, kriminal/penipuan, zat | `re` (stdlib) | Sama seperti di atas | `test_agent_guard.py::test_the_guard_refuses_a_jailbreak_and_harmful_requests` |
| **Refusal path** | `src/adapters/agent/agent_service.py` (`generate_stream`) | Refusal dikirim sebagai `TokenEvent` + `AssistantEvent`, lalu dipersist seperti jawaban biasa | LangChain event stream | Satu turn penuh tetap dipersist; history tidak berlubang | `test_agent_guard.py::test_a_refusal_is_persistable_as_the_assistant_turn` |
| **Tool timeout** | `src/adapters/agent/agent_service.py` (`search_web`, `rag_search`) | `asyncio.timeout` membungkus tiap tool call | `asyncio` | 20 s (`AGENT_TOOL_TIMEOUT_SECONDS`) | `test_agent_guard.py::test_a_hung_web_search_is_cut_off_at_the_tool_timeout` |
| **Agent idle timeout** | `src/use_cases/response_use_case.py` (`_llm_producer`) | Timeout per event dari generator agent | `asyncio` | 45 s (`AGENT_IDLE_TIMEOUT_SECONDS`) | **Tidak ada tes langsung** untuk jalur `Agent timeout` (lihat gap 9) |
| **SSE idle timeout** | `src/drivers/rest/routers/response.py` (`_reply_stream`) | Timeout per event SSE | `asyncio` | 60 s (`SSE_IDLE_TIMEOUT_SECONDS`) | Tidak ada tes khusus |
| **WS pipeline ceiling** | `src/drivers/realtime/routers/turn.py` (`_run_pipeline_bounded`) | `asyncio.timeout` + kirim `error` bertanda `turn_id` | `asyncio` | 2 × 45 s = 90 s | `test_turn_stream_events.py::test_a_pipeline_that_times_out_tells_the_kiosk` |
| **WS audio cap** | `src/drivers/realtime/routers/turn.py` (`TurnSocket.handle_audio`) | Buffer per turn dibatasi; kelebihan dibuang | Python `bytearray` | 30 s × 16 kHz × 2 byte ≈ 960 KB | Tidak ada tes khusus |
| **WS typed-question cap** | `src/drivers/realtime/routers/turn.py` (`_end`) | Panjang teks dicek sebelum pipeline | Python | 10.000 char | Tidak ada tes khusus |
| **Pseudo-tool mitigation** | `src/adapters/agent/middlewares/pseudo_tool_mitigation.py` | Filter `tool_calls` fiktif; retry sekali tanpa tools bila isi JSON sampah | LangChain `@wrap_model_call` | Satu retry model call (hanya saat terjadi) | Tidak ada tes khusus |
| **Error shielding** | `src/drivers/main.py` | Handler generik 500; `DEBUG` default `False` | FastAPI exception handler | O(1) | `test_api_boundary.py::test_debug_is_off_unless_asked_for` |
| **Face match floor** | `src/use_cases/face_use_case.py` | `threshold = max(request, FACE_MATCH_MIN_THRESHOLD)` | Python | Default 0,3 | `test_api_boundary.py::test_a_face_match_cannot_lower_the_threshold_below_the_floor` |
| **Filename sanitization** | `src/drivers/rest/schemas/file_schemas.py` | Buang `..`, `/`, `\`, NUL; batasi 255 char; whitelist ekstensi | Python `pathlib` | — | Tidak ada tes khusus |

---

## 3. Tech stack per layer

| Layer | Stack | Catatan |
|---|---|---|
| Boundary HTTP | FastAPI + Starlette ASGI middleware | Body limit berjalan sebelum framework menyentuh body |
| Validasi input | Pydantic v2 | `Field`, `field_validator`, `model_validator` |
| Auth | `secrets.compare_digest` (stdlib) | Satu token bersama, bukan akun |
| Input guard | `re` (stdlib) | Tanpa model, tanpa dependency tambahan |
| Prompt policy | File teks `src/prompts/*.txt` | Bukan hard guard — instruksi ke model |
| Agent | LangChain `create_agent` + `@wrap_model_call` middleware | Middleware dipakai untuk mitigasi pseudo-tool |
| Timeout | `asyncio.timeout` / `asyncio.wait_for` | Tool, idle stream, pipeline WS |
| Transport | SSE (REST) + WebSocket (turn) | `done` dikirim setelah commit |

### Keputusan: NeMo Guardrails dihapus

NeMo Guardrails pernah membungkus input guard. Yang dijalankan tetap regex yang
sama — tidak ada LLM rail atau dialogue rail tambahan — tetapi biayanya
~5–10 ms per turn (dibanding ~0,02 ms untuk regex saja saat itu), menambah
dependency besar, dan refusal-nya berbunyi sebagai bot lain ("Bengbot").
Sekarang regex dipanggil langsung oleh `LangchainAgentService.generate_stream`.

Bukti:

- `src/adapters/agent/input_guard.py` — docstring keputusan.
- `tests/unit/test_agent_guard.py::test_nemo_guardrails_is_no_longer_loaded`
  memastikan modul `nemoguardrails` tidak ikut ter-import.
- `tests/latency_probe.py` bagian `guardrail` mengukur `refusal_for` langsung.

---

## 4. Biaya dan limit (ringkas)

| Parameter | Default | Efek |
|---|---|---|
| `MAX_REQUEST_BODY_BYTES` | 51 MB | Batas body REST sebelum parsing |
| `CHAT_MAX_PROMPT_LENGTH` | 10.000 char | `user_input` REST dan teks WS |
| `CHAT_MAX_AUDIO_BASE64_SIZE` | 10 MB | `audio_input` REST |
| `AGENT_TOOL_TIMEOUT_SECONDS` | 20 s | Satu tool call (`search_web`, `rag_search`) |
| `AGENT_IDLE_TIMEOUT_SECONDS` | 45 s | Jarak antar-event dari agent |
| `SSE_IDLE_TIMEOUT_SECONDS` | 60 s | Jarak antar-event SSE |
| `FRAME_TIMEOUT_SECONDS` | 5 s | WS tanpa pesan → `ping` |
| `VOICE_MAX_UTTERANCE_SECONDS` | 30 s | Buffer audio satu turn |
| `DEEPGRAM_TIMEOUT_SECONDS` | 15 s | Transkripsi |
| `FACE_MATCH_MIN_THRESHOLD` | 0,3 | Lantai threshold `/face/match` |
| `RAG_MAX_FILE_SIZE` | 50 MB | Upload dokumen RAG |
| `RAG_ALLOWED_EXTENSIONS` | `.pdf .txt .docx .doc .md` | Whitelist ekstensi |
| `CHAT_HISTORY_LIMIT` | 20 pesan | Window konteks yang dikirim ke model |
| `OLLAMA_MAX_TOKENS` | 1.024 | Batas output model |
| `CORS_ORIGINS` | `*` | Origin yang diizinkan |

Semua nilai dibaca dari `src/config.py` (Pydantic `Settings`).

---

## 5. Gap yang diketahui

| # | Gap | Dampak | Status |
|---|---|---|---|
| 1 | **Tidak ada output moderation** | Jawaban model langsung dikirim ke SSE/TTS. Bila model menghasilkan konten berbahaya atau membocorkan prompt, tidak ada layer kode yang menghentikannya — hanya instruksi di `base.txt`. | Terbuka |
| 2 | **Tidak ada rate limit** | Satu token bersama, tanpa batas per kiosk/IP. Siapa pun yang memegang token bisa memicu turn sebanyak yang diinginkan; biaya LLM/STT/TTS mengikuti. `FRAME_TIMEOUT_SECONDS` hanya menjaga koneksi tetap hidup, bukan membatasi laju. | Terbuka |
| 3 | **Regex bisa diparaphrase** | Guard berbasis kata kunci. Serangan yang tidak memakai pola yang terdaftar lolos. Tidak ada klasifikasi semantik. | Diterima (trade-off latency & dependency) |
| 4 | **Guard hanya pada teks user** | Retrieved context (RAG) dan hasil web tidak melewati guard. `CLAUDE.md:112` meminta retrieved context diperlakukan sebagai data, tetapi `src/prompts/*.txt` tidak memuat instruksi itu dan tidak ada pemeriksaan kode — jadi ini konvensi dokumen, bukan guard. | Terbuka |
| 5 | **Frame WebSocket tidak dibatasi per pesan** | `BodySizeLimitMiddleware` melewati `scope["type"] != "http"`. Batas yang ada hanya buffer total per turn; frame tunggal bergantung pada default server (`ws-max-size`). | Terbuka |
| 6 | **Audio berlebih dibuang tanpa memberi tahu kiosk** | `handle_audio` hanya `logger.warning`; kiosk tidak menerima event bahwa sebagian audio hilang. | Terbuka |
| 7 | **CORS default `*`** | Bila token bocor, origin mana pun bisa memanggil API. `allow_credentials` dimatikan saat `*`, tetapi token tetap bisa dikirim eksplisit. | Terbuka |
| 8 | **Refusal tidak diukur** | Tidak ada metrik/counter berapa kali guard menolak, sehingga efektivitas dan false positive tidak terlihat. | Terbuka |
| 9 | **Beberapa guard tidak punya tes** | CORS, SSE idle timeout, agent idle timeout, audio cap, typed-question cap, pseudo-tool middleware, filename sanitization. Perilakunya ada di kode tetapi tidak dikunci — daftar ini adalah gabungan semua baris §2 yang bertanda "Tidak ada tes khusus". | Terbuka |

---

## 6. Verifikasi

```bash
# Guard input, sanitasi prompt, timeout tool, refusal persistable
pytest tests/unit/test_agent_guard.py

# Boundary REST: token, body size, UUID, threshold, debug
pytest tests/unit/test_api_boundary.py

# Kontrak event WS: turn_id, timeout, token
pytest tests/unit/test_turn_stream_events.py

# Semua unit test
pytest
```

Hasil terakhir dijalankan (2026-10-19, mesin dev, `.venv-verify`, Python 3.11):

```
pytest tests/unit/test_agent_guard.py tests/unit/test_api_boundary.py
21 passed, 1 warning in 12.36s

pytest
76 passed, 1 warning in 12.37s
```

Di mesin dev tanpa container, `test_agent_guard.py` dan `test_api_boundary.py`
memerlukan env minimal (`POSTGRES_*`, `MINIO_*`) dan `torch` (dipakai
`tts_service.py`). Tanpa itu pytest gagal saat collection, bukan saat tes
berjalan.

Pengukuran biaya guard:

```bash
python tests/latency_probe.py     # bagian "guardrail"
```

---

## 7. Menambah guard baru

1. Taruh di layer yang benar: boundary (middleware/schema), prompt (sanitasi),
   agent (input/output), atau transport (timeout).
2. Tambahkan tes yang mengunci **kedua arah**: serangan ditolak, permintaan
   wajar lolos. Contoh yang sudah ada:
   `test_ordinary_indonesian_is_not_mistaken_for_a_jailbreak` vs
   `test_the_narrowed_patterns_still_catch_the_attacks_they_were_for`.
3. Perbarui tabel §2 dan §5 di dokumen ini.
4. Jangan menyalin daftar regex ke dokumen publik atau log.
