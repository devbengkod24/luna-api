# Rencana Implementasi Guardrails

> Dokumen kerja. Untuk keadaan yang sudah aktif, lihat
> [`guardrails.md`](./guardrails.md). Untuk bentuk kode, lihat
> [`igardrail-spec.md`](./igardrail-spec.md).

## Ringkasan

Guardrails luna-api saat ini punya sembilan gap yang terdokumentasi di
`guardrails.md` §5. Rencana ini menutupnya bertahap, satu tool runtime per
tahap, dengan pengukuran di setiap langkah.

**Prinsip:** satu tahap = satu perubahan yang bisa diukur dan bisa dimatikan.
Jangan mengerjakan dua tahap sekaligus.

## Tool yang dipakai

| # | Tool | Peran | Tahap | Lisensi | Runtime? |
|---|---|---|---|---|---|
| 1 | Regex input guard *(sudah ada)* | jailbreak, harmful | 0 | — | selalu |
| 2 | Canary token | deteksi kebocoran system prompt | 0 | — | selalu |
| 3 | Redaksi log (`logging.Filter`) | cegah PII/secret di log | 0 | — | selalu |
| 4 | DeepEval | ukur ASR & over-refusal | 0 | Apache 2.0 | CI saja |
| 5 | Prompt Guard 2 86M | injection di RAG/web | 1 | Llama 4 Community | kondisional |
| 6 | Regex + checksum PII | NIK, NPWP, telepon, email, kartu | 2 | — | selalu |
| 7 | Presidio | PII nama & variasi | 2 | MIT | flag |
| 8 | Qwen3Guard 0.6B | moderasi semantik | 3 | Apache 2.0 | flag |
| 9 | Langfuse | trace & refusal rate | 4 | MIT core | async |
| 10 | Garak + Promptfoo | red team | 4 | Apache 2.0 / MIT | CI |
| 11 | gpt-oss-safeguard 20B | audit policy offline | 4 | Apache 2.0 | offline |

**Tidak dipakai:** LLM Guard (archived), DeBERTa v2 (archived + English-only),
NeMo Guardrails (tidak menambah deteksi, sudah dihapus), Guardrails AI
(type-checker), Rebuff (tidak aktif).

---

## Peta tool → file → fungsi

| Tahap | Tool | File | Fungsi | Stage |
|---|---|---|---|---|
| 0 | Canary | `adapters/prompts/file_prompt_repository.py` | `_context_value` | prompt |
| 0 | Redaksi log | `drivers/logging.py` | filter baru | log |
| 0 | DeepEval | `tests/eval/` (baru) | CI | ukur |
| 1 | Prompt Guard 2 86M | `adapters/agent/agent_service.py` | `rag_search` | RETRIEVED |
| 1 | Prompt Guard 2 86M | `adapters/agent/agent_service.py` | `search_web` | RETRIEVED |
| 2 | Regex PII + Presidio | `use_cases/response_use_case.py` | sebelum `out_q.put` | OUTPUT |
| 3 | Qwen3Guard | `use_cases/response_use_case.py` | input + audit output | INPUT / audit |
| 4 | Langfuse | `drivers/` | OTel exporter | trace |
| 4 | Garak/Promptfoo | `.github/workflows/` | CI | red team |
| 4 | gpt-oss-safeguard | job terpisah | mingguan | audit |

Nomor baris bisa bergeser; nama fungsi tidak.

---

## Tahap 0 — Prasyarat (tanpa tool baru)

Tujuan: menciptakan **titik sisip** dan **alat ukur**. Tidak ada model baru.

### 0.1 Guard sebelum `out_q.put`

**File:** `src/use_cases/response_use_case.py`, `_llm_producer`

**Masalah.** Urutan sekarang mengirim token ke client sebelum guard apa pun:

```python
await out_q.put(event)                      # TokenEvent keluar ke client

if isinstance(event, TokenEvent):
    if enable_tts and event.content:
        tts_buffer += event.content
        sentences, tts_buffer = _drain_sentences(tts_buffer)
        for sentence in sentences:
            await sentence_q.put(sentence)  # baru di sini TTS
```

Chat menampilkan teks sebelum guard berjalan.

**Perubahan.** Buffer kalimat dulu, guard, baru kirim ke dua tujuan. Bentuk
lengkap ada di [`igardrail-spec.md`](./igardrail-spec.md) §5.3.

**Konsekuensi yang diterima:**

| Aspek | Sebelum | Sesudah |
|---|---|---|
| Chat | token per token | per kalimat |
| Voice | per kalimat | tidak berubah |
| Latensi teks pertama | ~0 | +1 batas kalimat + guard |
| Kebocoran di chat | mungkin | tertutup |

**Selesai jika:** chat dan voice sama-sama melewati guard; `AssistantEvent`
menggunakan teks ter-guard; tes use case mengunci perilaku ini.

### 0.2 Canary token

**File:** `src/adapters/prompts/file_prompt_repository.py`

Sisipkan token acak per proses ke system prompt. Periksa token itu di output;
jika muncul, system prompt bocor. Murah, deterministik.

**Selesai jika:** ada tes yang membuktikan token di output memicu penolakan,
dan tes yang membuktikan jawaban normal tidak terpengaruh.

### 0.3 Redaksi log

**File:** `src/drivers/logging.py` atau `src/drivers/log_redaction.py`

Aturan "jangan log PII" sudah ada di `CLAUDE.md` tetapi belum ditegakkan kode.
Filter menolak nama, NIM, base64 panjang, dan DSN.

**Selesai jika:** log yang mengandung NIK/base64/DSN tidak lolos.

### 0.4 DeepEval + golden set

**File:** `tests/eval/` (baru), tidak masuk jalur request.

```
tests/eval/
  golden/
    must_answer.yaml      # 50 pertanyaan kampus
    must_refuse.yaml      # 30 permintaan terlarang
    ambiguous.yaml        # 30 pertanyaan edukatif
    injection.yaml        # 20 serangan, termasuk lewat dokumen RAG
  test_guardrails.py
```

**Kenapa di tahap 0:** setiap tahap berikutnya dibandingkan terhadap baseline.
Tanpa baseline, tidak bisa membuktikan tahap 1 atau 3 lebih baik.

| Metrik | Ambang |
|---|---|
| ASR | harus turun |
| Over-refusal | naik < 2–3% |
| Latensi p95 | dalam target voice |

**Selesai jika:** baseline tercatat dan tes berjalan di CI.

---

## Tahap 1 — Injection di konten eksternal

Satu-satunya tool runtime baru di tahap ini.

### 1.1 Interface dulu

Buat `src/ports/guardrails.py` dan `src/adapters/guardrails/` sesuai
[`igardrail-spec.md`](./igardrail-spec.md). Jangan menyentuh `agent_service.py`
sebelum port dan fake tersedia.

### 1.2 Sisip di `rag_search`

**File:** `src/adapters/agent/agent_service.py`, `rag_search`

Setelah `retrieve()`, sebelum hasil diformat:

```python
results = await self.rag_service.retrieve(query, limit=settings.rag_top_k)
if not results:
    return "No relevant information found in Udinus knowledge base."

safe_results = []
for i, result in enumerate(results):
    content = result.get("content", "")
    verdict = await self.guard.check(
        content, GuardStage.RETRIEVED,
        GuardContext(turn_id=turn_id, source="rag", chunk_index=i,
                     tool_name="rag_search"),
    )
    if verdict.action in ("drop", "refuse"):
        continue
    safe_results.append(result)

if not safe_results:
    return "No relevant information found in Udinus knowledge base."
```

**Perilaku:** chunk dibuang, bukan seluruh jawaban ditolak. Satu chunk buruk
bukan alasan mahasiswa kehilangan jawaban.

### 1.3 Sisip di `search_web`

**File:** sama, `search_web`, setelah hasil diterima dari Exa. Gunakan
`source="web"`. Jika `drop`, kembalikan
`"Web results were discarded by the safety filter."`

### 1.4 Model: Prompt Guard 2 86M

| Model | Lisensi | Bahasa | Status | Dipakai? |
|---|---|---|---|---|
| Prompt Guard 2 **86M** | Llama 4 Community | Inggris + non-Inggris | aktif | **ya** |
| Prompt Guard 2 22M | Llama 4 Community | fokus Inggris | aktif | fallback |
| DeBERTa v2 | Apache 2.0 | Inggris saja | **archived** | tidak |

**Alasan 86M:** 22M fokus bahasa Inggris; 86M dilatih Inggris + non-Inggris.
DeBERTa archived dan tidak menangani non-Inggris.

**Kewajiban lisensi:** tampilkan "Built with Llama", sertakan lisensi +
`NOTICE`, patuhi Llama 4 Acceptable Use Policy.

### 1.5 Fail mode

| Kondisi | Perilaku |
|---|---|
| Model gagal load | fail-open: retrieval jalan, log error |
| Timeout per chunk | drop chunk itu |
| Semua chunk di-drop | "informasi tidak ditemukan" |

**Selesai jika:** ASR injection turun, pertanyaan normal tetap terjawab,
latensi terukur, dan flag bisa mematikan guard tanpa deploy.

---

## Tahap 2 — PII di output dan log

**Tidak boleh dilewati.** Audit kode menemukan tidak ada runtime PII guard:

| Yang ditemukan | Apa sebenarnya | Guard PII? |
|---|---|---|
| `tts_normalizer.py` daftar NIK/NPWP/NIM | singkatan untuk dibaca TTS | bukan |
| `tts_normalizer.py` regex email/URL | untuk dibaca TTS | bukan |
| `tts_service.py` `_mask_key` | mask API key Cartesia | hanya key itu |
| `.claude/agents/pii-log-auditor.md` | audit manual | bukan runtime |

**Temuan serius:** `tts_normalizer.py` justru mengekspansi "NIK" dan "NPWP"
agar TTS membacanya. Jika NIK muncul di output, sistem akan mengucapkannya.
Itu jalur kebocoran, bukan mitigasi.

### Urutan yang benar

1. **Regex + checksum** — NIK 16 digit + validasi struktur, NPWP, telepon
   Indonesia, email, kartu kredit (Luhn), API key umum.
2. **Presidio** — hanya jika regex terbukti kurang, terutama untuk nama orang.

### Perilaku

| Stage | Aksi |
|---|---|
| Chat | tampilkan versi tersensor |
| Voice | lewati kalimat teredaksi |
| Secret | refuse, jangan kirim sama sekali |

**Selesai jika:** output berisi NIK/API key tidak sampai ke TTS maupun SSE;
log tidak mengandung PII.

---

## Tahap 3 — Qwen3Guard (flag, audit dulu)

**Hardware tersedia:** `docker-compose.gpu.yml` mereservasi NVIDIA untuk
`ollama`, `qdrant`, `fastapi`; `Dockerfile.gpu` berbasis CUDA 12.4.

**Deployment:** Ollama tidak menyediakan Qwen3Guard resmi. Gunakan vLLM:

```bash
vllm serve Qwen/Qwen3Guard-Gen-0.6B --port 8000 --max-model-len 32768
```

**Perilaku awal:** input dinilai sebelum agent; output **hanya diaudit**
setelah turn selesai. Ia tidak mencegah pada awalnya.

**Syarat promosi ke blocking:**

| Syarat | Ambang |
|---|---|
| ASR pada golden set | turun vs Tahap 2 |
| Over-refusal | naik < 2–3% |
| Latensi sebelum audio pertama | dalam target UX |
| Bisa dimatikan | feature flag |
| Tes dua arah | ada |

Jika gagal salah satu: tetap audit, jangan blocking.

---

## Tahap 4 — Operasional

| Tool | Lokasi | Catatan |
|---|---|---|
| Langfuse | OTel exporter | core MIT; fitur `/ee` butuh lisensi |
| Garak + Promptfoo | CI | tidak pernah runtime |
| gpt-oss-safeguard | job mingguan | audit policy, bukan inline |

---

## Ambang keputusan sebelum mengadopsi classifier

Jangan terima classifier karena benchmark vendor. Terima jika pada dataset
Indonesia sendiri:

1. ASR turun dibanding tahap sebelumnya.
2. Over-refusal naik < 2–3%.
3. Latensi tambahan masih nyaman untuk voice.
4. Ada rollback: adapter di belakang port, bisa dimatikan tanpa deploy.

---

## Batas yang tidak bisa ditutup tool

Policy data pribadi mahasiswa tidak bisa ditegakkan classifier selama kiosk
belum punya identitas pengguna. Klasifikasi hanya bisa menolak; ia tidak bisa
memberi akses yang sah. Otorisasi tetap di backend.

---

## Terkait

- [`guardrails.md`](./guardrails.md) — keadaan sekarang dan gap.
- [`igardrail-spec.md`](./igardrail-spec.md) — kontrak kode.
