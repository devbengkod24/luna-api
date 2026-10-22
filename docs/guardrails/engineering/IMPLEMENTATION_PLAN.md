# Rencana Implementasi Guardrails

> Dokumen kerja. Untuk peta tiga tahap (`INPUT`, `RETRIEVED`, `OUTPUT`) dan tool
> di tiap tahap, lihat [`THREE_STAGES.md`](../architecture/THREE_STAGES.md). Untuk kriteria
> hasil akhir, lihat [`TARGET_OUTCOMES.md`](../product/TARGET_OUTCOMES.md). Untuk keadaan
> yang sudah aktif, lihat [`CURRENT_STATE.md`](../architecture/CURRENT_STATE.md). Untuk bentuk kode,
> lihat [`IGARDRAIL_SPEC.md`](./IGARDRAIL_SPEC.md).

## Ringkasan

Guardrails luna-api saat ini punya sembilan gap yang terdokumentasi di
`CURRENT_STATE.md` §5. Rencana ini menutupnya bertahap, satu tool runtime per
tahap, dengan pengukuran di setiap langkah.

**Prinsip:** satu tahap = satu perubahan yang bisa diukur dan bisa dimatikan.
Jangan mengerjakan dua tahap sekaligus.

## Tool yang dipakai

| # | Tool | Peran | Tahap | Lisensi | Runtime? |
|---|---|---|---|---|---|
| 1 | Regex input guard *(sudah ada)* | jailbreak, harmful | 0 | — | selalu |
| 1b | **Pola privasi data pribadi** | NIM/IPK/transkrip/ijazah pihak lain (T15) | 0 | — | selalu |
| 2 | Canary token | deteksi kebocoran system prompt | 0 | — | selalu |
| 3 | Redaksi log (`logging.Filter`) | cegah PII/secret di log | 0 | — | selalu |
| 4 | DeepEval | ukur ASR & over-refusal | 0 | Apache 2.0 | CI saja |
| 5 | Prompt Guard 2 86M | injection di RAG/web | 1 | Llama 4 Community | kondisional |
| 6 | Regex + checksum PII | NIK, NPWP, telepon, email, kartu | 2 | — | selalu |
| 7 | Presidio | PII nama & variasi | 2 | MIT | flag |
| 8 | Langfuse | trace & refusal rate | 3 | MIT core | async |
| 9 | Garak + Promptfoo | red team | 3 | Apache 2.0 / MIT | CI |
| 10 | LLM yang sudah ada (`OLLAMA_LLM_MODEL`) | audit offline dengan policy UDINUS | 3 | — | offline |

**Tidak dipakai:** LLM Guard (archived), DeBERTa v2 (archived + English-only),
NeMo Guardrails (tidak menambah deteksi, sudah dihapus), Guardrails AI
(type-checker), Rebuff (tidak aktif), Qwen3Guard (butuh service GPU kedua),
gpt-oss-safeguard (model kedua untuk pekerjaan yang bisa dilakukan model yang
sudah berjalan).

---

## Sebelum mulai: keputusan yang harus diambil

Empat hal ini **tidak bisa ditebak** dari kode dan harus diputuskan sebelum
baris pertama ditulis. Tanpa ini, pekerjaan akan berhenti di tengah.

### K1 — Sumber `turn_id`

`GuardContext` mewajibkan `turn_id`, tetapi hari ini:

- `ResponseTextUseCase.execute` tidak punya parameter itu;
- `IAgentService.generate_stream` tidak membawanya;
- REST tidak punya konsep turn; `turn_id` hanya hidup di WS router
  (`drivers/realtime/routers/turn.py`).

| Opsi | Perubahan | Konsekuensi |
|---|---|---|
| **A. Opsional** | `turn_id: str \| None` di `GuardContext`; WS mengisi dari turn-nya, REST mengisi `uuid4` per request | Port tidak berubah signature. Korelasi REST tetap ada. |
| B. Wajib di port | Tambah `turn_id` ke `generate_stream` dan semua pemanggil | Konsisten, tetapi menyentuh port + 3 pemanggil + fake |

**Rekomendasi: A.** Alasan: guard tidak membutuhkan `turn_id` untuk bekerja;
ia hanya membutuhkannya untuk log. Mengubah signature port untuk kebutuhan
logging adalah biaya yang tidak sepadan.

### K2 — Perubahan wiring

Tidak ada guard yang bisa dipanggil tanpa perubahan ini.

| File | Perubahan |
|---|---|
| `src/ports/guardrails.py` | **baru** — `GuardStage`, `GuardContext`, `Verdict`, `IGuardrail` |
| `src/ports/services.py` | **tidak berubah** jika K1 memilih opsi A |
| `src/adapters/agent/agent_service.py` | `__init__(self, rag_service, guard: IGuardrail)`; simpan `self.guard` |
| `src/drivers/dependencies.py` | `build_guardrail(settings)` lalu inject ke `LangchainAgentService` dan use case |
| `src/use_cases/response_use_case.py` | terima `guard` di konstruktor; pakai di `_llm_producer` |

### K3 — Buffer kalimat tanpa TTS

Hari ini `tts_buffer` hanya diakumulasi di dalam `if enable_tts`. Guard output
harus berjalan **juga saat TTS mati**, karena chat tetap membocorkan teks.

Aturan: buffer kalimat dipisahkan dari kondisi `enable_tts`. TTS hanya
menentukan apakah hasilnya dikirim ke `sentence_q`, bukan apakah guard berjalan.

### K4 — Perintah verifikasi

`pytest` tidak ada di `requirements/base.txt`. Jalankan di container:

```bash
docker compose exec fastapi pytest tests/unit/test_agent_guard.py
docker compose exec fastapi pytest tests/unit/
docker compose exec fastapi pytest -k "guard"
```

Jika `pytest` belum terpasang di container:

```bash
docker compose exec fastapi pip install pytest pytest-asyncio
```

Cek service siap:

```bash
curl -s localhost:8000/ready
```

---

## Peta tool → file → fungsi

| Tahap | Tool | File | Fungsi | Stage |
|---|---|---|---|---|
| 0 | Pola privasi data pribadi | `adapters/agent/input_guard.py` | `_PRIVACY` baru | INPUT |
| 0 | Aturan privasi prompt | `prompts/base.txt` | bagian PRIVACY | prompt |
| 0 | Canary | `adapters/prompts/file_prompt_repository.py` | `_context_value` | prompt |
| 0 | Redaksi log | `drivers/logging.py` | filter baru | log |
| 0 | DeepEval | `tests/eval/` (baru) | CI | ukur |
| 1 | Prompt Guard 2 86M | `adapters/agent/agent_service.py` | `rag_search` | RETRIEVED |
| 1 | Prompt Guard 2 86M | `adapters/agent/agent_service.py` | `search_web` | RETRIEVED |
| 2 | Regex PII + Presidio | `use_cases/response_use_case.py` | sebelum `out_q.put` | OUTPUT |
| 3 | Langfuse | `drivers/` | OTel exporter | trace |
| 3 | Garak/Promptfoo | `.github/workflows/` | CI | red team |
| 3 | LLM yang sudah ada | job terpisah | mingguan | audit |

Nomor baris bisa bergeser; nama fungsi tidak.

## File yang disentuh per tahap

Daftar ini yang membuat pekerjaan bisa dimulai tanpa menebak.

### Tahap 0

| File | Aksi |
|---|---|
| `src/adapters/agent/input_guard.py` | tambah `_PRIVACY` untuk NIM/IPK/transkrip/ijazah pihak lain (T15) |
| `src/prompts/base.txt` | tambah bagian PRIVACY: jangan berikan data mahasiswa/dosen lain |
| `src/adapters/prompts/file_prompt_repository.py` | sisipkan canary token |
| `src/drivers/logging.py` | filter redaksi |
| `tests/eval/` (baru) | golden set + `test_guardrails.py` |
| `tests/unit/test_agent_guard.py` | tes canary dua arah |
| `src/use_cases/response_use_case.py` | buffer kalimat sebelum `out_q.put` |

### Tahap 1

| File | Aksi |
|---|---|
| `src/ports/guardrails.py` | **baru** |
| `src/adapters/guardrails/__init__.py` | **baru** |
| `src/adapters/guardrails/noop.py` | **baru** |
| `src/adapters/guardrails/composite.py` | **baru** |
| `src/adapters/guardrails/settings.py` | **baru** |
| `src/adapters/guardrails/prompt_guard.py` | **baru** |
| `src/adapters/agent/agent_service.py` | terima guard; sisip di `rag_search`, `search_web` |
| `src/drivers/dependencies.py` | rakit composite, inject |
| `tests/unit/fakes.py` | tambah `FakeGuardrail` |
| `tests/unit/test_guard_composite.py` | **baru** |
| `tests/unit/test_prompt_guard.py` | **baru** |

### Tahap 2

| File | Aksi |
|---|---|
| `src/adapters/guardrails/canary.py` | **baru** |
| `src/adapters/guardrails/secret.py` | **baru** |
| `src/adapters/guardrails/pii_regex.py` | **baru** |
| `src/adapters/guardrails/presidio.py` | **baru**, flag off |
| `src/use_cases/response_use_case.py` | pakai rantai `OUTPUT` |
| `tests/unit/test_pii_regex.py` | **baru** |

### Tahap 3

| File | Aksi |
|---|---|
| `src/drivers/telemetry.py` | **baru**, exporter Langfuse |
| `.github/workflows/guardrails-redteam.yml` | **baru** |
| `scripts/audit_offline.py` | **baru**, job mingguan |

## Definition of done per tahap

Sebuah tahap **tidak selesai** hanya karena kode ditulis. Semua baris harus
terpenuhi.

| Tahap | Tes hijau | Flag bisa dimatikan | Angka tercatat |
|---|---|---|---|
| 0 | `test_agent_guard.py`, `test_tts_chunking.py`, `test_response_use_case.py`, **tes privasi T15** | — (belum ada adapter) | baseline ASR & over-refusal di PR |
| 1 | `test_guard_composite.py`, `test_prompt_guard.py`, `test_agent_guard.py` | `prompt_guard.enabled=false` | ASR injection sebelum/sesudah |
| 2 | `test_pii_regex.py`, `test_response_use_case.py`, `test_agent_guard.py` | `pii_regex.enabled=false`, `presidio.enabled=false` | over-refusal sebelum/sesudah |
| 3 | tes tetap hijau | kill switch global | refusal rate 7 hari |

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
lengkap ada di [`IGARDRAIL_SPEC.md`](./IGARDRAIL_SPEC.md) §5.3.

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

### 0.2b Pola privasi data pribadi (T15)

**File:** `src/adapters/agent/input_guard.py` + `src/prompts/base.txt`

Hari ini `_HARMFUL` tidak memuat permintaan data mahasiswa/dosen lain, dan
`base.txt` tidak punya aturan privasi. Artinya "Berikan NIM mahasiswa X" tidak
tertangkap oleh apa pun.

**Perubahan kode:** kategori `_PRIVACY` di `input_guard.py`, dipanggil dari
`refusal_for()` sebelum `_HARMFUL`:

| Pola | Contoh |
|---|---|
| NIM orang lain | "NIM teman saya", "NIM mahasiswa bernama X" |
| IPK / nilai orang lain | "IPK teman saya", "nilai si X" |
| Transkrip / ijazah | "tampilkan transkrip X", "ijazah mahasiswa itu" |
| Kontak pribadi | "nomor HP dosen X", "alamat rumah mahasiswa X" |

**Perubahan prompt:** bagian PRIVACY di `base.txt` sebagai lapisan kedua —
model tetap menolak meskipun regex meleset.

**Batas yang harus dijaga:** permintaan data **milik sendiri** tetap boleh
dijawab ("berapa NIM saya?"). Pola harus mensyaratkan pihak lain, bukan sekadar
kata "NIM".

**Selesai jika:** empat tes T15 hijau dan golden set `must_refuse.yaml`
memuat minimal 10 variasi bahasa.

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
[`IGARDRAIL_SPEC.md`](./IGARDRAIL_SPEC.md). Jangan menyentuh `agent_service.py`
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

## Tahap 3 — Operasional dan audit

Tidak ada model baru di jalur runtime. Tahap ini menutup gap #8 (refusal tidak
diukur) dan memberi sinyal audit tanpa menambah latensi pengguna.

### 3.1 Langfuse

Trace per turn, jumlah refusal, jumlah redaksi, jumlah chunk dibuang.
Core MIT; fitur `/ee` (audit log, RBAC, retention, SCIM) butuh lisensi.

### 3.2 Garak + Promptfoo

Red team terjadwal di CI. Tidak pernah di jalur request.

### 3.3 Audit offline dengan model yang sudah ada

Tidak ada model baru. Audit memakai LLM yang **sudah** berjalan di
`luna-api` (`OLLAMA_LLM_MODEL`, default `gpt-oss:120b-cloud`), dipanggil dari
job terpisah di luar jalur request.

**Kenapa tidak menambah model:**

| Alasan | Dampak |
|---|---|
| GPU sudah dipakai Ollama dan Qdrant | tidak perlu berebut VRAM |
| Model kedua = service kedua | tidak ada container, port, atau dependency baru |
| Latensi tidak relevan untuk audit | tidak perlu model khusus yang cepat |
| Sudah ada akses model | tidak ada biaya setup |

**Bias yang harus diwaspadai:** model yang sama dengan yang menghasilkan
jawaban cenderung menilai outputnya sendiri sebagai wajar. Mitigasinya:

- pakai prompt audit yang eksplisit dan berbeda dari system prompt produksi,
- sertakan policy UDINUS sebagai teks di prompt audit,
- validasi sampel kecil secara manual setiap bulan,
- bandingkan temuan antar waktu, bukan menilai satu sesi.

Jika bias terbukti terlalu besar (audit hampir selalu meloloskan), itu sinyal
untuk mempertimbangkan judge yang berbeda — tetapi keputusan itu diambil
berdasarkan data, bukan di awal.

**Bentuk job:**

```
ambil sample percakapan 7 hari terakhir
  → untuk setiap turn: prompt audit + policy UDINUS + transkrip
  → kumpulkan keputusan (patuh / melanggar / ambigu)
  → tulis ringkasan mingguan
  → temuan baru masuk ke tests/eval/golden/
```

**Syarat sebelum audit dianggap berguna:**

| Syarat | Ambang |
|---|---|
| Sample per minggu | cukup untuk menangkap pola, bukan semua turn |
| Policy | ditulis eksplisit di prompt, bisa direvisi tanpa ubah kode |
| Tindak lanjut | temuan masuk ke golden set, bukan hanya dibaca |
| Validasi | sampel kecil diperiksa manusia setiap bulan |

### 3.4 Kalau nanti butuh moderasi semantik inline

Ini **bukan** bagian rencana ini. Jika ASR pada serangan parafrase tetap tinggi
setelah Tahap 1–2, buka keputusan baru dengan syarat:

| Syarat | Ambang |
|---|---|
| ASR pada golden set | turun dibanding Tahap 2 |
| Over-refusal | naik < 2–3% |
| Latensi sebelum audio pertama | dalam target UX |
| Bisa dimatikan | feature flag |
| Tes dua arah | ada |

Tanpa kelima angka itu, jangan tambah model ke jalur request.

---

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

- [`CURRENT_STATE.md`](../architecture/CURRENT_STATE.md) — keadaan sekarang dan gap.
- [`IGARDRAIL_SPEC.md`](./IGARDRAIL_SPEC.md) — kontrak kode.
