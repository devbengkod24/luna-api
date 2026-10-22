# Target Hasil Akhir Guardrails

> Dokumen ini menjawab satu pertanyaan: **"kalau guardrails selesai, seperti apa
> hasilnya?"** Isinya kriteria yang bisa diperiksa, bukan daftar pekerjaan.
> Rencana pengerjaan ada di [`IMPLEMENTATION_PLAN.md`](../engineering/IMPLEMENTATION_PLAN.md).
> Peta tahap ada di [`THREE_STAGES.md`](../architecture/THREE_STAGES.md).

## Cara membaca dokumen ini

Setiap target punya tiga bagian:

- **Kriteria** — apa yang harus benar.
- **Cara memeriksa** — bukti yang harus ada, bukan opini.
- **Status** — apakah sudah tercapai, sebagian, atau belum.

Target yang tidak punya cara pemeriksaan bukan target; itu harapan.

---

## 1. Target perilaku

### T1 — Setiap tahap punya guard

| | |
|---|---|
| **Kriteria** | `INPUT`, `RETRIEVED`, dan `OUTPUT` masing-masing punya minimal satu adapter yang aktif secara default. |
| **Cara memeriksa** | `GuardSettings` default menampilkan `enabled: true` untuk canary, secret, `pii_regex`, dan `prompt_guard`. Tes composite membuktikan setiap stage punya rantai tidak kosong. |
| **Status** | Belum — hari ini hanya `INPUT` yang aktif. |

### T2 — Chat dan voice sama-sama dijaga

| | |
|---|---|
| **Kriteria** | Tidak ada jalur di mana teks keluar ke client atau TTS tanpa melewati `OUTPUT`. |
| **Cara memeriksa** | Tes use case: `test_redacted_sentence_is_not_spoken_but_is_shown`, `test_assistant_event_uses_guarded_text`. Grep memastikan tidak ada `out_q.put(TokenEvent` atau `sentence_q.put` di luar jalur guard. |
| **Status** | Belum — token keluar di `response_use_case.py:368` sebelum guard. |

### T3 — Konten eksternal tidak dipercaya

| | |
|---|---|
| **Kriteria** | Setiap chunk RAG dan hasil web melewati classifier sebelum masuk konteks model. Chunk yang ditandai dibuang, bukan diteruskan. |
| **Cara memeriksa** | `test_prompt_guard_drops_injected_chunk`, `test_prompt_guard_passes_ukt_faq_chunk`, `test_all_chunks_dropped_returns_not_found`. |
| **Status** | Belum — gap #4 masih terbuka. |

### T4 — Kebocoran data dicegah, bukan hanya dicatat

| | |
|---|---|
| **Kriteria** | Canary, secret, dan PII regex **memblokir** pengiriman. Tidak ada jalur fail-open untuk ketiganya. |
| **Cara memeriksa** | `test_canary_blocks_leaked_system_prompt`, `test_secret_blocks_openai_key`, `test_pii_regex_redacts_nik`. Fail mode di `GuardSettings` untuk ketiganya `refuse`. |
| **Status** | Belum — belum ada runtime PII guard. |

### T5 — Pertanyaan wajar tetap dijawab

| | |
|---|---|
| **Kriteria** | Over-refusal tidak naik lebih dari 2–3% dibanding baseline pada golden set. |
| **Cara memeriksa** | `tests/eval/test_guardrails.py` membandingkan `must_answer.yaml` dan `ambiguous.yaml` sebelum dan sesudah setiap tahap. Angka dicatat di PR. |
| **Status** | Belum — belum ada golden set. |

### T6 — Penolakan tidak merusak percakapan

| | |
|---|---|
| **Kriteria** | Penolakan tetap dipersist sebagai `AssistantEvent`; history tidak berlubang. Jika penolakan terjadi setelah kalimat terkirim, yang dikirim kalimat penutup, bukan penolakan penuh. |
| **Cara memeriksa** | `test_refusal_is_emitted_and_persisted`, `test_all_sentences_redacted_sends_fallback`. |
| **Status** | Sebagian — refusal input sudah persist; perilaku output belum ada. |

### T15 — Permintaan data pribadi pihak lain ditolak

| | |
|---|---|
| **Kriteria** | Permintaan NIM, IPK, transkrip, ijazah, atau kontak mahasiswa/dosen lain ditolak **sebelum** agent memprosesnya. Permintaan data milik sendiri tetap boleh dijawab. |
| **Cara memeriksa** | `test_input_guard_refuses_other_students_nim`, `test_input_guard_refuses_other_students_transcript`, `test_input_guard_refuses_other_students_contact`, `test_input_guard_passes_own_nim_question`. `must_refuse.yaml` memuat minimal 10 variasi bahasa (formal, gaul, campur kode). |
| **Status** | Belum — tidak ada pola di `_HARMFUL`, tidak ada aturan privasi di `base.txt`. |

**Catatan penting:** T15 hanya bisa **menolak**. Ia tidak bisa memberi akses ke
data milik sendiri, karena kiosk belum punya identitas pengguna. Otorisasi
tetap di backend (lihat §4).

---

## 2. Target teknis

### T7 — Tool bisa diganti tanpa menyentuh use case

| | |
|---|---|
| **Kriteria** | Use case hanya mengenal `IGuardrail`. Menambah atau menghapus adapter hanya mengubah `dependencies.py` dan `GuardSettings`. |
| **Cara memeriksa** | Grep: tidak ada import `presidio`, `transformers`, atau `prompt_guard` di `src/use_cases/`. Tes use case memakai `FakeGuardrail`. |
| **Status** | Belum — belum ada port. |

### T8 — Bisa dimatikan saat insiden tanpa deploy

| | |
|---|---|
| **Kriteria** | Kill switch global dan flag per adapter bisa diubah dengan mengedit file JSON yang di-mount. Berlaku dalam hitungan detik, tanpa rebuild image. |
| **Cara memeriksa** | Tes: ubah file, tunggu satu siklus reload, adapter berhenti berjalan. Jika file rusak, konfigurasi terakhir yang valid tetap dipakai. |
| **Status** | Belum — `Settings` dibaca saat startup. |

### T9 — Tidak ada teks sensitif di log

| | |
|---|---|
| **Kriteria** | Log guard hanya berisi stage, action, adapter, reason code, latensi, dan panjang teks. Tidak ada teks, potongan teks, NIK, nama, atau secret. |
| **Cara memeriksa** | Tes filter log: input berisi NIK dan base64 tidak muncul di output log. Audit manual dengan `pii-log-auditor`. |
| **Status** | Belum — belum ada filter. |

### T10 — Latensi voice tetap dalam target

| | |
|---|---|
| **Kriteria** | Tambahan latensi sebelum audio pertama tetap dalam anggaran yang Anda tetapkan (default rencana: `OUTPUT` 300 ms per stage). |
| **Cara memeriksa** | `tests/latency_probe.py` diperluas untuk mengukur per stage. Angka p50 dan p95 dicatat per adapter. |
| **Status** | Belum — belum ada pengukuran. |

### T11 — Setiap adapter punya tes dua arah

| | |
|---|---|
| **Kriteria** | Setiap adapter minimal punya satu tes yang menahan serangan dan satu tes yang meloloskan pertanyaan wajar berbahasa Indonesia. |
| **Cara memeriksa** | Daftar adapter di `GuardSettings` dibandingkan dengan daftar tes. Tidak ada adapter tanpa pasangan tes. |
| **Status** | Sebagian — `input_guard` sudah punya, adapter baru belum. |

---

## 3. Target operasional

### T12 — Keputusan guard bisa diukur

| | |
|---|---|
| **Kriteria** | Ada metrik harian: jumlah refusal, jumlah redaksi, jumlah chunk dibuang, per adapter dan per reason code. |
| **Cara memeriksa** | Dashboard atau query log menunjukkan angka tersebut untuk 7 hari terakhir. |
| **Status** | Belum — gap #8 masih terbuka. |

### T13 — Perubahan adapter bisa dibandingkan

| | |
|---|---|
| **Kriteria** | Setiap perubahan adapter (ambang baru, adapter baru, perubahan urutan) disertai hasil golden set sebelum/sesudah. |
| **Cara memeriksa** | PR menyertakan tabel ASR dan over-refusal sebelum vs sesudah. |
| **Status** | Belum — belum ada golden set. |

### T14 — Red team berjalan berkala

| | |
|---|---|
| **Kriteria** | Garak dan Promptfoo berjalan di CI atau terjadwal, dengan hasil yang disimpan. |
| **Cara memeriksa** | Ada workflow atau jadwal; hasil terakhir tidak lebih tua dari 30 hari. |
| **Status** | Belum. |

---

## 4. Batas yang tetap ada setelah selesai

Ini bukan kegagalan. Ini batas yang harus diketahui dan diterima.

| Batas | Kenapa tidak bisa ditutup |
|---|---|
| **Otorisasi data mahasiswa** | Guard hanya bisa menolak. Memberi akses yang sah butuh identitas pengguna dan otorisasi backend. |
| **Konten berbahaya secara semantik** | Tidak ada model yang menilai makna di jalur request. Output guard hanya menangkap kebocoran berpola. |
| **Parafrase yang tidak dikenali** | Regex bisa diparaphrase (gap #3). Tidak ada klasifikasi semantik yang menutupnya. |
| **Rate limit** | Gap #2 tidak termasuk rencana guardrails ini. |
| **Batas frame WebSocket** | Gap #5 tidak termasuk. |
| **CORS `*`** | Gap #7 tidak termasuk. |
| **Notifikasi audio berlebih** | Gap #6 tidak termasuk. |

Kalau salah satu dari ini dianggap wajib, ia harus menjadi target tersendiri —
bukan diasumsikan tertutup oleh pekerjaan guardrails.

---

## 5. Definisi "selesai"

Guardrails dianggap selesai untuk fase ini jika:

1. T1–T4, T7–T9, T11, **T15** tercapai.
2. T5 dan T10 punya angka, meskipun belum sempurna.
3. T6, T12–T14 punya rencana dan pemilik, meskipun belum berjalan.
4. Batas di §4 tercatat di `CURRENT_STATE.md` §5 dengan status yang jujur.

Yang **tidak** termasuk definisi selesai: membuat semua pertanyaan berbahaya
tertolak. Itu tujuan yang tidak bisa dibuktikan, dan mengejarnya akan
menghasilkan over-refusal yang merusak produk.

---

## 6. Ringkasan status saat ini

| Target | Status |
|---|---|
| T1 tiga tahap aktif | Belum |
| T2 chat + voice dijaga | Belum |
| T3 konten eksternal diperiksa | Belum |
| T4 kebocoran dicegah | Belum |
| T5 over-refusal terkendali | Belum (belum ada baseline) |
| T6 penolakan tidak merusak history | Sebagian |
| T7 tool bisa diganti | Belum |
| T8 flag tanpa deploy | Belum |
| T9 log bersih | Belum |
| T10 latensi terukur | Belum |
| T11 tes dua arah | Sebagian |
| T12 keputusan terukur | Belum |
| T13 perbandingan adapter | Belum |
| T14 red team berkala | Belum |
| T15 permintaan data pihak lain ditolak | Belum |

---

## Terkait

- [`README.md`](../README.md) — pintu masuk.
- [`THREE_STAGES.md`](../architecture/THREE_STAGES.md) — peta tahap dan tool.
- [`IMPLEMENTATION_PLAN.md`](../engineering/IMPLEMENTATION_PLAN.md) — cara mencapainya.
- [`IGARDRAIL_SPEC.md`](../engineering/IGARDRAIL_SPEC.md) — kontrak kode.
- [`CURRENT_STATE.md`](../architecture/CURRENT_STATE.md) — keadaan sekarang dan gap.
