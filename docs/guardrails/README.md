# Guardrails luna-api

Lapisan pengaman untuk Luna: input user, konten yang diambil dari RAG/web, dan
jawaban sebelum sampai ke layar maupun TTS.

## Kebijakan model

Ini aturan yang membedakan rencana ini dari "pasang classifier sebanyak
mungkin".

| Aturan | Konsekuensi |
|---|---|
| **Tidak ada LLM generatif di jalur request** | Pengguna tidak menunggu inferensi model besar. |
| **Satu classifier kecil diizinkan** | Prompt Guard 2 86M, hanya untuk konten RAG/web. |
| **Penilaian makna dilakukan offline** | Audit mingguan, bukan sebelum jawaban dikirim. |
| **Setiap tool harus menutup gap yang terukur** | Tidak ada dependency tanpa alasan. |

Ringkasnya: **runtime memakai regex + satu classifier kecil; LLM hanya untuk
audit di luar jalur request.**

## Struktur dokumen

```
docs/guardrails/
├── README.md                          ← kamu di sini
├── architecture/
│   ├── THREE_STAGES.md                peta stage + tool
│   └── CURRENT_STATE.md               keadaan sekarang + gap
├── engineering/
│   ├── IMPLEMENTATION_PLAN.md         rencana bertahap
│   └── IGARDRAIL_SPEC.md              kontrak kode
└── product/
    └── TARGET_OUTCOMES.md             kriteria hasil akhir
```

| Dokumen | Isi | Kapan dibaca |
|---|---|---|
| [`THREE_STAGES.md`](./architecture/THREE_STAGES.md) | **Peta tiga stage.** `INPUT`, `RETRIEVED`, `OUTPUT`: apa yang diperiksa, tool apa, dan statusnya. | Pertama kali. Mulai dari sini. |
| [`TARGET_OUTCOMES.md`](./product/TARGET_OUTCOMES.md) | **Target hasil akhir.** 15 kriteria yang bisa diperiksa, batas yang tetap ada, definisi "selesai". | Sebelum mulai dan saat menilai apakah sudah cukup. |
| [`CURRENT_STATE.md`](./architecture/CURRENT_STATE.md) | **Keadaan sekarang.** Layer yang aktif, urutan eksekusi, biaya, dan gap. | Sebelum mengubah apa pun. Sumber kebenaran tentang apa yang sudah ada. |
| [`IMPLEMENTATION_PLAN.md`](./engineering/IMPLEMENTATION_PLAN.md) | **Rencana bertahap.** Tool, file, tahap, kriteria selesai. | Saat mengerjakan. Ikuti urutan, jangan lompat. |
| [`IGARDRAIL_SPEC.md`](./engineering/IGARDRAIL_SPEC.md) | **Kontrak kode.** `IGuardrail`, `Verdict`, `CompositeGuardrail`, `GuardSettings`, tes. | Saat menulis kode. Ini yang mengikat interface. |

## Tool yang dipakai

| Tool | Stage | Jenis | Status |
|---|---|---|---|
| Regex input guard | `INPUT` | deterministik | aktif sekarang |
| **Prompt Guard 2 86M** | `RETRIEVED` | **classifier kecil** | rencana |
| Canary token | `OUTPUT` | deterministik | rencana |
| Regex + checksum PII | `OUTPUT` | deterministik | rencana |
| Presidio | `OUTPUT` | NER, flag | opsional |
| Redaksi log | log | deterministik | rencana |
| DeepEval | CI | evaluasi | rencana |
| Garak + Promptfoo | CI | red team | rencana |
| Langfuse | operasional | tracing | rencana |
| LLM yang sudah ada | offline | audit | rencana |

Yang **tidak** dipakai: LLM Guard (archived), DeBERTa v2 (archived +
English-only), NeMo Guardrails (tidak menambah deteksi), Guardrails AI
(type-checker), Rebuff (tidak aktif), Qwen3Guard (butuh service GPU kedua),
gpt-oss-safeguard (model kedua untuk pekerjaan yang bisa dilakukan model yang
sudah berjalan).

## Aturan dasar

1. **Guard tidak boleh hanya berupa prompt.** Setiap klaim keamanan harus punya
   kode dan tes yang menguncinya.
2. **Satu tool tidak menutup semua gap.** Lihat tabel gap di `CURRENT_STATE.md`
   §5 sebelum menambah dependency.
3. **Setiap adapter punya tes dua arah**: satu yang menahan serangan, satu yang
   meloloskan pertanyaan wajar. Guard yang hanya diuji dengan serangan akan
   diam-diam mulai menolak pertanyaan biasa.
4. **Jangan menambah tool sebelum gap sebelumnya terukur tertutup.** Tanpa
   baseline, "lebih baik" hanya klaim.
5. **Latensi diukur, bukan diasumsikan.** Regex ~0,006 ms; classifier kecil
   10–40 ms per chunk; LLM judge ratusan ms sampai detik. Yang terakhir tidak
   masuk jalur request.

## Status sekilas

| Stage | Sekarang | Setelah Tahap 0–2 |
|---|---|---|
| `INPUT` | regex jailbreak/harmful | sama |
| `RETRIEVED` | **tidak ada** | + Prompt Guard 86M |
| `OUTPUT` | **tidak ada** | + canary, secret, PII, Presidio |

Detail dan bukti ada di [`THREE_STAGES.md`](./architecture/THREE_STAGES.md) dan
[`CURRENT_STATE.md`](./architecture/CURRENT_STATE.md) §5.

## Mulai dari mana

- Baru pertama kali? Baca `architecture/THREE_STAGES.md`.
- Ingin tahu tujuannya? Baca `product/TARGET_OUTCOMES.md`.
- Ingin tahu apa yang sudah aktif? Baca `architecture/CURRENT_STATE.md` §1 dan §5.
- Siap mengerjakan? Buka `engineering/IMPLEMENTATION_PLAN.md`, mulai dari Tahap 0.
- Menulis kode guard? Buka `engineering/IGARDRAIL_SPEC.md` §1–§3.
- Ingin menambah classifier? Baca `IMPLEMENTATION_PLAN.md` §"Ambang keputusan" —
  jangan pasang tanpa golden set.
- Ingin moderasi semantik inline? Baca `IMPLEMENTATION_PLAN.md` §"Kalau nanti
  butuh moderasi semantik inline" — ada lima angka yang harus dipenuhi dulu.
