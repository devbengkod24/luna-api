# Guardrails luna-api

Tiga dokumen, tiga tujuan berbeda. Baca sesuai kebutuhan.

| Dokumen | Isi | Kapan dibaca |
|---|---|---|
| [`guardrails.md`](./guardrails.md) | **Keadaan sekarang.** Semua layer yang aktif, urutan eksekusi, biaya, dan gap yang diketahui. | Sebelum mengubah apa pun. Ini sumber kebenaran tentang apa yang sudah ada. |
| [`implementation-plan.md`](./implementation-plan.md) | **Rencana bertahap.** Tool apa, di file mana, tahap berapa, dan kriteria selesainya. | Saat mulai mengerjakan. Ikuti urutan tahap, jangan lompat. |
| [`igardrail-spec.md`](./igardrail-spec.md) | **Kontrak kode.** `IGuardrail`, `Verdict`, `CompositeGuardrail`, `GuardSettings`, dan tes yang wajib ada. | Saat menulis kode guardrail. Ini yang mengikat bentuk interface. |

## Aturan dasar

1. **Guard tidak boleh hanya berupa prompt.** Setiap klaim keamanan harus punya
   kode dan tes yang menguncinya.
2. **Satu tool tidak menutup semua gap.** Lihat tabel gap di `guardrails.md` §5
   sebelum menambah dependency.
3. **Setiap adapter punya tes dua arah**: satu yang menahan serangan, satu yang
   meloloskan pertanyaan wajar. Guard yang hanya diuji dengan serangan akan
   diam-diam mulai menolak pertanyaan biasa.
4. **Jangan menambah tool sebelum gap sebelumnya terukur tertutup.** Tanpa
   baseline, "lebih baik" hanya klaim.

## Mulai dari mana

- Baru pertama kali? Baca `guardrails.md` §1 (alur end-to-end) dan §5 (gap).
- Siap mengerjakan? Buka `implementation-plan.md`, mulai dari Tahap 0.
- Menulis kode guard? Buka `igardrail-spec.md` §1–§3.
- Menambah model/classifier? Baca dulu `implementation-plan.md` §"Ambang
  keputusan" — jangan pasang classifier tanpa golden set.
