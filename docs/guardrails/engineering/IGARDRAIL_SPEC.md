# Spesifikasi `IGuardrail` & `CompositeGuardrail`

Kontrak kode untuk semua guardrail di luna-api. Dokumen ini mengikat bentuk
interface, bukan hanya niat. Jika kode menyimpang dari sini, salah satu harus
diperbaiki.

Peta tiga tahap dan tool di tiap tahap ada di
[`THREE_STAGES.md`](../architecture/THREE_STAGES.md). Rencana pengerjaan ada di
[`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md). Keadaan layer yang aktif
ada di [`CURRENT_STATE.md`](../architecture/CURRENT_STATE.md).

---

## 0. Keputusan yang mengikat

| # | Keputusan |
|---|---|
| 1 | Tiga stage: `INPUT`, `RETRIEVED`, `OUTPUT` |
| 2 | `Verdict` membawa `text` opsional; `None` = tidak berubah |
| 3 | Pipeline berurutan; `refuse`/`drop` berhenti, `redact` lanjut |
| 4 | Per kalimat untuk scanner murah; seluruh jawaban untuk semantik |
| 5 | Timeout per adapter + anggaran per stage, sisa diteruskan |
| 6 | Log metadata saja — tanpa teks |
| 7 | Flag per adapter + kill switch global |
| 8 | Guard sekali; akumulasi teks ter-guard menjadi sumber `AssistantEvent` |
| 9 | Semantik menilai gabungan semua `AssistantEvent` per turn |
| 10 | `GuardContext` tanpa identitas pengguna |
| 11 | Urutan output: canary → secret → PII regex → Presidio → semantik |
| 12 | Penolakan = `TokenEvent` + `AssistantEvent`; kalimat penutup jika sudah ada yang terkirim |
| 13 | Voice melewati kalimat teredaksi; chat menampilkan versi tersensor |
| 14 | Flag dari file JSON yang di-mount, dibaca ulang saat `mtime` berubah |
| 15 | Tiga lapis tes: contract adapter, composite, use case dengan fake |
| 16 | `Verdict` frozen dataclass + konstruktor bernama |
| 17 | `IGuardrail.check(text, stage, context)` — satu metode |
| 18 | Adapter tidak pernah melempar; composite menerapkan fail mode |
| 19 | Composite dirakit di `dependencies.py` |
| 20 | `GuardSettings` JSON dengan `enabled`, `budget_ms`, `fail_mode` |
| 21 | Tidak ada model semantik di jalur request |
| 22 | Audit mendalam dilakukan offline dengan LLM yang sudah ada, bukan inline |

---

## 1. Domain types

`src/ports/guardrails.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol


class GuardStage(StrEnum):
    INPUT = "input"
    RETRIEVED = "retrieved"
    OUTPUT = "output"


@dataclass(frozen=True)
class GuardContext:
    """Fakta minimum yang boleh diketahui guard.

    Tidak ada identitas pengguna di sini. Guard tidak membutuhkannya, dan
    membawanya menciptakan jalur kebocoran kedua.
    """

    turn_id: str
    source: Literal["rag", "web", "model"] | None = None
    chunk_index: int | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class Verdict:
    action: Literal["allow", "redact", "drop", "refuse"]
    text: str | None = None
    reason: str | None = None
    adapter: str | None = None  # diisi composite, bukan adapter

    @classmethod
    def allow(cls) -> "Verdict":
        return cls(action="allow")

    @classmethod
    def redact(cls, text: str, *, reason: str) -> "Verdict":
        if not text:
            raise ValueError("redact requires non-empty text")
        return cls(action="redact", text=text, reason=reason)

    @classmethod
    def drop(cls, *, reason: str) -> "Verdict":
        return cls(action="drop", reason=reason)

    @classmethod
    def refuse(cls, *, reason: str) -> "Verdict":
        return cls(action="refuse", reason=reason)


class IGuardrail(Protocol):
    async def check(
        self,
        text: str,
        stage: GuardStage,
        context: GuardContext | None = None,
    ) -> Verdict: ...
```

### Aturan interface

- Adapter yang tidak mendukung suatu stage mengembalikan `Verdict.allow()` dengan
  `reason="unsupported_stage"`. Ia tidak melempar.
- Adapter tidak pernah mengisi `adapter`; composite yang mengisinya.
- `redact` tanpa teks adalah kesalahan programmer, bukan kegagalan runtime.

---

## 2. Composite

`src/adapters/guardrails/composite.py`

```python
class CompositeGuardrail(IGuardrail):
    def __init__(
        self,
        chains: dict[GuardStage, list[IGuardrail]],
        settings: GuardSettings,
    ) -> None:
        self._chains = chains
        self._settings = settings

    async def check(self, text, stage, context=None) -> Verdict:
        budget = self._settings.stage(stage).budget_ms / 1000
        deadline = time.monotonic() + budget
        current = text

        for adapter in self._chains.get(stage, []):
            name = type(adapter).__name__
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._fail(stage, name, "stage_budget_exhausted")

            try:
                verdict = await asyncio.wait_for(
                    adapter.check(current, stage, context),
                    timeout=remaining,
                )
            except (asyncio.TimeoutError, Exception):
                return self._fail(stage, name, "adapter_error")

            if verdict.action == "allow":
                continue
            if verdict.action == "redact":
                current = verdict.text
                continue
            if verdict.action in ("drop", "refuse"):
                return replace(verdict, adapter=name)

        if current != text:
            return Verdict.redact(current, reason="composite")
        return Verdict.allow()
```

### Aturan composite

1. **Berurutan.** Tidak ada paralelisme — hasil harus deterministik.
2. **Berhenti pada `drop`/`refuse`.** Scanner berikutnya tidak dijalankan.
3. **`redact` mengubah input untuk scanner berikutnya.** Adapter berikutnya
   melihat teks yang sudah diredaksi.
4. **Anggaran per stage**, bukan per adapter. Sisa waktu diteruskan.
5. **Adapter tidak pernah melempar keluar.** Exception dan timeout diubah
   menjadi fail mode.
6. **`adapter` diisi composite**, bukan adapter.
7. Jika teks berubah tetapi tidak ada yang `refuse`, hasil akhirnya `redact`.

### Urutan rantai

| Stage | Urutan |
|---|---|
| `INPUT` | `regex_input` |
| `RETRIEVED` | `prompt_guard` → `provenance_policy` |
| `OUTPUT` | `canary` → `secret` → `pii_regex` → `presidio` (flag) |

Tidak ada adapter semantik di rantai mana pun. Jika suatu saat dibutuhkan,
ia harus melewati ambang keputusan di
[`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) §"Kalau nanti butuh
moderasi semantik inline" terlebih dahulu.

---

## 3. Fail mode

`src/adapters/guardrails/fail_mode.py`

| Stage | Default | Alasan |
|---|---|---|
| `INPUT` | `allow` | Jangan matikan kiosk karena classifier error |
| `RETRIEVED` | `drop` | Buang chunk, jawab dengan konteks lain |
| `OUTPUT` (canary/secret/pii) | `refuse` | Jangan pernah kirim kredensial |
| `OUTPUT` (presidio) | `allow` | Nama bukan kegagalan fatal |
| Semantik (background) | `allow` + alert | Tidak mencegah, hanya menandai |

Fail mode bisa diubah lewat `GuardSettings` tanpa deploy.

---

## 4. GuardSettings

`src/adapters/guardrails/settings.py`

```json
{
  "enabled": true,
  "stages": {
    "input":     {"budget_ms": 50,  "fail_mode": "allow"},
    "retrieved": {"budget_ms": 200, "fail_mode": "drop"},
    "output":    {"budget_ms": 300, "fail_mode": "refuse"}
  },
  "adapters": {
    "canary":       {"enabled": true,  "fail_mode": "refuse"},
    "secret":       {"enabled": true,  "fail_mode": "refuse"},
    "pii_regex":    {"enabled": true,  "fail_mode": "refuse"},
    "presidio":     {"enabled": false, "fail_mode": "allow"},
    "prompt_guard": {"enabled": true,  "fail_mode": "drop"}
  }
}
```

**Mekanisme:**

- File di-mount, mis. `/app/config/guard_settings.json`.
- Dibaca ulang ketika `st_mtime` berubah; dicek maksimal sekali per 5 detik.
- Jika file hilang atau JSON tidak valid: pakai konfigurasi terakhir yang valid,
  log error, dan **jangan** mematikan guard.
- `enabled: false` = `NoopGuardrail` untuk semua stage.

---

## 5. Titik sisip

### 5.1 INPUT — `response_use_case.py`

Sebelum `agent_service.generate_stream(...)`:

```python
verdict = await guard.check(user_input, GuardStage.INPUT, ctx)
if verdict.action == "refuse":
    # TokenEvent + AssistantEvent, sama seperti refusal input_guard sekarang
```

### 5.2 RETRIEVED — `agent_service.py`

`rag_search`, setelah `retrieve()`:

```python
for i, result in enumerate(results):
    verdict = await guard.check(
        content, GuardStage.RETRIEVED,
        GuardContext(turn_id=turn_id, source="rag", chunk_index=i,
                     tool_name="rag_search"),
    )
    if verdict.action in ("drop", "refuse"):
        continue
    safe_results.append(result)
```

`search_web` sama, dengan `source="web"`.

Jika semua chunk dibuang: kembalikan
`"No relevant information found in Udinus knowledge base."`

### 5.3 OUTPUT — `response_use_case.py`

**Perubahan inti:** buffer kalimat **sebelum** `out_q.put`.

```python
if isinstance(event, TokenEvent):
    buffer += event.content
    sentences, buffer = _drain_sentences(buffer)
    for sentence in sentences:
        verdict = await guard.check(sentence, GuardStage.OUTPUT, ctx)

        if verdict.action == "refuse":
            text = refusal_text(already_sent=bool(sent_texts))
        elif verdict.action == "redact":
            text = verdict.text or sentence
            if enable_tts:
                continue          # voice: lewati kalimat teredaksi
        else:
            text = sentence

        sent_texts.append(text)
        await out_q.put(TokenEvent(content=text))     # chat
        if enable_tts and verdict.action != "redact":
            await sentence_q.put(text)                # voice
    continue   # jangan jatuh ke out_q.put(event) di atas
```

`AssistantEvent`:

```python
elif isinstance(event, AssistantEvent):
    text = event.message.content
    if text != "".join(sent_texts):
        verdict = await guard.check(text, GuardStage.OUTPUT, ctx)
        text = verdict.text or text
    await out_q.put(AssistantEvent(message=replace(event.message, content=text)))
    await message_repo.create(Message(..., body=AIMessage(content=text)))
```

**Aturan:**

- Teks yang dipersist = teks yang dikirim, bukan output mentah model.
- `AssistantEvent` hanya di-guard ulang jika isinya berbeda dari akumulasi.
- Voice melewati kalimat `redact`; chat menampilkannya.
- Jika semua kalimat teredaksi dan voice aktif, kirim satu kalimat fallback.

### 5.4 Semantik output — background

Setelah `DoneEvent`:

```python
asyncio.create_task(audit_output(turn_id, "".join(sent_texts)))
```

Hasil: log + metrik + alert. Tidak mengubah apa pun yang sudah dikirim.

---

## 6. Penolakan

| Kondisi | Yang dikirim |
|---|---|
| Belum ada kalimat terkirim | Kalimat penolakan penuh (`REFUSE_*` dari `input_guard`) |
| Sudah ada kalimat terkirim | Kalimat penutup: "Maaf, saya tidak dapat melanjutkan bagian itu." |
| Voice, semua kalimat teredaksi | Satu kalimat fallback |

Penolakan tetap dipersist sebagai `AssistantEvent`, sehingga history tidak
berlubang.

---

## 7. Observability

Satu baris log per keputusan, tanpa teks:

```
guard stage=output action=redact adapter=pii_regex reason=nik
      latency_ms=1.2 turn_id=... len_in=84 len_out=84
```

Yang **tidak** boleh dicatat: teks, potongan teks, NIK, nama, secret, hash
yang bisa dibalik.

---

## 8. Kontrak tes

### 8.1 Contract test per adapter

Setiap adapter minimal punya dua tes:

| Tes | Tujuan |
|---|---|
| `test_<adapter>_blocks_<attack>` | menahan |
| `test_<adapter>_passes_ordinary_indonesian` | meloloskan |

Contoh wajib:

- `test_canary_blocks_leaked_system_prompt`
- `test_canary_passes_normal_answer`
- `test_secret_blocks_openai_key`
- `test_secret_passes_ordinary_number`
- `test_pii_regex_redacts_nik`
- `test_pii_regex_passes_nim` (NIM bukan NIK)
- `test_prompt_guard_drops_injected_chunk`
- `test_prompt_guard_passes_ukt_faq_chunk`

### 8.2 Composite test

| Tes | Tujuan |
|---|---|
| `test_stops_on_refuse` | adapter berikutnya tidak dijalankan |
| `test_continues_on_redact_with_new_text` | adapter berikutnya melihat teks teredaksi |
| `test_respects_stage_budget` | adapter lambat tidak melewati anggaran |
| `test_adapter_error_uses_fail_mode` | fail mode diterapkan |
| `test_adapter_field_is_set_by_composite` | adapter tidak bisa memalsukan nama |

### 8.3 Use case test dengan fake

`FakeGuardrail` yang dapat diprogram per stage:

```python
FakeGuardrail(verdicts={GuardStage.OUTPUT: [Verdict.allow(), Verdict.refuse(...)]})
```

| Tes | Tujuan |
|---|---|
| `test_refusal_is_emitted_and_persisted` | TokenEvent + AssistantEvent |
| `test_redacted_sentence_is_not_spoken_but_is_shown` | aturan voice vs chat |
| `test_assistant_event_uses_guarded_text` | sumber kebenaran teks |
| `test_assistant_event_is_reguarded_when_different` | isi berbeda |
| `test_all_sentences_redacted_sends_fallback` | fallback voice |
| `test_guard_failure_does_not_break_stream` | fail mode |

### 8.4 Aturan tes

- Setiap adapter wajib punya tes "loloskan". Guard yang hanya diuji dengan
  serangan akan diam-diam mulai menolak pertanyaan wajar.
- Jangan menguji teks penolakan secara harfiah di banyak tempat; uji konstanta
  `REFUSE_*` yang sama.

---

## 9. Yang tidak termasuk spesifikasi ini

- `TOOL_CALL` stage — belum ada tool dengan efek samping.
- Model semantik di jalur request — audit dilakukan offline.
- Presidio di input — nilainya rendah.
- Redaksi `ToolResultEvent` — teks tool masuk konteks model dan sudah ditangani
  di `RETRIEVED`.

---

## 10. Urutan implementasi

1. `ports/guardrails.py` + `Verdict` + `GuardContext`.
2. `NoopGuardrail` + `CompositeGuardrail` + `FakeGuardrail` (tes).
3. `GuardSettings` + pembacaan file.
4. `canary`, `secret`, `pii_regex` (deterministik, tanpa model).
5. Perubahan `response_use_case.py`.
6. `prompt_guard` + sisip di `rag_search`/`search_web`.
7. Contract test per adapter.
8. `presidio` (flag, default off).
9. Langfuse + Garak/Promptfoo + audit offline dengan LLM yang sudah ada.
