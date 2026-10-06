#!/usr/bin/env python3
"""Latency probe for the cloud half of a voice turn. SPENDS API CREDITS.

    docker compose exec fastapi python tests/latency_probe.py            # everything
    docker compose exec fastapi python tests/latency_probe.py deepgram   # one section

Named without a `test_` prefix so pytest never collects it: it calls Cartesia,
Deepgram and Ollama Cloud with the keys in the environment, and it asserts
nothing. Like luna-edge's `tests/rig_signals.py`, its job is to turn guesses
about where the seconds go into numbers, so the fixes proposed in the code
review can be judged against a baseline and re-measured after.

Sections, each answering one question from the review:

  guardrail  What the input guard costs per turn (it replaced NeMo Guardrails,
             which ran the same regexes at ~5-10 ms).
  cartesia   Time to first audio chunk per sentence: the adapter (a new
             AsyncCartesia client per sentence) against one client reused —
             measured 2026-09-13 as no gain. The synthesized PCM feeds `deepgram`.
  deepgram   `transcribe_pcm` on a connection that went cold (Deepgram closes
             idle connections after ~5 s, and every real turn is further apart
             than that), warm, and cold-but-kept-warm by a cheap request every
             4 s on the SDK's own connection pool.
  ollama     Time to first token with the real system prompt, the emotion held
             fixed against the emotion changing on every call. The emotion sits
             mid-prompt, so if the backend caches prompt prefixes, changing it
             discards the cache for everything after it.

Nothing printed contains a key, a transcript of a real person, or a face.
Ollama Cloud has a heavy latency tail (about 1 call in 6 takes 5-22 s), so read
the medians, and run it twice before believing a difference under ~300 ms.
"""

import asyncio
import statistics
import sys
import time
from contextlib import aclosing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

logging.disable(logging.WARNING)

from src.config import settings

SENTENCES = [
    "Halo, selamat datang di Universitas Dian Nuswantoro.",
    "Pendaftaran mahasiswa baru dibuka setiap bulan Januari.",
    "Silakan tanyakan apa saja tentang fakultas di sini.",
    "Terima kasih sudah berkunjung, sampai jumpa lagi.",
]
PCM_RATE = 16_000
DEEPGRAM_IDLE_S = 6.0      # past Deepgram's ~5 s idle close
KEEP_WARM_EVERY_S = 4.0
ROUNDS = 3
OLLAMA_CALLS = 6


def ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def summary(values: list[float]) -> str:
    if not values:
        return "no samples"
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return (
        f"median {statistics.median(ordered):8.2f} ms   p90 {p90:8.2f} ms   "
        f"min {ordered[0]:8.2f}   max {ordered[-1]:8.2f}   n={len(ordered)}"
    )


def header(title: str) -> None:
    print(f"\n== {title}")


# --- guardrail ---------------------------------------------------------------

async def probe_guardrail() -> None:
    header("guardrail: the input guard on every turn")
    from src.adapters.agent.input_guard import refusal_for

    questions = ["siapa rektor udinus sekarang?", "halo luna, apa kabar?", "kapan pendaftaran dibuka"]
    direct = []
    for i in range(300):
        started = time.perf_counter()
        refusal_for(questions[i % 3])
        direct.append(ms(time.perf_counter() - started))
    print(f"  input_guard.refusal_for  {summary(direct)}")
    print("  (NeMo Guardrails ran the same checks at ~5-10 ms per turn before it was removed)")


# --- cartesia ----------------------------------------------------------------

async def _first_chunk(stream) -> tuple[float, float, bytes]:
    """(ms to first chunk, ms to last chunk, all PCM) for one sentence."""
    started = time.perf_counter()
    first = None
    pcm = bytearray()
    async for chunk in stream:
        if first is None:
            first = time.perf_counter() - started
        pcm.extend(chunk)
    return ms(first or 0.0), ms(time.perf_counter() - started), bytes(pcm)


async def probe_cartesia() -> bytes | None:
    header(f"cartesia: time to first audio chunk per sentence ({PCM_RATE} Hz, raw PCM)")
    if settings.tts_provider.strip().lower() != "cartesia" or not settings.cartesia_api_keys:
        print("  skipped: TTS_PROVIDER is not cartesia or CARTESIA_API_KEY is empty")
        return None

    from cartesia import AsyncCartesia

    from src.adapters.speech.tts_normalizer import normalize_for_tts
    from src.adapters.speech.tts_service import CartesiaTTSService
    from src.shared.constants import Character

    adapter = CartesiaTTSService(
        api_key=settings.cartesia_api_keys,
        voice_id=settings.cartesia_voice_id,
        model_id=settings.cartesia_model_id,
        sample_rate=PCM_RATE,
        language=settings.cartesia_language,
    )
    async def reused_stream(client, text):
        stream = await client.tts.generate_sse(
            model_id=settings.cartesia_model_id,
            transcript=normalize_for_tts(text),
            voice={"mode": "id", "id": settings.cartesia_voice_id},
            output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": PCM_RATE},
            language=settings.cartesia_language,
        )
        async for event in stream:
            if event.type == "chunk" and event.audio:
                yield event.audio

    fresh_first, fresh_total, reused_first, reused_total = [], [], [], []
    pcm = bytearray()
    try:
        # The adapter first: it retires keys that are out of credit, and the
        # reused client must start on the key the adapter settled on.
        started = time.perf_counter()
        await _first_chunk(adapter.synthesize_stream("Tes.", Character.LUNA))
        retired = len(adapter._keys._retired)
        print(
            f"  key pool: {retired} of {len(adapter._keys)} keys out of credit; the first "
            f"sentence after every process start paid {ms(time.perf_counter() - started):.0f} ms "
            "to find a live one"
        )
        key = adapter._keys.current()
        async with AsyncCartesia(api_key=key) as client:
            # One unmeasured call each, so DNS and the reused client's first
            # handshake are not billed to either side.
            await _first_chunk(adapter.synthesize_stream("Tes.", Character.LUNA))
            await _first_chunk(reused_stream(client, "Tes."))
            for _ in range(ROUNDS - 1):
                for sentence in SENTENCES:
                    # Interleaved, so drift in the network hits both equally.
                    first, total, audio = await _first_chunk(
                        adapter.synthesize_stream(sentence, Character.LUNA)
                    )
                    fresh_first.append(first)
                    fresh_total.append(total)
                    if len(pcm) < PCM_RATE * 2 * 6:
                        pcm.extend(audio)
                    first, total, _ = await _first_chunk(reused_stream(client, sentence))
                    reused_first.append(first)
                    reused_total.append(total)
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {str(e)[:160]}")
        return bytes(pcm) or None

    print(f"  first chunk, new client per sentence (today)  {summary(fresh_first)}")
    print(f"  first chunk, one client reused                {summary(reused_first)}")
    print(f"  whole sentence, new client per sentence       {summary(fresh_total)}")
    print(f"  whole sentence, one client reused             {summary(reused_total)}")
    print(f"  audio kept for the deepgram section: {len(pcm) / 2 / PCM_RATE:.1f} s")
    return bytes(pcm)


# --- deepgram ----------------------------------------------------------------

async def probe_deepgram(pcm: bytes | None) -> None:
    header("deepgram: transcribe_pcm on a cold, warm and kept-warm connection")
    if not settings.deepgram_api_key:
        print("  skipped: DEEPGRAM_API_KEY is empty")
        return
    if not pcm:
        print("  skipped: no audio (the cartesia section produces it)")
        return

    from src.adapters.speech.deepgram_stt_service import DeepgramSTTService

    stt = DeepgramSTTService(api_key=settings.deepgram_api_key)
    pool = stt._client._client_wrapper.httpx_client.httpx_client
    base = stt._client._client_wrapper.get_environment().base

    async def timed(audio: bytes) -> tuple[float, int]:
        started = time.perf_counter()
        text = await stt.transcribe_pcm(audio, PCM_RATE)
        return ms(time.perf_counter() - started), len(text)

    async def keep_warm(seconds: float) -> None:
        waited = 0.0
        while waited + KEEP_WARM_EVERY_S <= seconds:
            await asyncio.sleep(KEEP_WARM_EVERY_S)
            waited += KEEP_WARM_EVERY_S
            try:
                # Unauthenticated, so it is refused and costs nothing; what it
                # buys is the pooled TLS connection staying open.
                await pool.get(f"{base}/v1/listen")
            except Exception:
                pass
        await asyncio.sleep(seconds - waited)

    # A spoken question is a few seconds, not eight. The upload is part of the
    # cost and a fresh TCP connection starts it with a small congestion window,
    # so both lengths are measured.
    short = pcm[: int(2.5 * PCM_RATE) * 2]
    for audio in (short, pcm):
        seconds = len(audio) / 2 / PCM_RATE
        cold, warm, kept = [], [], []
        chars = set()
        try:
            await timed(audio)                          # DNS, unmeasured
            for _ in range(ROUNDS):
                await asyncio.sleep(DEEPGRAM_IDLE_S)
                value, n = await timed(audio)
                cold.append(value)
                chars.add(n)
                await asyncio.sleep(1.0)
                value, n = await timed(audio)
                warm.append(value)
                await keep_warm(DEEPGRAM_IDLE_S)
                value, n = await timed(audio)
                kept.append(value)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {str(e)[:160]}")

        print(f"  --- {seconds:.1f} s of audio, transcripts of {sorted(chars)} chars (0 = not understood)")
        print(f"  cold, {DEEPGRAM_IDLE_S:.0f} s idle (every real turn)   {summary(cold)}")
        print(f"  warm, 1 s idle                   {summary(warm)}")
        print(f"  {DEEPGRAM_IDLE_S:.0f} s idle, kept warm every {KEEP_WARM_EVERY_S:.0f} s  {summary(kept)}")


# --- ollama ------------------------------------------------------------------

async def probe_ollama() -> None:
    header(f"ollama: time to first token, {settings.ollama_llm_model} via {settings.ollama_base_url}")
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    from src.adapters.prompts import FilePromptRepository
    from src.domain.entities import UserContext
    from src.shared.constants import Character

    # The same knobs LangchainAgentService builds its model with. No tools are
    # bound, so absolute numbers sit below a real turn's; the comparison between
    # the two prompt layouts is what this section is for.
    llm = ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
        top_p=settings.ollama_top_p,
        top_k=settings.ollama_top_k,
        min_p=settings.ollama_min_p,
        num_ctx=settings.ollama_num_ctx,
        num_predict=160,
        num_batch=settings.ollama_num_batch,
        repeat_penalty=settings.ollama_repeat_penalty,
        repeat_last_n=settings.ollama_repeat_last_n,
        reasoning=settings.ollama_reasoning or None,
    )
    prompts = FilePromptRepository()
    question = HumanMessage(content="Halo Luna, fakultas apa saja yang ada di Udinus?")
    emotions = ["happy", "sad", "angry", "surprise", "neutral", "fear"]

    async def ttft(emotion: str) -> tuple[float, float] | None:
        system = prompts.get_system_prompt(Character.LUNA, UserContext(emotion=emotion))
        started = time.perf_counter()
        first_any = first_content = None
        try:
            async with asyncio.timeout(60):
                # Drained rather than abandoned at the first token: breaking out
                # leaves langchain-ollama's inner stream to be closed by the
                # garbage collector, which raises inside the ollama client.
                # `num_predict` keeps the rest of the reply short.
                async with aclosing(llm.astream([SystemMessage(content=system), question])) as stream:
                    async for chunk in stream:
                        if first_any is None:
                            first_any = time.perf_counter() - started
                        if chunk.content and first_content is None:
                            first_content = time.perf_counter() - started
        except Exception as e:
            print(f"  call failed: {type(e).__name__}: {str(e)[:120]}")
            return None
        if first_content is None:
            return None
        return ms(first_any), ms(first_content)

    blocks = {"fixed emotion (A)": [], "emotion changes every call": [], "fixed emotion again (A')": []}
    await ttft("neutral")                               # model load, unmeasured
    for label, pick in (
        ("fixed emotion (A)", lambda i: "neutral"),
        ("emotion changes every call", lambda i: emotions[i % len(emotions)]),
        ("fixed emotion again (A')", lambda i: "neutral"),
    ):
        for i in range(OLLAMA_CALLS):
            result = await ttft(pick(i))
            if result is not None:
                blocks[label].append(result)

    for label, results in blocks.items():
        print(f"  first content token, {label:28s} {summary([r[1] for r in results])}")
    print("  (A' repeats A after the changing block, so drift over the run is visible)")


async def main() -> int:
    wanted = set(sys.argv[1:]) or {"guardrail", "cartesia", "deepgram", "ollama"}
    print(__doc__.strip().split("\n\n")[0])
    if "guardrail" in wanted:
        await probe_guardrail()
    pcm = None
    if wanted & {"cartesia", "deepgram"}:
        pcm = await probe_cartesia()
    if "deepgram" in wanted:
        await probe_deepgram(pcm)
    if "ollama" in wanted:
        await probe_ollama()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
