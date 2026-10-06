"""The speech adapters on the voice turn's path.

* Deepgram closes an idle connection after ~5 s, and turns are further apart
  than that, so every transcription opened a new TLS connection: measured
  1440 ms cold against ~910 ms kept warm for 2.5 s of audio. `keep_warm` has to
  go through the very client the SDK transcribes with, or it warms a connection
  nobody uses.
* Cartesia's SSE stream had no timeout. A stream that stopped mid-sentence held
  the reply until the whole pipeline was killed, silently.
* A PCM chunk of odd length cannot be played: the page drops it, and splitting
  a sample across chunks would shift every later sample by a byte into noise.
* The local TTS models (MMS, Chatterbox) ran inference inside `async def`,
  blocking the event loop — the turn socket, cancels and every other reply —
  for the whole synthesis.
"""

import asyncio
import time

import httpx
import numpy as np
import pytest

from src.adapters.speech.deepgram_stt_service import DeepgramSTTService
from src.adapters.speech.tts_service import CartesiaTTSService, ChatterboxSTTSService
from src.config import settings


# --- Deepgram -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_keep_warm_uses_the_client_that_transcribes():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.host))
        return httpx.Response(401)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stt = DeepgramSTTService(api_key="k", http_client=client)
    await stt.keep_warm()
    assert seen and seen[0][1] == "api.deepgram.com", seen
    assert stt._client._client_wrapper.httpx_client.httpx_client is client, (
        "keep_warm warmed a different connection pool from the one transcription uses"
    )


@pytest.mark.asyncio
async def test_keep_warm_never_raises():
    def handler(request):
        raise httpx.ConnectError("offline")

    stt = DeepgramSTTService(api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await stt.keep_warm()   # a warm-up that fails must not fail the turn


# --- Cartesia -----------------------------------------------------------------

def cartesia():
    return CartesiaTTSService(api_key="k", voice_id="v", sample_rate=24000)


@pytest.mark.asyncio
async def test_a_stalled_tts_stream_is_given_up_on(monkeypatch):
    monkeypatch.setattr(settings, "tts_stream_idle_timeout_seconds", 0.2)
    tts = cartesia()

    async def stalls(transcript, api_key):
        yield b"\x00\x00" * 100
        await asyncio.sleep(30)
        yield b"\x00\x00"

    monkeypatch.setattr(tts, "_stream_with_key", stalls)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        async for _ in tts.synthesize_stream("Halo.", 0):
            pass
    assert time.monotonic() - started < 2, "a stalled stream held the sentence past its idle timeout"


@pytest.mark.asyncio
async def test_pcm_chunks_never_split_a_sample(monkeypatch):
    tts = cartesia()

    async def odd(transcript, api_key):
        for piece in (b"\x01", b"\x02\x03", b"\x04\x05\x06", b"\x07"):
            yield piece

    monkeypatch.setattr(tts, "_stream_with_key", odd)
    chunks = [c async for c in tts.synthesize_stream("Halo.", 0)]
    assert all(len(c) % 2 == 0 for c in chunks), [len(c) for c in chunks]
    assert b"".join(chunks) == b"\x01\x02\x03\x04\x05\x06\x07"[:6], "bytes were lost or reordered"


# --- local TTS models --------------------------------------------------------

@pytest.mark.asyncio
async def test_local_tts_inference_does_not_block_the_event_loop():
    class SlowModel:
        def generate(self, text, audio_prompt_path=None):
            time.sleep(0.5)
            return np.zeros(2400, dtype=np.float32)

    tts = ChatterboxSTTSService.__new__(ChatterboxSTTSService)
    tts.model, tts.sr, tts.voice_clone_dir, tts.device = SlowModel(), 24000, "/nonexistent", "cpu"

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    task = asyncio.create_task(ticker())
    await tts.synthesize("Halo.", 0)
    task.cancel()
    assert ticks >= 5, f"the event loop ticked {ticks} times during a 0.5 s synthesis"
