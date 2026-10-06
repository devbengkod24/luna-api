"""What the TTS is asked to say, sentence by sentence. FIXED — guards only.

The reply is spoken while it is still being written: `_stream_conversation`
cuts the token stream into sentences and hands each one to the TTS as soon as
it ends. The cut used to be any `.`, `!`, `?` or newline, so a dot that is not
the end of a sentence ended one anyway. The normalizer then saw fragments it
could not read:

    "Rp 7.500.000"   -> "Rp 7." / "500." / "000"   spoken "tujuh rupiah / lima ratus / nol"
    "Prof. Dr. X"    -> three TTS requests, three pauses in one name
    "dinus.ac.id"    -> "dinus." / "ac." / "id."

The price is wrong out loud, and every extra fragment is another TTS round
trip. The guards at the bottom keep a fix honest: real sentences must still be
cut, and the first one must still reach the TTS before the reply has finished.

The normalizer tests at the end are separate bugs in `tts_normalizer.py`, found
while checking what the fragments turn into.
"""

import asyncio
import re

import pytest

from src.adapters.speech.tts_normalizer import normalize_for_tts
from src.domain.events import AudioEvent, TokenEvent
from src.shared.constants import Character
from src.use_cases.response_use_case import ResponseTextUseCase

from .fakes import RESPONSE_OPTIONS, FakeConversationRepo, FakeMessageRepo, FakePromptRepository
from src.ports.services import IAgentService, ITTSService

# Roughly how an LLM streams: words, numbers and punctuation as separate tokens,
# with the leading space attached to the word.
_TOKEN = re.compile(r"\s*[^\W\d_]+|\s*\d+|\s*[^\w\s]")


def tokens(text: str) -> list[str]:
    pieces = _TOKEN.findall(text)
    assert "".join(pieces) == text, "tokenizer must be lossless"
    return pieces


class TokenAgent(IAgentService):
    """Streams a reply token by token, yielding the loop between tokens so the
    TTS worker can run concurrently, the way a network stream lets it."""

    def __init__(self, text: str, tts: "RecordingTts"):
        self.pieces = tokens(text)
        self.tts = tts
        self.requests_when_finished: int | None = None

    async def generate_stream(self, user_input, messages, context=None, system_prompt=None):
        for piece in self.pieces:
            yield TokenEvent(content=piece)
            await asyncio.sleep(0)
        self.requests_when_finished = len(self.tts.requests)


class RecordingTts(ITTSService):
    def __init__(self):
        self.requests: list[str] = []

    async def synthesize(self, text: str, character: int) -> str:
        self.requests.append(text)
        await asyncio.sleep(0)
        return "audio"


async def spoken(text: str) -> tuple[list[str], TokenAgent]:
    tts = RecordingTts()
    agent = TokenAgent(text, tts)
    use_case = ResponseTextUseCase(
        conversation_repo=FakeConversationRepo(),
        message_repo=FakeMessageRepo(),
        agent_service=agent,
        tts_service=tts,
        prompt_repo=FakePromptRepository(),
        options=RESPONSE_OPTIONS,
    )
    audio = 0
    async for event in use_case.execute(None, "pertanyaan", Character.LUNA, None, enable_tts=True):
        audio += isinstance(event, AudioEvent)
    assert audio == len(tts.requests)
    return tts.requests, agent


def said_whole(requests: list[str], atom: str) -> bool:
    return any(atom in request for request in requests)


# --- the dot that is not a sentence end ------------------------------------

@pytest.mark.asyncio
async def test_a_rupiah_amount_reaches_the_tts_in_one_piece():
    requests, _ = await spoken("Biaya kuliahnya Rp 7.500.000 per semester.")
    assert said_whole(requests, "Rp 7.500.000"), f"the amount was cut: {requests}"


@pytest.mark.asyncio
async def test_academic_titles_do_not_end_the_sentence():
    requests, _ = await spoken("Rektor Udinus adalah Prof. Dr. Pulung Nurtantio Andono.")
    assert said_whole(requests, "Prof. Dr. Pulung"), f"the name was cut: {requests}"
    assert len(requests) == 1, f"one sentence took {len(requests)} TTS requests: {requests}"


@pytest.mark.asyncio
async def test_a_domain_name_is_not_three_sentences():
    requests, _ = await spoken("Info lengkap ada di dinus.ac.id ya.")
    assert said_whole(requests, "dinus.ac.id"), f"the address was cut: {requests}"


@pytest.mark.asyncio
async def test_a_decimal_is_not_cut():
    requests, _ = await spoken("IPK minimal 3.25 untuk beasiswa ini.")
    assert said_whole(requests, "3.25"), f"the number was cut: {requests}"


# --- guards: what a fix must keep ------------------------------------------

@pytest.mark.asyncio
async def test_real_sentences_are_still_spoken_separately():
    requests, _ = await spoken("Halo! Ada yang bisa saya bantu? Silakan bertanya.")
    assert requests == ["Halo!", "Ada yang bisa saya bantu?", "Silakan bertanya."], requests


@pytest.mark.asyncio
async def test_the_first_sentence_is_synthesized_before_the_reply_finishes():
    reply = (
        "Tentu. Universitas Dian Nuswantoro memiliki beberapa fakultas, antara lain "
        "Fakultas Ilmu Komputer, Fakultas Ekonomi dan Bisnis, Fakultas Ilmu Budaya, "
        "Fakultas Kesehatan, dan Fakultas Teknik."
    )
    requests, agent = await spoken(reply)
    assert agent.requests_when_finished, (
        "nothing reached the TTS until the whole reply was written — the first "
        "sentence must start speaking while the rest is still streaming"
    )
    assert requests[0] == "Tentu.", requests


# --- the normalizer ---------------------------------------------------------

def test_a_grade_is_not_read_as_a_time():
    assert "jam" not in normalize_for_tts("IPK minimal 3.25 ya."), normalize_for_tts(
        "IPK minimal 3.25 ya."
    )


def test_a_decimal_is_not_read_as_hundreds():
    spoken_form = normalize_for_tts("Nilai rata-rata 2.75.")
    assert "ratus" not in spoken_form, f"2.75 read as 275: {spoken_form!r}"


def test_a_time_after_jam_does_not_say_jam_twice():
    spoken_form = normalize_for_tts("Kuliah mulai jam 13.00.")
    assert "jam jam" not in spoken_form, spoken_form


def test_a_rupiah_amount_with_a_magnitude_reads_in_order():
    spoken_form = normalize_for_tts("Biayanya Rp 12,5 juta per tahun.")
    assert "dua belas koma lima juta rupiah" in spoken_form, spoken_form


def test_a_sentence_ending_in_a_number_keeps_its_full_stop():
    spoken_form = normalize_for_tts("Nilai rata-rata 2.75. Cukup bagus.")
    assert "koma tujuh lima." in spoken_form, spoken_form


def test_thousands_and_decimals_by_indonesian_convention():
    assert "seribu lima ratus mahasiswa" in normalize_for_tts("Ada 1.500 mahasiswa.")
    assert "dua koma lima persen" in normalize_for_tts("Diskon 2,5% untuk anda.")
    assert "tiga koma dua lima" in normalize_for_tts("IPK minimal 3.25 ya.")


def test_times_and_amounts_that_read_correctly_today_still_do():
    assert "delapan tiga puluh" in normalize_for_tts("Pendaftaran dibuka pukul 08.30 WIB.")
    assert "tujuh juta lima ratus ribu rupiah" in normalize_for_tts("Biaya Rp 7.500.000 per semester.")
