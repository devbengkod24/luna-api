"""Every event `/turn/stream` sends names the turn it belongs to. FIXED — guards only.

The kiosk decides whether a cloud event belongs to the turn it is holding by
its `turn_id`. It used to read a missing id as "the current turn"
(`TurnManager.owns` in luna-edge), which is only safe if nothing about an OLD
turn ever arrives without an id, and three things did:

* `error` from a failed transcription or a crashed pipeline. Arriving after the
  kiosk has already opened the next turn, it closes that turn and releases the
  floor while the visitor is still talking.
* `interrupted`, sent when a turn is cancelled. It clears the "processing"
  indicator of the turn that replaced it.
* Nothing at all when the pipeline times out. `wait_for` around `_run_pipeline`
  cancels it silently, while the receive loop keeps sending `ping`. The kiosk
  counts every ping as the pipeline still talking, so its reply watchdog never
  fires (luna-edge `tests/cloud_keepalive.py` is the other half of this).

The first test is the guard: the happy path already stamps its events, and a
fix must not lose that.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import src.adapters.persistence as persistence
from src.config import settings
from src.domain.events import TokenEvent
from src.drivers import dependencies
from src.drivers.realtime.routers import turn as turn_router

from .fakes import FakeAgentService, FakeConversationRepo, FakeMessageRepo, FakePromptRepository

TURN = "0123456789abcdef0123456789abcdef"
PCM = b"\x00\x00" * 1600          # 100 ms of silence at 16 kHz
AUDIO_FRAME = bytes([0x01]) + (0).to_bytes(8, "big") + PCM


class ScriptedStt:
    def __init__(self, behaviour: str = "ok"):
        self.behaviour = behaviour
        self.warmed = 0

    async def keep_warm(self) -> None:
        self.warmed += 1

    async def transcribe_pcm(self, pcm16: bytes, sample_rate: int) -> str:
        if self.behaviour == "raise":
            raise RuntimeError("deepgram exploded")
        if self.behaviour == "hang":
            await asyncio.sleep(30)
        return "siapa rektor udinus"


class FakeDb:
    def __init__(self):
        self.log: list[str] = []

    @asynccontextmanager
    async def session(self):
        yield object()
        self.log.append("commit")

    async def get_session(self):
        async with self.session() as session:
            yield session


class FakeContainer:
    def __init__(self, stt):
        self.stt_service = stt
        self.agent_service = FakeAgentService(events=[TokenEvent(content="Halo.")])
        self.tts_service = None
        self.db_connection = FakeDb()
        self.prompt_repository = FakePromptRepository()


@pytest.fixture
def kiosk(monkeypatch):
    """A connected `/turn/stream` socket over a container of fakes."""
    monkeypatch.setattr(settings, "enable_voice_ws", True)
    monkeypatch.setattr(settings, "enable_stt", True)
    monkeypatch.setattr(settings, "enable_tts", False)
    # Short, so a silent pipeline and its pings are observable in a unit test.
    monkeypatch.setattr(settings, "frame_timeout_seconds", 0.2)
    monkeypatch.setattr(settings, "agent_idle_timeout_seconds", 0.25)
    monkeypatch.setattr(settings, "stt_keep_warm_interval_seconds", 0.05)
    monkeypatch.setattr(settings, "api_access_token", "")
    for module in (persistence, dependencies):
        monkeypatch.setattr(module, "ConversationRepository", lambda session: FakeConversationRepo())
        monkeypatch.setattr(module, "MessageRepository", lambda session: FakeMessageRepo())

    def connect(stt, query="", headers=None, container=None):
        app = FastAPI()
        app.include_router(turn_router.router)
        app.state.container = container or FakeContainer(stt)
        client = TestClient(app)
        return client.websocket_connect(
            f"/turn/stream?kiosk_id=test&sample_rate=16000{query}", headers=headers or {}
        )

    return connect


def send(ws, event, payload):
    ws.send_text(json.dumps({"event": event, "payload": payload}))


def open_and_end_turn(ws):
    send(ws, "turn_begin", {"turn_id": TURN, "trigger": "ddsd"})
    assert ws.receive_json()["event"] == "ack"
    ws.send_bytes(AUDIO_FRAME)
    send(ws, "turn_end", {"turn_id": TURN})


def receive_until(ws, wanted, seconds=3.0):
    """Messages up to and including the first whose event is in `wanted`.
    Relies on the router's own pings to never block past the deadline."""
    seen = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        message = ws.receive_json()
        seen.append(message)
        if message["event"] in wanted:
            break
    return seen


def unstamped(messages):
    return [
        m["event"] for m in messages
        if m["event"] not in ("ping", "ack") and m.get("payload", {}).get("turn_id") != TURN
    ]


# --- guard ------------------------------------------------------------------

def test_a_successful_turn_stamps_every_event(kiosk):
    with kiosk(ScriptedStt("ok")) as ws:
        open_and_end_turn(ws)
        messages = receive_until(ws, {"done", "error"})
    events = [m["event"] for m in messages]
    assert "done" in events, events
    assert not unstamped(messages), f"events without this turn's id: {unstamped(messages)}"


# --- once unstamped or silent ---------------------------------------------------

def test_a_failed_transcription_names_its_turn(kiosk):
    with kiosk(ScriptedStt("raise")) as ws:
        open_and_end_turn(ws)
        messages = receive_until(ws, {"error", "no_speech"})
    [error] = [m for m in messages if m["event"] == "error"]
    assert error["payload"].get("turn_id") == TURN, (
        f"`error` carries no turn_id: {error['payload']} — arriving after the kiosk "
        "opened its next turn, it closes that one instead"
    )


def test_a_cancelled_turn_names_itself_in_interrupted(kiosk):
    with kiosk(ScriptedStt("hang")) as ws:
        open_and_end_turn(ws)
        send(ws, "cancel", {"turn_id": TURN, "reason": "barge_in"})
        messages = receive_until(ws, {"interrupted"})
    [interrupted] = [m for m in messages if m["event"] == "interrupted"]
    assert interrupted["payload"].get("turn_id") == TURN, (
        f"`interrupted` carries no turn_id: {interrupted['payload']}"
    )


def test_a_pipeline_that_times_out_tells_the_kiosk(kiosk):
    with kiosk(ScriptedStt("hang")) as ws:
        open_and_end_turn(ws)
        # The pipeline is killed at 2 x agent_idle_timeout = 0.5 s.
        messages = receive_until(ws, {"error", "done", "no_speech"}, seconds=2.5)
    events = [m["event"] for m in messages]
    endings = [m for m in messages if m["event"] in ("error", "done", "no_speech")]
    assert endings, (
        f"the pipeline was cancelled by its timeout and the kiosk heard only {events}: "
        "pings, which the kiosk counts as the reply still coming"
    )
    assert endings[0]["payload"].get("turn_id") == TURN, endings[0]


# --- added with the fixes ------------------------------------------------------

def test_done_is_sent_only_after_the_reply_is_committed(kiosk):
    container = FakeContainer(ScriptedStt("ok"))
    with kiosk(None, container=container) as ws:
        open_and_end_turn(ws)
        messages = receive_until(ws, {"done", "error"})
        committed_when_done_arrived = list(container.db_connection.log)
    assert messages[-1]["event"] == "done", [m["event"] for m in messages]
    assert committed_when_done_arrived == ["commit"], (
        "`done` reached the kiosk before the conversation was committed; a page "
        "that continues the conversation at once can find it missing"
    )


def test_the_stt_connection_is_kept_warm_while_the_visitor_speaks(kiosk):
    stt = ScriptedStt("ok")
    with kiosk(stt) as ws:
        send(ws, "turn_begin", {"turn_id": TURN, "trigger": "ddsd"})
        assert ws.receive_json()["event"] == "ack"
        for _ in range(6):
            ws.send_bytes(AUDIO_FRAME)
            time.sleep(0.05)
        send(ws, "turn_end", {"turn_id": TURN})
        receive_until(ws, {"done", "error"})
        warmed_by_done = stt.warmed
        time.sleep(0.3)
    assert warmed_by_done >= 2, f"keep_warm ran {warmed_by_done} times during a 0.3 s turn"
    assert stt.warmed == warmed_by_done, "keep_warm went on after the turn ended"


def test_a_typed_turn_warms_nothing(kiosk):
    stt = ScriptedStt("ok")
    with kiosk(stt) as ws:
        send(ws, "turn_begin", {"turn_id": TURN, "trigger": "text", "text": "Siapa rektor Udinus?"})
        assert ws.receive_json()["event"] == "ack"
        send(ws, "turn_end", {"turn_id": TURN})
        receive_until(ws, {"done", "error"})
    assert stt.warmed == 0


def test_a_conversation_id_that_is_not_a_uuid_is_not_adopted(kiosk):
    container = FakeContainer(ScriptedStt("ok"))
    with kiosk(None, query="&conversation_id=not-a-uuid", container=container) as ws:
        send(ws, "set_conversation", {"conversation_id": "also-not-a-uuid"})
        reply = ws.receive_json()
        assert reply["event"] == "error", reply
        open_and_end_turn(ws)
        messages = receive_until(ws, {"done", "error"})
    assert messages[-1]["event"] == "done", (
        "a malformed conversation id was carried into the turn, which then failed in the database: "
        f"{[m['event'] for m in messages]}"
    )


def test_turn_end_on_a_socket_that_missed_turn_begin_names_the_turn(kiosk):
    # A reconnect between the kiosk's turn_begin and turn_end lands turn_end on a
    # socket that never saw the turn. Unstamped, the kiosk now ignores the
    # answer and waits out its watchdog.
    with kiosk(ScriptedStt("ok")) as ws:
        send(ws, "turn_end", {"turn_id": TURN})
        [reply] = receive_until(ws, {"no_speech"})
    assert reply["payload"].get("turn_id") == TURN, reply


def test_the_turn_socket_refuses_a_kiosk_without_the_token(kiosk, monkeypatch):
    monkeypatch.setattr(settings, "api_access_token", "sekret")
    with pytest.raises(WebSocketDisconnect) as refused:
        with kiosk(ScriptedStt("ok")) as ws:
            ws.receive_json()
    assert refused.value.code == 1008


def test_the_turn_socket_accepts_the_token_as_a_header(kiosk, monkeypatch):
    monkeypatch.setattr(settings, "api_access_token", "sekret")
    with kiosk(ScriptedStt("ok"), headers={"Authorization": "Bearer sekret"}) as ws:
        open_and_end_turn(ws)
        messages = receive_until(ws, {"done", "error"})
    assert messages[-1]["event"] == "done"
