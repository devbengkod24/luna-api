"""The REST boundary: who may call it, how much they may send, and what they get back.

Each of these was open on the running stack:

* No credential anywhere. `GET /face` listed every enrolled name and NIM, and
  `POST /face/match` took its `threshold` from the caller, so `threshold=0`
  with any vector returned the nearest real person. With `API_ACCESS_TOKEN`
  set, every /api/v1 route requires it; empty keeps a developer box open.
* `DEBUG` defaulted to True, and Starlette in debug mode answers an unhandled
  exception with the traceback, bypassing the generic 500 handler.
* A JSON body was parsed whole before any size rule ran: 100 faces x 10 MB of
  base64 is a gigabyte in memory. `POST /file` read the upload whole before its
  50 MB check.
* A conversation id that is not a UUID reached the database and came back as a
  500. And `DELETE /conversation/{a}/message/{m}` never looked at `{a}`.
"""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.config import Settings, settings
from src.domain.entities import Message, UserMessage
from src.domain.entities.face import FaceData
from src.drivers import dependencies
from src.drivers import main
from src.shared.constants import Character
from src.use_cases import DeleteMessageUseCase, ListConversationMessagesUseCase

from .fakes import FakeFaceService, FakeMessageRepo

TOKEN = "unit-test-token"


class RecordingFaces(FakeFaceService):
    def __init__(self):
        super().__init__()
        self.thresholds: list[float] = []

    async def search_face(self, embedding, threshold=0.6):
        self.thresholds.append(threshold)
        return None


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(settings, "api_access_token", "")
    faces = RecordingFaces()
    main.app.state.container = SimpleNamespace(
        face_service=faces, face_recognition=None, storage_service=None, db_connection=None,
        uuid_generator=None,
    )
    client = TestClient(main.app, raise_server_exceptions=False)
    yield client, faces
    main.app.dependency_overrides.clear()
    del main.app.state.container


def guarded_routes():
    return [
        ("GET", "/api/v1/face"),
        ("POST", "/api/v1/face/match"),
        ("GET", "/api/v1/photo"),
        ("GET", "/api/v1/conversation"),
        ("GET", "/api/v1/file"),
        ("POST", "/api/v1/response/text"),
    ]


# --- credential --------------------------------------------------------------

def test_every_api_route_refuses_a_caller_without_the_token(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(settings, "api_access_token", TOKEN)
    for method, path in guarded_routes():
        response = client.request(method, path, json={})
        assert response.status_code == 401, f"{method} {path} answered {response.status_code}"
        response = client.request(method, path, json={}, headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401, f"{method} {path} took a wrong token"


def test_the_token_works_as_a_header_and_as_a_query_parameter(api, monkeypatch):
    client, faces = api
    monkeypatch.setattr(settings, "api_access_token", TOKEN)
    assert client.get("/api/v1/face", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    # An <img src> in the dashboard cannot send a header.
    assert client.get(f"/api/v1/face?token={TOKEN}").status_code == 200


def test_liveness_needs_no_token(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(settings, "api_access_token", TOKEN)
    assert client.get("/health").status_code == 200


def test_with_no_token_configured_the_api_stays_open(api):
    client, _ = api
    assert client.get("/api/v1/face").status_code == 200


def test_a_face_match_cannot_lower_the_threshold_below_the_floor(api):
    client, faces = api
    response = client.post("/api/v1/face/match", json={"embedding": [0.1] * 128, "threshold": 0.0})
    assert response.status_code == 200
    assert faces.thresholds and faces.thresholds[-1] >= settings.face_match_min_threshold, (
        f"searched at threshold {faces.thresholds[-1]}: any vector returns the nearest enrolled person"
    )


def test_a_face_match_above_the_floor_is_left_alone(api):
    client, faces = api
    client.post("/api/v1/face/match", json={"embedding": [0.1] * 128, "threshold": 0.8})
    assert faces.thresholds[-1] == 0.8


# --- size ----------------------------------------------------------------------

def test_an_oversized_body_is_refused_before_it_is_parsed(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(settings, "max_request_body_bytes", 1024)
    body = json.dumps({"faces": [{"name": "x", "image_base64": "A" * 5000}]})
    response = client.post("/api/v1/face", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 413, response.status_code


def test_an_oversized_body_without_a_length_is_refused_too(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(settings, "max_request_body_bytes", 1024)

    def chunks():
        for _ in range(10):
            yield b"A" * 512

    response = client.post("/api/v1/face", content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 413, response.status_code


# --- identifiers -----------------------------------------------------------------

def test_a_malformed_conversation_id_is_a_422_not_a_database_error(api):
    client, _ = api
    main.app.dependency_overrides[dependencies.get_list_conversation_messages_use_case] = (
        lambda: ListConversationMessagesUseCase(FakeMessageRepo())
    )
    assert client.get("/api/v1/conversation/not-a-uuid/message").status_code == 422
    assert client.post(
        "/api/v1/response/text",
        json={"conversation_id": "not-a-uuid", "user_input": "halo", "character": 0},
    ).status_code == 422


def test_a_message_cannot_be_deleted_through_another_conversation(api):
    client, _ = api
    repo = FakeMessageRepo()
    mine, theirs = str(uuid4()), str(uuid4())
    message = Message(conversation_id=mine, character=Character.LUNA, body=UserMessage(content="hi"))
    import asyncio

    asyncio.run(repo.create(message))
    main.app.dependency_overrides[dependencies.get_delete_message_use_case] = lambda: DeleteMessageUseCase(repo)

    response = client.delete(f"/api/v1/conversation/{theirs}/message/{message.id}")
    assert response.status_code == 404, "deleted a message by naming a conversation it is not in"
    assert asyncio.run(repo.get_by_id(message.id)) is not None
    assert client.delete(f"/api/v1/conversation/{mine}/message/{message.id}").status_code == 204


# --- errors ------------------------------------------------------------------------

def test_debug_is_off_unless_asked_for():
    assert Settings.model_fields["debug"].default is False
    assert main.app.debug is False, "Starlette would answer unhandled errors with a traceback"
