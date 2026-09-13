"""In-memory fakes for ports used by unit tests."""

from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional, Tuple
from uuid import uuid4

from src.domain.entities import Conversation, File, Message, StoredObject
from src.domain.entities.face import FaceData
from src.domain.entities.vision import DetectedFace
from src.domain.events import AssistantEvent, LogEvent, StreamEvent, TokenEvent
from src.ports.identifiers import IUuidGenerator
from src.ports.prompts import IPromptRepository
from src.ports.repositories import IConversationRepository, IFileRepository, IMessageRepository
from src.ports.services import (
    IAgentService,
    IFaceService,
    IRAGService,
    IStorageService,
    ISTTService,
    ITTSService,
)
from src.ports.vision import IFaceDetector
from src.shared.constants import Character


class FakeConversationRepo(IConversationRepository):
    def __init__(self):
        self._items: dict[str, Conversation] = {}

    async def create(self, conversation: Conversation) -> Conversation:
        new_id = conversation.id or str(uuid4())
        stored = Conversation(
            id=new_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
        self._items[new_id] = stored
        return stored

    async def get_by_id(self, id: str) -> Optional[Conversation]:
        return self._items.get(id)

    async def list_cursor(self, limit, cursor=None):
        items = list(self._items.values())[:limit]
        return items, None

    async def update_title(self, id: str, title: str) -> bool:
        if id not in self._items:
            return False
        existing = self._items[id]
        self._items[id] = Conversation(
            id=existing.id,
            title=title,
            created_at=existing.created_at,
            updated_at=existing.updated_at,
        )
        return True

    async def delete(self, id: str) -> bool:
        return self._items.pop(id, None) is not None


class FakeMessageRepo(IMessageRepository):
    def __init__(self):
        self._items: list[Message] = []

    async def create(self, message: Message) -> Message:
        if message.id is None:
            message.id = str(uuid4())
        self._items.append(message)
        return message

    async def get_by_id(self, id: str) -> Optional[Message]:
        return next((m for m in self._items if m.id == id), None)

    async def get_by_conversation_id(self, conversation_id: str) -> List[Message]:
        return [m for m in self._items if m.conversation_id == conversation_id]

    async def get_recent_by_conversation_id(
        self, conversation_id: str, limit: int
    ) -> List[Message]:
        items = [m for m in self._items if m.conversation_id == conversation_id]
        return items[-limit:]

    async def get_by_conversation_id_cursor(
        self, conversation_id: str, limit: int, cursor: Optional[str] = None
    ) -> Tuple[List[Message], Optional[str]]:
        items = [m for m in self._items if m.conversation_id == conversation_id]
        return items[:limit], None

    async def delete(self, id: str) -> bool:
        before = len(self._items)
        self._items = [m for m in self._items if m.id != id]
        return len(self._items) != before


class FakeFileRepo(IFileRepository):
    def __init__(self):
        self._items: dict[str, File] = {}

    async def create(self, file: File) -> File:
        if file.id is None:
            file.id = str(uuid4())
        self._items[file.id] = file
        return file

    async def get_by_id(self, id: str) -> Optional[File]:
        return self._items.get(id)

    async def list(self) -> List[File]:
        return list(self._items.values())

    async def update(self, file: File) -> File:
        self._items[file.id] = file
        return file

    async def delete(self, id: str) -> bool:
        return self._items.pop(id, None) is not None


class FakeStorageService(IStorageService):
    def __init__(self):
        self.uploaded: dict[str, bytes] = {}
        self.modified: dict[str, datetime] = {}
        self.deleted: list[str] = []

    async def upload_file(self, file_content: bytes, filename: str, content_type: str) -> str:
        # The key as given, as MinIO returns it. A rewritten path here would let
        # a use case pass that addresses objects by the wrong name.
        path = filename
        self.uploaded[path] = file_content
        self.modified[path] = datetime.now(timezone.utc)
        return path

    async def download_file(self, storage_path: str) -> bytes:
        return self.uploaded.get(storage_path, b"")

    async def delete_file(self, storage_path: str) -> bool:
        self.deleted.append(storage_path)
        self.modified.pop(storage_path, None)
        return self.uploaded.pop(storage_path, None) is not None

    async def list_files(self, prefix: str) -> list[StoredObject]:
        return [
            StoredObject(path=p, size_bytes=len(b), modified_at=self.modified[p])
            for p, b in self.uploaded.items()
            if p.startswith(prefix)
        ]

    async def stat_file(self, storage_path: str) -> StoredObject | None:
        if storage_path not in self.uploaded:
            return None
        return StoredObject(
            path=storage_path,
            size_bytes=len(self.uploaded[storage_path]),
            modified_at=self.modified[storage_path],
        )


class FakeRagService(IRAGService):
    def __init__(self):
        self.indexed: list[str] = []
        self.deleted: list[str] = []

    async def index_document(
        self, file_id: str, file_path: str, tags: Optional[List[str]] = None
    ) -> bool:
        self.indexed.append(file_id)
        return True

    async def retrieve(self, query, tags=None, limit=5):
        return []

    async def delete_by_file_id(self, file_id: str) -> bool:
        self.deleted.append(file_id)
        return True


class FakeFaceService(IFaceService):
    def __init__(self):
        self._by_id: dict[str, FaceData] = {}
        self._by_nim: dict[str, FaceData] = {}

    async def initialize(self) -> None:  # pragma: no cover
        return None

    async def add_face(self, face: FaceData) -> str:
        self._by_id[face.id] = face
        if face.nim:
            self._by_nim[face.nim] = face
        return face.id

    async def search_face(self, embedding, threshold=0.6):  # pragma: no cover
        return None

    async def get_all(self):
        return list(self._by_id.values())

    async def get_all_cursor(self, limit, cursor=None):
        return list(self._by_id.values())[:limit], None

    async def get_by_id(self, face_id: str):
        return self._by_id.get(face_id)

    async def get_by_nim(self, nim: str):
        return self._by_nim.get(nim)

    async def delete(self, face_id: str) -> bool:
        face = self._by_id.pop(face_id, None)
        if face and face.nim:
            self._by_nim.pop(face.nim, None)
        return face is not None


class FakeFaceDetector(IFaceDetector):
    def __init__(self, detections: List[DetectedFace] | None = None):
        self.detections = detections or []

    async def detect_with_embedding(self, image_bytes: bytes) -> List[DetectedFace]:
        return self.detections

    async def detect_from_bytes(self, image_bytes: bytes):  # pragma: no cover
        return []

    async def detect(self, image):  # pragma: no cover
        return []


class FakeUuidGenerator(IUuidGenerator):
    def __init__(self, ids: Optional[List[str]] = None):
        self._ids = list(ids or [])
        self.calls = 0

    def new_id(self) -> str:
        self.calls += 1
        if self._ids:
            return self._ids.pop(0)
        return f"uuid-{self.calls}"


class FakePromptRepository(IPromptRepository):
    def get_system_prompt(self, character: Character, context=None) -> str:
        return f"system-prompt({character.name})"

    def get_tts_system_prompt(self) -> str:
        return "tts-system-prompt"

    def build_title_prompt(self, user_input: str) -> str:
        return f"title-prompt({user_input})"


class FakeAgentService(IAgentService):
    """Scriptable agent — yields the events you provide on each call."""

    def __init__(
        self,
        events: List[StreamEvent] | None = None,
        title: str = "Generated Title",
    ):
        self.events = events or []
        self.title = title
        self.received_messages: List[Message] = []
        self.received_input: Optional[str] = None

    async def generate_stream(
        self, user_input, messages, context=None, system_prompt=None
    ) -> AsyncGenerator[StreamEvent, None]:
        self.received_input = user_input
        self.received_messages = list(messages)
        for event in self.events:
            yield event

    async def generate_title(self, title_prompt: str) -> str:
        return self.title

    async def preprocess_tts(self, text: str, system_prompt: str) -> str:
        return text


class FakeTtsService(ITTSService):
    async def synthesize(self, text: str, character) -> str:
        return f"audio({text})"


class FakeSttService(ISTTService):
    async def transcribe(self, speech_base64: str) -> str:
        return "transcribed text"

    async def transcribe_pcm(self, pcm16: bytes, sample_rate: int) -> str:
        return "transcribed text"
