"""Qdrant face vector store — implements IFaceService."""

import logging

logger = logging.getLogger(__name__)

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import settings
from src.domain.entities.face import FaceData
from src.ports.services import IFaceService


def _face_payload(face: FaceData) -> dict:
    return {"name": face.name, "nim": face.nim}


class FaceStore(IFaceService):
    """Face vector store using Qdrant."""

    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=True,
        )
        self.collection_name = settings.faces_collection_name
        self._warned_dim = False

    async def initialize(self) -> None:
        collections = await self.client.get_collections()
        names = [c.name for c in collections.collections]
        if self.collection_name not in names:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.face_embedding_dim, distance=Distance.COSINE
                ),
            )

    async def add_face(self, face: FaceData) -> str:
        point = PointStruct(
            id=face.id,
            vector=face.embedding,
            payload=_face_payload(face),
        )
        await self.client.upsert(collection_name=self.collection_name, points=[point])
        return face.id

    async def search_face(
        self, embedding: list[float], threshold: float = 0.6
    ) -> tuple[FaceData, float] | None:
        # A vector of the wrong width is a configuration mistake, not a failed
        # lookup, and Qdrant answers it with an exception that used to surface as
        # a 500 and a traceback on every single attempt. Catch it here: an
        # unrecognised visitor is an ordinary outcome, so the caller gets "no
        # match" — but say plainly what is wrong, because the alternative is a
        # kiosk where recognition silently never works.
        expected = settings.face_embedding_dim
        if len(embedding) != expected:
            # Once, not per query. The kiosk asks about once a second while a
            # face is visible, and a warning repeated sixty times a minute stops
            # being a warning and starts hiding the next real one.
            if not self._warned_dim:
                self._warned_dim = True
                logger.warning(
                    "Face vector is %d-dimensional but this index holds %d. "
                    "Nothing will ever match. Whichever model enrols must also "
                    "query: SFace emits 128, InsightFace ArcFace 512. Set "
                    "FACE_EMBEDDING_DIM, recreate the collection, and re-seed.",
                    len(embedding), expected,
                )
            return None

        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=1,
            score_threshold=threshold,
        )
        points = response.points if hasattr(response, "points") else []
        if not points:
            return None
        top = points[0]
        return (
            FaceData(
                id=str(top.id),
                name=top.payload.get("name", "Unknown"),
                nim=top.payload.get("nim"),
                embedding=[],
            ),
            top.score,
        )

    async def get_all(self) -> list[FaceData]:
        faces: list[FaceData] = []
        offset = None
        while True:
            records, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                faces.append(FaceData(
                    id=str(r.id),
                    name=r.payload.get("name", "Unknown"),
                    nim=r.payload.get("nim"),
                    embedding=[],
                ))
            if next_offset is None:
                break
            offset = next_offset
        return faces

    async def get_all_cursor(
        self, limit: int, cursor: str | None = None
    ) -> tuple[list[FaceData], str | None]:
        limit_fetch = limit + 1
        records, _ = await self.client.scroll(
            collection_name=self.collection_name,
            limit=limit_fetch,
            offset=cursor,
            with_payload=True,
            with_vectors=False,
        )
        has_next = len(records) == limit_fetch
        if has_next:
            records = records[:limit]
        faces = [
            FaceData(
                id=str(r.id),
                name=r.payload.get("name", "Unknown"),
                nim=r.payload.get("nim"),
                embedding=[],
            )
            for r in records
        ]
        next_cursor: str | None = None
        if has_next and records:
            next_cursor = str(records[-1].id)
        return faces, next_cursor

    async def get_by_id(self, face_id: str) -> FaceData | None:
        results = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[face_id],
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        r = results[0]
        return FaceData(
            id=str(r.id),
            name=r.payload.get("name", "Unknown"),
            nim=r.payload.get("nim"),
            embedding=[],
        )

    async def get_by_nim(self, nim: str) -> FaceData | None:
        results, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="nim", match=MatchValue(value=nim))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        r = results[0]
        return FaceData(
            id=str(r.id),
            name=r.payload.get("name", "Unknown"),
            nim=r.payload.get("nim"),
            embedding=[],
        )

    async def delete(self, face_id: str) -> bool:
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=[face_id],
            )
            return True
        except Exception as e:
            logger.warning("Failed to delete face %s: %s", face_id, e)
            return False

    async def close(self) -> None:
        await self.client.close()
