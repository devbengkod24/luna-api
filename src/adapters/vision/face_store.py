"""Qdrant face vector store — implements IFaceService."""

import logging
from typing import List, Optional, Tuple

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

    async def initialize(self) -> None:
        collections = await self.client.get_collections()
        names = [c.name for c in collections.collections]
        if self.collection_name not in names:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
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
        self, embedding: List[float], threshold: float = 0.6
    ) -> Optional[Tuple[FaceData, float]]:
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

    async def get_all(self) -> List[FaceData]:
        faces: List[FaceData] = []
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
        self, limit: int, cursor: Optional[str] = None
    ) -> Tuple[List[FaceData], Optional[str]]:
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
        next_cursor: Optional[str] = None
        if has_next and records:
            next_cursor = str(records[-1].id)
        return faces, next_cursor

    async def get_by_id(self, face_id: str) -> Optional[FaceData]:
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

    async def get_by_nim(self, nim: str) -> Optional[FaceData]:
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
