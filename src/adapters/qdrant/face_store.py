"""Qdrant face vector store adapter implementation."""
import uuid
from typing import List, Optional, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.config import settings
from src.ports.vision import IFaceVectorStore
from src.domain.entities.vision import FaceData


class FaceStore(IFaceVectorStore):
    """Face vector store using Qdrant."""
    def __init__(self):
        """Initialize Qdrant client."""
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=True,
        )
    
    async def initialize(self, collection_name: str) -> None:
        """Initialize the face embeddings collection."""
        collections = await self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if collection_name not in collection_names:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=512,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Created collection: {collection_name}")
    
    async def add_face(self, face_data: FaceData) -> str:
        """Add a face to the vector store.
        
        Args:
            face_data: Face data with embedding
            
        Returns:
            str: Face ID
        """
        point = PointStruct(
            id=face_data.id,
            vector=face_data.embedding,
            payload=face_data.to_payload(),
        )
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        
        return face_data.id
    
    async def search_face(
        self, embedding: List[float], threshold: float = 0.6
    ) -> Optional[Tuple[FaceData, float]]:
        """Search for a matching face in the vector store.
        
        Args:
            embedding: Face embedding vector
            threshold: Minimum similarity threshold (default 0.6 for cosine)
            
        Returns:
            Optional[Tuple[FaceData, float]]: Matching face data and score, or None
        """
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=1,
            score_threshold=threshold,
        )

        points = response.points if hasattr(response, 'points') else []
        
        if not points:
            return None
        
        top_result = points[0]
        face_data = FaceData(
            id=str(top_result.id),
            name=top_result.payload.get("name", "Unknown"),
            nim=top_result.payload.get("nim", ""),
            embedding=[],  # Don't return embedding in search result
        )
        
        return (face_data, top_result.score)
    
    async def get_all_faces(self) -> List[FaceData]:
        """Get all registered faces.
        
        Returns:
            List[FaceData]: List of all face data (without embeddings)
        """
        # Scroll through all points
        faces = []
        offset = None
        
        while True:
            records, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            
            for record in records:
                faces.append(FaceData(
                    id=str(record.id),
                    name=record.payload.get("name", "Unknown"),
                    nim=record.payload.get("nim", ""),
                    embedding=[],
                ))
            
            if next_offset is None:
                break
            offset = next_offset
        
        return faces
    
    async def delete_face(self, face_id: str) -> bool:
        """Delete a face from the vector store.
        
        Args:
            face_id: Face ID to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=[face_id],
            )
            return True
        except Exception:
            return False
    
    async def get_face_by_nim(self, nim: str) -> Optional[FaceData]:
        """Get face data by NIM.
        
        Args:
            nim: Student NIM
            
        Returns:
            Optional[FaceData]: Face data if found
        """
        results, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="nim",
                        match=MatchValue(value=nim),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        
        if not results:
            return None
        
        record = results[0]
        return FaceData(
            id=str(record.id),
            name=record.payload.get("name", "Unknown"),
            nim=record.payload.get("nim", ""),
            embedding=[],
        )
    
    async def update_face(self, face_id: str, name: str, nim: str) -> bool:
        """Update face metadata.
        
        Args:
            face_id: Face ID to update
            name: New name
            nim: New NIM
            
        Returns:
            bool: True if updated successfully
        """
        try:
            await self.client.set_payload(
                collection_name=self.collection_name,
                payload={"name": name, "nim": nim},
                points=[face_id],
            )
            return True
        except Exception:
            return False

