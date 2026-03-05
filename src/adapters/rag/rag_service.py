"""LlamaIndex RAG adapter (retriever-only) with Qdrant and Ollama."""

from typing import List

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

from src.config import settings
from src.ports.services import IRAGAdapter
from src.domain.entities.knowledge import RetrievalResultEntity

import logging
logger = logging.getLogger(__name__)


class RAGAdapter(IRAGAdapter):
    """RAG adapter for retrieval using LlamaIndex + Qdrant."""

    def __init__(self):
        """Initialize LlamaIndex retriever with Qdrant and Ollama."""
        self.aclient = AsyncQdrantClient(
            host=settings.qdrant_host,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=True,
        )
        self.client = QdrantClient(
            host=settings.qdrant_host,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=True,
        )
        self.collection_name = settings.knowledge_collection_name

        self.embed_model = OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=f"http://{settings.ollama_host}:{settings.ollama_port}",
        )

        self.llm = Ollama(
            model=settings.llm_model,
            base_url=f"http://{settings.ollama_host}:{settings.ollama_port}",
        )

        # self.reranker = GGUFReranker(
        #     model_path=settings.reranker_model,
        #     projector_path="projector.safetensors"
        # )
        
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        self.vector_store: QdrantVectorStore | None = None
        self.storage_context: StorageContext | None = None

        print(f"RAGAdapter (retriever) initialized with Ollama embedding: {settings.embedding_model}")

    async def initialize(self) -> None:
        """Initialize Qdrant collection and vector store for retrieval."""
        collections = await self.aclient.get_collections()
        collection_names = [c.name for c in collections.collections]

        test_embedding = self.embed_model.get_text_embedding("test")
        embedding_dim = len(test_embedding)

        if self.collection_name not in collection_names:
            print(f"Creating collection {self.collection_name} with hybrid search...")
            await self.aclient.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-dense": VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "text-sparse-new": SparseVectorParams()
                },
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")

        self.vector_store = QdrantVectorStore(
            aclient=self.aclient,
            client=self.client,
            prefer_grpc=True,
            collection_name=self.collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            batch_size=20,
        )

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        collection_info = await self.aclient.get_collection(self.collection_name)
        print(f"Total points in Qdrant: {collection_info.points_count}")
        print("LlamaIndex retriever initialized")

    async def retrieve(self, query: str, limit: int = 5) -> list[RetrievalResultEntity]:
        """Search for relevant documents.
        
        Args:
            query: Retrieval query
            limit: Maximum number of results
            
        Returns:
            list[RetrievalResultEntity]: Relevant document nodes with scores
        """
        try:
            if not self.vector_store or not self.storage_context:
                raise RuntimeError("RAGAdapter not initialized. Call initialize() first.")

            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
            )

            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=limit * 2,  # get more for potential rerank
            )

            nodes = await retriever.aretrieve(query)

            logger.info(f"RAG retrieve query: '{query}' -> found {len(nodes)} nodes")
            for i, node in enumerate(nodes[:5], 1):  # log top 5
                score = node.score or 0.0
                preview = node.text[:200] if node.text else ""
                logger.info(f"  [{i}] Score: {score:.4f} | Content preview: {preview}...")

            # reranked_nodes = await self._rerank(query, combined_nodes[:limit * 2])

            # top_nodes = reranked_nodes[:limit]
            top_nodes = nodes[:limit]

            results = [
                RetrievalResultEntity(
                    content=node.text,
                    score=node.score or 0.0,
                )
                for node in top_nodes
            ]

            return results

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            import traceback

            traceback.print_exc()
            return []
