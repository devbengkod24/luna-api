import logging
import asyncio
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.node_parser.text.semantic_splitter import SemanticSplitterNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    SparseVectorParams,
    VectorParams,
)

from src.config import settings
from src.ports.services import IRAGService
from src.shared.exceptions import ServiceUnavailableException

logger = logging.getLogger(__name__)


class LlamaIndexRAGService(IRAGService):
    """RAG service: retrieval and document indexing using LlamaIndex + Qdrant."""

    def __init__(self):
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
            model_name=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        Settings.embed_model = self.embed_model

        chunk_strategy = (settings.rag_chunk_strategy or "token").strip().lower()
        if chunk_strategy == "semantic":
            self.node_parser = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embed_model,
            )
        else:
            self.node_parser = SentenceSplitter(
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
            )

        self.vector_store: QdrantVectorStore | None = None
        self.storage_context: StorageContext | None = None
        self._index: VectorStoreIndex | None = None

        logger.info(
            "RagService created (embedding_model=%s, qdrant_host=%s, chunk_strategy=%s)",
            settings.ollama_embedding_model,
            settings.qdrant_host,
            chunk_strategy,
        )

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self.vector_store is not None and self.storage_context is not None

    async def initialize(self) -> None:
        """Initialize Qdrant collection and vector store. Call on startup."""
        if self._index is not None and self.vector_store is not None and self.storage_context is not None:
            return

        collections = await self.aclient.get_collections()
        collection_names = [c.name for c in collections.collections]

        test_embedding = await asyncio.to_thread(self.embed_model.get_text_embedding, "test")
        embedding_dim = len(test_embedding)

        if self.collection_name not in collection_names:
            logger.info("Creating Qdrant collection %s", self.collection_name)
            await self.aclient.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-dense": VectorParams(size=embedding_dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={"text-sparse-new": SparseVectorParams()},
            )

        self.vector_store = QdrantVectorStore(
            aclient=self.aclient,
            client=self.client,
            prefer_grpc=True,
            collection_name=self.collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            batch_size=20,
        )

        persist_dir = str(Path(settings.persist_dir))
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        try:
            self.storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir,
                vector_store=self.vector_store,
            )
        except Exception:
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
            )

        # Cache index to avoid rebuilding on every retrieve()
        self._index = await asyncio.to_thread(
            VectorStoreIndex.from_vector_store,
            self.vector_store,
            storage_context=self.storage_context,
        )

        collection_info = await self.aclient.get_collection(self.collection_name)
        logger.info(
            "RagService initialized (collection=%s, points=%s)",
            self.collection_name,
            collection_info.points_count,
        )

    async def index_document(
        self,
        file_id: str,
        file_path: str,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Index document into vector store with file_id and tags as metadata."""
        if not self.storage_context:
            raise ServiceUnavailableException("RagService not initialized. Call initialize() first.")

        documents = await asyncio.to_thread(
            lambda: SimpleDirectoryReader(
                input_files=[file_path],
                required_exts=[".pdf", ".docx", ".doc", ".md", ".txt"],
            ).load_data()
        )

        for doc in documents:
            doc.metadata["file_id"] = file_id
            if tags:
                doc.metadata["tags"] = tags

        nodes = await asyncio.to_thread(self.node_parser.get_nodes_from_documents, documents)

        await asyncio.to_thread(
            VectorStoreIndex,
            nodes,
            storage_context=self.storage_context,
            store_nodes_override=True,
            show_progress=False,
        )

        logger.info("Indexed %d nodes for file_id=%s (tags=%s)", len(nodes), file_id, tags)
        return True

    async def retrieve(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[dict]:
        """Retrieve relevant documents, optionally filtered by tags."""
        if not self.vector_store or not self.storage_context:
            raise ServiceUnavailableException("RagService not initialized. Call initialize() first.")
        if self._index is None:
            self._index = await asyncio.to_thread(
                VectorStoreIndex.from_vector_store,
                self.vector_store,
                storage_context=self.storage_context,
            )

        retriever_kwargs = {"index": self._index, "similarity_top_k": limit}

        if tags:
            retriever_kwargs["filters"] = Filter(
                must=[
                    FieldCondition(key="tags", match=MatchAny(any=tags))
                ]
            )

        retriever = VectorIndexRetriever(**retriever_kwargs)
        nodes = await retriever.aretrieve(query)

        logger.info("RAG retrieve (query_len=%d) -> %d nodes (tags=%s)", len(query), len(nodes), tags)
        return [{"content": node.text, "score": node.score or 0.0} for node in nodes[:limit]]

    async def delete_by_file_id(self, file_id: str) -> bool:
        """Delete all vectors associated with a file_id."""
        try:
            await self.aclient.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="file_id", match=MatchValue(value=file_id))
                    ]
                ),
            )
            logger.info("Deleted vectors for file_id=%s", file_id)
            return True
        except Exception as e:
            logger.exception("Error deleting vectors for file_id=%s: %s", file_id, e)
            return False

    async def close(self) -> None:
        """Close underlying Qdrant clients."""
        try:
            await self.aclient.close()
        except Exception:
            logger.exception("Failed to close AsyncQdrantClient")
        try:
            self.client.close()
        except Exception:
            logger.exception("Failed to close QdrantClient")