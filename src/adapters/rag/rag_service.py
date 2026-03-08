"""LlamaIndex RAG adapter: retrieval + vector store (Qdrant + Ollama)."""

import logging
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

        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model,
        )

        self.vector_store: QdrantVectorStore | None = None
        self.storage_context: StorageContext | None = None

        print(f"RagService initialized with Ollama embedding: {settings.ollama_embedding_model}")

    async def initialize(self) -> None:
        """Initialize Qdrant collection and vector store. Call on startup."""
        collections = await self.aclient.get_collections()
        collection_names = [c.name for c in collections.collections]

        test_embedding = self.embed_model.get_text_embedding("test")
        embedding_dim = len(test_embedding)

        if self.collection_name not in collection_names:
            print(f"Creating collection {self.collection_name}...")
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

        collection_info = await self.aclient.get_collection(self.collection_name)
        print(f"RagService initialized. Points in Qdrant: {collection_info.points_count}")

    async def index_document(
        self,
        file_id: str,
        file_path: str,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Index document into vector store with file_id and tags as metadata."""
        if not self.storage_context:
            raise ServiceUnavailableException("RagService not initialized. Call initialize() first.")

        documents = SimpleDirectoryReader(
            input_files=[file_path],
            required_exts=[".pdf", ".docx", ".doc", ".md", ".txt"]
        ).load_data()

        for doc in documents:
            doc.metadata["file_id"] = file_id
            if tags:
                doc.metadata["tags"] = tags

        nodes = self.node_parser.get_nodes_from_documents(documents)

        VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            store_nodes_override=True,
            show_progress=True,
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

        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
        )

        retriever_kwargs = {"index": index, "similarity_top_k": limit * 2}

        if tags:
            retriever_kwargs["filters"] = Filter(
                must=[
                    FieldCondition(key="tags", match=MatchAny(any=tags))
                ]
            )

        retriever = VectorIndexRetriever(**retriever_kwargs)
        nodes = await retriever.aretrieve(query)

        logger.info("RAG retrieve '%s' -> %d nodes (tags=%s)", query, len(nodes), tags)
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
