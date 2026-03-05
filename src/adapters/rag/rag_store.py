"""RAG store for Qdrant using LlamaIndex for indexing."""

from pathlib import Path
from typing import Dict, List

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser.text.semantic_splitter import (
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    SparseVectorParams,
    VectorParams,
)

from src.config import settings
from src.ports.services import IRAGStore


class RAGStore(IRAGStore):
    """RAG store using LlamaIndex + Qdrant for storing documents."""

    def __init__(self):
        """Initialize LlamaIndex with Qdrant and Ollama for storing documents."""

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

        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model,
        )

        self.vector_store: QdrantVectorStore | None = None
        self.storage_context: StorageContext | None = None

        print(f"RAGStore initialized with Ollama embedding: {settings.embedding_model}")

    async def initialize(self) -> None:
        """Initialize Qdrant collection and vector store for storing documents."""
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
                sparse_vectors_config={"text-sparse-new": SparseVectorParams()},
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

        persist_dir = str(Path(settings.rag_dir) / "index")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        try:
            self.storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir,
                vector_store=self.vector_store,
            )
        except FileNotFoundError as e:
            print(
                f"Persist files missing in {persist_dir}, starting with empty storage context: {e}"
            )
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
            )
        except Exception as e:
            print(
                f"Warning initializing storage context with persist_dir={persist_dir}: {e}"
            )
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
            )

        collection_info = await self.aclient.get_collection(self.collection_name)
        print(f"Total points in Qdrant (store): {collection_info.points_count}")
        print("RAGStore initialized")

    async def add_document(
        self,
        file_path: Path,
    ) -> int:
        """Add a document to the vector store.

        Args:
            file_path: Path to the document file

        Returns:
            int: Number of chunks created
        """
        if not self.storage_context:
            raise RuntimeError("RAGStore not initialized. Call initialize() first.")

        documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()

        nodes = self.node_parser.get_nodes_from_documents(documents)

        VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            store_nodes_override=True,
        )

        return len(nodes)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store.

        Args:
            document_id: Document ID to delete

        Returns:
            bool: True if deleted successfully
        """
        try:
            await self.aclient.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            print(f"Deleted document {document_id} from vector store")
            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False

    async def get_all_document_ids(self) -> List[str]:
        """Get all unique document IDs in the vector store.

        Returns:
            List[str]: List of document IDs
        """
        try:
            document_paths = set()
            offset = None

            while True:
                records, next_offset = await self.aclient.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for record in records:
                    payload = record.payload
                    file_path = payload.get("file_path") or payload.get("file_name")

                    doc_id = payload.get("doc_id")
                    if not doc_id and file_path:
                        try:
                            file_path_obj = Path(file_path)
                            doc_id = file_path_obj.stem
                        except Exception:
                            doc_id = None

                    if doc_id:
                        document_paths.add(str(doc_id))

                if next_offset is None:
                    break
                offset = next_offset

            return list(document_paths)
        except Exception as e:
            print(f"Error getting document IDs: {e}")
            return []

    async def get_all_documents_metadata(self) -> List[Dict]:
        """Get all documents with their metadata from vector store.

        Returns:
            List[Dict]: List of documents with their metadata
        """
        try:
            documents_dict: Dict[str, Dict] = {}
            offset = None

            while True:
                records, next_offset = await self.aclient.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for record in records:
                    payload = record.payload
                    file_path = payload.get("file_path") or payload.get("file_name")

                    try:
                        if file_path and "index" in Path(str(file_path)).parts:
                            continue
                    except Exception:
                        pass

                    file_path_obj = Path(file_path) if file_path else None
                    file_name = payload.get("file_name") or (
                        file_path_obj.name if file_path_obj else None
                    )
                    document_id = payload.get("doc_id") or (
                        file_path_obj.stem if file_path_obj else None
                    )

                    if not file_name:
                        continue

                    if file_name not in documents_dict:
                        file_type = (
                            payload.get("file_type")
                            or (
                                file_path_obj.suffix.lower().lstrip(".")
                                if file_path_obj and file_path_obj.suffix
                                else "unknown"
                            )
                        ) or "unknown"

                        file_size = (
                            payload.get("file_size")
                            or payload.get("file_size_bytes")
                            or 0
                        )
                        try:
                            file_size = int(file_size) if file_size else 0
                        except Exception:
                            file_size = 0

                        creation_date = (
                            payload.get("creation_date")
                            or payload.get("date_created")
                            or payload.get("uploaded_at")
                            or ""
                        )

                        documents_dict[file_name] = {
                            "document_id": document_id,
                            "file_name": file_name,
                            "file_type": file_type,
                            "file_size": int(file_size) if file_size else 0,
                            "creation_date": (
                                str(creation_date) if creation_date else ""
                            ),
                        }

                if next_offset is None:
                    break
                offset = next_offset

            return list(documents_dict.values())
        except Exception as e:
            print(f"Error getting documents metadata: {e}")
            import traceback

            traceback.print_exc()
            return []
