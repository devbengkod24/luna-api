"""LlamaIndex RAG adapter with Qdrant and Ollama."""
import uuid
from typing import List
from pathlib import Path
from typing import Optional

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.node_parser.text.semantic_splitter import SemanticSplitterNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, VectorsConfig, SparseVectorParams
# from rerank import GGUFReranker

from src.config import settings
from src.ports.rag import IRAGAdapter
from src.domain.entities.document import DocumentMetadata, RetrievalResult, NodeMetadata


class RAGAdapter(IRAGAdapter):
    """RAG adapter using LlamaIndex with Qdrant and Ollama."""
    
    def __init__(self):
        """Initialize LlamaIndex with Qdrant and Ollama."""
        # Initialize Qdrant client
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
        self.collection_name = settings.qdrant_collection_name
        
        # Initialize Ollama embedding
        self.embed_model = OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=f"http://{settings.ollama_host}:{settings.ollama_port}",
        )
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=settings.llm_model,
            base_url=f"http://{settings.ollama_host}:{settings.ollama_port}",
        )

        # self.reranker = GGUFReranker(
        #     model_path=settings.reranker_model,
        #     projector_path="projector.safetensors"
        # )
        
        # Configure global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model,

        )
        
        # Initialize vector store
        self.vector_store = None
        
        print(f"RAGAdapter initialized with Ollama embedding: {settings.embedding_model}")
    
    async def initialize(self):
        """Initialize Qdrant collection and vector store."""
        collections = await self.aclient.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        # Get embedding dimension
        test_embedding = self.embed_model.get_text_embedding("test")
        embedding_dim = len(test_embedding)

        if self.collection_name not in collection_names:
            print(f"Creating collection {self.collection_name} with hybrid search...")
            # Use dictionary format instead of VectorsConfig class
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
                }
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")

        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            aclient=self.aclient,
            client=self.client,
            prefer_grpc=True, 
            collection_name=self.collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            batch_size=20
        )

        # Create or load index
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # num_docs = len(self.storage_context.docstore.docs)
        # print("Docstore size:", num_docs)

        # if num_docs == 0:
        #     print("WARNING: docstore kosong, BM25 akan error")

        collections = await self.aclient.get_collections()
        collection_info = await self.aclient.get_collection(self.collection_name)
        print(f"Total points in Qdrant: {collection_info.points_count}")
        
        print(f"LlamaIndex vector store initialized")
    
    async def retrieve(self, query: str, limit: int = 5) -> List[RetrievalResult]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List[RetrievalResult]: Relevant document nodes with scores
        """
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
            )

            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=limit * 2,  # Get more for reranking
            )

            # bm25_retriever = BM25Retriever.from_defaults(
            #     index=index,
            #     similarity_top_k=limit * 2,
            # )

            nodes = await retriever.aretrieve(query)
            # bm25_nodes = await bm25_retriever.aretrieve(query)

            # combined_nodes = self._reciprocal_rank_fusion(
            #     vector_nodes, bm25_nodes, k=60
            # )

            # reranked_nodes = await self._rerank(query, combined_nodes[:limit * 2])

            # top_nodes = reranked_nodes[:limit]
            top_nodes = nodes[:limit]

            results = [
                RetrievalResult(
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
    
    # def _reciprocal_rank_fusion(
    #     self, 
    #     list1: List, 
    #     list2: List, 
    #     k: int = 60
    # ) -> List:
        """Combine two ranked lists using RRF."""
        scores = {}
        
        # Score from list1
        for rank, item in enumerate(list1, 1):
            node_id = item.node_id if hasattr(item, 'node_id') else id(item)
            scores[node_id] = scores.get(node_id, 0) + 1 / (rank + k)
        
        # Score from list2
        for rank, item in enumerate(list2, 1):
            node_id = item.node_id if hasattr(item, 'node_id') else id(item)
            scores[node_id] = scores.get(node_id, 0) + 1 / (rank + k)
        
        # Combine and sort
        combined = {}
        for item in list1 + list2:
            node_id = item.node_id if hasattr(item, 'node_id') else id(item)
            if node_id not in combined:
                combined[node_id] = item
        
        # Sort by RRF score
        sorted_items = sorted(
            combined.values(),
            key=lambda x: scores.get(
                x.node_id if hasattr(x, 'node_id') else id(x), 
                0
            ),
            reverse=True
        )
        
        return sorted_items

    async def add_document(
        self,
        document_id: str,
        file_path: Path,
        metadata: DocumentMetadata,
        node_metadata_template: Optional[NodeMetadata] = None, 
    ) -> int:
        """Add a document to the vector store.
        
        Args:
            document_id: Unique document ID
            file_path: Path to the document file
            metadata: Document-level metadata
            node_metadata_template: Template for node metadata (optional)
            
        Returns:
            int: Number of chunks created
        """
        documents = SimpleDirectoryReader(
            input_files=[str(file_path)]
        ).load_data()

        nodes = self.node_parser.get_nodes_from_documents(documents)
        
        for idx, node in enumerate(nodes):
            node.metadata.update({
                "doc_id": document_id,
                "chunk_id": f"{document_id}_{idx:02d}",
                "position": idx,
                "source": node_metadata_template.source,
                "section": node_metadata_template.section or "",
                "document_type": node_metadata_template.document_type,
                "language": node_metadata_template.language or "id",
                "year": node_metadata_template.year,
                "tags": node_metadata_template.tags.copy() if node_metadata_template.tags else [],
            })

        VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
            transformations=[],
            store_nodes_override=True
        )

        num_chunks = len(nodes)
         
        print(f"Added document {document_id} with {num_chunks} chunks")
        
        return num_chunks
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            # Delete all points with matching doc_id
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
            # Scroll through all points and collect unique doc_ids
            document_ids = set()
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
                    doc_id = record.payload.get("doc_id")
                    if doc_id:
                        document_ids.add(doc_id)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            return list(document_ids)
        except Exception as e:
            print(f"Error getting document IDs: {e}")
            return []

