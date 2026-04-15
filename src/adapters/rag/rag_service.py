"""LlamaIndex RAG adapter: retrieval + vector store (Qdrant + Ollama).

Pipeline retrieval (terinspirasi GPT4All database.cpp):
  1. Over-fetch dense candidates (embedding/vector search)
  2. Over-fetch sparse candidates (BM25)
  3. Reciprocal Rank Fusion (RRF) — gabungkan ranking dense + sparse
  4. Cross-encoder Rerank (opsional, env USE_RERANKER)
  5. Return top-K
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
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
from src.ports.services import IRAGService
from src.shared.exceptions import ServiceUnavailableException

logger = logging.getLogger(__name__)


# ── Stop words untuk RRF bobot BM25 dinamis ──────────────────────────────────
# Daftar stop words dari GPT4All database.cpp baris ~2515
# Digunakan untuk menghitung jumlah kata bermakna dalam query
STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by",
        "for", "if", "in", "into", "is", "it", "no", "not", "of",
        "on", "or", "such", "that", "the", "their", "then", "there",
        "these", "they", "this", "to", "was", "will", "with",
        # Tambahan stop words bahasa Indonesia
        "di", "ke", "dari", "yang", "dan", "ini", "itu", "untuk",
        "dengan", "pada", "adalah", "atau", "juga", "akan", "sudah",
        "telah", "oleh", "apa", "siapa", "dimana", "kapan",
        "bagaimana", "mengapa", "berapa",
    }
)

# ── Konstanta RRF ─────────────────────────────────────────────────────────────
# k=60 dari paper: "Reciprocal Rank Fusion outperforms Condorcet and
# individual Rank Learning Methods" (doi: 10.1145/1571941.157211)
RRF_K = 60

# Token-based chunking: 512 token per chunk, tanpa overlap (ref: indexing.py)
CHUNK_SIZE = 512


def _compute_bm25_weight(query: str) -> float:
    """Hitung bobot BM25 dinamis berdasarkan panjang query bermakna.

    Mirroring logika GPT4All database.cpp baris 2556-2582:
    - Query panjang (banyak kata bermakna) → bobot BM25 rendah, embedding dominan
    - Query pendek (1-2 kata) → bobot BM25 tinggi, keyword matching dominan
    """
    meaningful_words = [w for w in query.lower().split() if w not in STOP_WORDS]
    n = max(len(meaningful_words), 1)
    query_length_weight = 1.0 / (n**2)
    bm_weight = 0.25 + query_length_weight * 0.50
    return min(bm_weight, 0.75)  # clamp agar tidak terlalu dominan


def reciprocal_rank_fusion(
    dense_nodes: list,
    sparse_nodes: list,
    query: str,
    k: int = RRF_K,
) -> list:
    """Reciprocal Rank Fusion (RRF) — gabungkan ranking dense dan sparse.

    Formula:  score_rrf = w_bm25 × 1/(k + rank_sparse) + w_emb × 1/(k + rank_dense)

    Mirroring GPT4All ``reciprocalRankFusion()`` di database.cpp.

    Args:
        dense_nodes: Hasil retrieval dari embedding/vector search.
        sparse_nodes: Hasil retrieval dari BM25/sparse search.
        query: Query string untuk menentukan dynamic BM25 weight.
        k: Konstanta RRF (default 60 dari paper).

    Returns:
        List of nodes sorted by RRF score (descending).
    """
    bm_weight = _compute_bm25_weight(query)
    emb_weight = 1.0 - bm_weight

    node_map: Dict[str, object] = {}
    dense_rank: Dict[str, int] = {}
    sparse_rank: Dict[str, int] = {}

    for rank, node in enumerate(dense_nodes):
        nid = node.node_id
        dense_rank[nid] = rank
        node_map[nid] = node

    for rank, node in enumerate(sparse_nodes):
        nid = node.node_id
        sparse_rank[nid] = rank
        if nid not in node_map:
            node_map[nid] = node

    max_dense = len(dense_nodes)
    max_sparse = len(sparse_nodes)

    scored: list[Tuple[str, float]] = []
    for nid in node_map:
        d_rank = dense_rank.get(nid, max_dense)
        s_rank = sparse_rank.get(nid, max_sparse)
        emb_score = 1.0 / (k + d_rank)
        bm25_score = 1.0 / (k + s_rank)
        rrf_score = bm_weight * bm25_score + emb_weight * emb_score
        scored.append((nid, rrf_score))

    scored.sort(key=lambda x: x[1], reverse=True)

    result = []
    for nid, rrf_score in scored:
        node = node_map[nid]
        node.score = rrf_score
        result.append(node)

    return result


class LlamaIndexRAGService(IRAGService):
    """RAG service: retrieval + indexing menggunakan LlamaIndex + Qdrant.

    Pipeline retrieval (terinspirasi GPT4All):
      1. Over-fetch dense candidates (embedding/vector search)
      2. Over-fetch sparse candidates (BM25)
      3. Reciprocal Rank Fusion (RRF)
      4. Cross-encoder Rerank (opsional)
      5. Return top-K
    """

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
        # Agar retrieval pakai Ollama embed, bukan default OpenAI
        Settings.embed_model = self.embed_model

        # Cross-encoder reranker: download otomatis dari HuggingFace, berjalan lokal
        # device="cpu" → agar tidak rebutan VRAM dengan Ollama (LLM tetap di GPU)
        if settings.use_reranker:
            self.reranker = SentenceTransformerRerank(
                model=settings.reranker_model,
                top_n=8,
                device="cpu",
            )
        else:
            self.reranker = None

        self.vector_store: QdrantVectorStore | None = None
        self.storage_context: StorageContext | None = None

        logger.info(
            "RagService initialized — embedding: %s, reranker: %s",
            settings.ollama_embedding_model,
            settings.reranker_model if settings.use_reranker else "disabled",
        )

    async def initialize(self) -> None:
        """Initialize Qdrant collection dan vector store. Dipanggil saat app startup.

        Jika collection sudah ada tapi dimensi vektornya tidak sesuai dengan
        embedding model aktif, collection akan di-drop dan dibuat ulang otomatis.
        """
        test_embedding = self.embed_model.get_text_embedding("test")
        embedding_dim = len(test_embedding)

        collections = await self.aclient.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name in collection_names:
            # Cek apakah dimensi cocok
            try:
                info = await self.aclient.get_collection(self.collection_name)
                vectors_config = info.config.params.vectors
                existing_dim = None
                if isinstance(vectors_config, dict):
                    dense_cfg = vectors_config.get("text-dense")
                    if dense_cfg:
                        existing_dim = dense_cfg.size
                else:
                    # single-vector collection (bukan named vectors)
                    existing_dim = getattr(vectors_config, "size", None)

                if existing_dim is not None and existing_dim != embedding_dim:
                    logger.warning(
                        "Collection '%s' memiliki dim=%d, tapi embedding model "
                        "menghasilkan dim=%d. Menghapus dan membuat ulang...",
                        self.collection_name, existing_dim, embedding_dim,
                    )
                    await self.aclient.delete_collection(self.collection_name)
                    collection_names.remove(self.collection_name)
            except Exception as e:
                logger.warning("Gagal membaca info collection: %s", e)

        if self.collection_name not in collection_names:
            logger.info(
                "Membuat collection '%s' (dim=%d)...",
                self.collection_name, embedding_dim,
            )
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

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
        )

        collection_info = await self.aclient.get_collection(self.collection_name)
        logger.info(
            "RagService initialized. Points in Qdrant: %d", collection_info.points_count
        )

    async def index_document(
        self,
        file_id: str,
        file_path: str,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Index dokumen ke vector store dengan token-based chunking (512 token, no overlap).

        Menggantikan SemanticSplitter dengan TokenTextSplitter sesuai reference experiment.
        """
        if not self.storage_context:
            raise ServiceUnavailableException("RagService not initialized. Call initialize() first.")

        # Baca file sebagai plain text
        try:
            text = Path(file_path).read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            logger.warning("Empty file: %s", file_path)
            return False

        # Token-based chunking (ref: indexing.py)
        splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
        chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]

        import uuid
        filename = Path(file_path).name
        rel_path = Path(file_path).name  # store just filename as relpath

        nodes = []
        total_chunks = len(chunks)
        for i, chunk_text in enumerate(chunks):
            node_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_id}::chunk_{i}"))
            node = TextNode(
                id_=node_id,
                text=chunk_text,
                metadata={
                    "file_id": file_id,
                    "file_name": filename,
                    "file_path": rel_path,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                },
            )
            if tags:
                node.metadata["tags"] = tags
            nodes.append(node)

        VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            store_nodes_override=True,
            embed_model=self.embed_model,
            show_progress=True,
        )

        logger.info(
            "Indexed %d chunks (chunk_size=%d tokens) for file_id=%s",
            len(nodes), CHUNK_SIZE, file_id,
        )
        return True

    async def retrieve(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[dict]:
        """Hybrid retrieval dengan Reciprocal Rank Fusion (RRF) + optional reranker.

        Pipeline (terinspirasi GPT4All database.cpp:2589-2601):
          1. Over-fetch: ambil limit*3 kandidat via dense (embedding) search
          2. Over-fetch: ambil limit*3 kandidat via hybrid (dense + BM25) search
          3. RRF Fusion: scoring ulang dengan bobot BM25 dinamis
          4. Cross-encoder Rerank (jika USE_RERANKER=true)
          5. Return top `limit` hasil

        Tags parameter diabaikan (sesuai keputusan implementasi — no tag filtering).
        """
        if not self.vector_store or not self.storage_context:
            raise ServiceUnavailableException("RagService not initialized. Call initialize() first.")

        fetch_k = limit * 3  # Over-fetch factor

        # ── Step 1: Dense retrieval (embedding only) ──────────────────────────
        dense_store = QdrantVectorStore(
            aclient=self.aclient,
            client=self.client,
            prefer_grpc=True,
            collection_name=self.collection_name,
            enable_hybrid=False,  # dense only
            batch_size=20,
        )
        dense_index = VectorStoreIndex.from_vector_store(
            vector_store=dense_store,
            embed_model=self.embed_model,
        )
        dense_retriever = VectorIndexRetriever(
            index=dense_index,
            similarity_top_k=fetch_k,
        )
        dense_nodes = await dense_retriever.aretrieve(query)

        # ── Step 2: Hybrid retrieval (dense + BM25 sparse via Qdrant) ─────────
        hybrid_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
        )
        hybrid_retriever = VectorIndexRetriever(
            index=hybrid_index,
            similarity_top_k=fetch_k,
        )
        sparse_nodes = await hybrid_retriever.aretrieve(query)

        # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────────
        fused_nodes = reciprocal_rank_fusion(
            dense_nodes=dense_nodes,
            sparse_nodes=sparse_nodes,
            query=query,
        )

        # ── Step 4: Cross-encoder Rerank (jika aktif) ─────────────────────────
        if self.reranker:
            candidates = fused_nodes[:fetch_k]
            final_nodes = self.reranker.postprocess_nodes(candidates, query_str=query)
        else:
            final_nodes = fused_nodes

        # ── Step 5: Ambil top `limit` hasil ───────────────────────────────────
        results: List[dict] = []
        for n in final_nodes[:limit]:
            content = getattr(n, "text", None) or n.get_content()
            score = getattr(n, "score", 0.0) or 0.0
            results.append({"content": content, "score": float(score)})

        logger.info(
            "RAG retrieve '%s' → %d nodes (dense=%d, sparse=%d, fused=%d)",
            query, len(results), len(dense_nodes), len(sparse_nodes), len(fused_nodes),
        )
        return results

    async def delete_by_file_id(self, file_id: str) -> bool:
        """Delete semua vectors yang terkait dengan file_id."""
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
