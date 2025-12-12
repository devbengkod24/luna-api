# RAG System Setup dengan LlamaIndex

## 📋 Overview

Sistem RAG (Retrieval Augmented Generation) menggunakan LlamaIndex dengan:
- **Vector Store**: Qdrant
- **Embeddings**: Ollama (bge-m3)
- **LLM**: Ollama (sahabatai)
- **Document Loaders**: PDF, TXT, DOCX, HTML, MD

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Konfigurasi Environment

Tambahkan di `.env`:
```env
# RAG Configuration
QDRANT_HOST=qdrant
QDRANT_GRPC_PORT=6334
QDRANT_COLLECTION_NAME=vsa_collection

# Ollama
OLLAMA_HOST=ollama
OLLAMA_PORT=11434

# LLM Models
LLM_MODEL=csalab/sahabatai1:llama3_instruct_Q4_K_M
EMBEDDING_MODEL=bge-m3:latest
```

### 3. Start Services

```bash
docker-compose up -d
```

### 4. Load Existing Documents (Optional)

Untuk load dokumen yang sudah ada di `storage/rag/`:

```bash
python scripts/load_rag_documents.py
```

## 📁 Struktur Folder

```
storage/
├── rag/
│   ├── documents/      # Uploaded documents (managed by API)
│   ├── metadata/       # Document metadata JSON files
│   └── [sources]/      # Existing documents (optional, for batch loading)
```

## 🔌 API Endpoints

### Upload Document

```bash
POST /v1/documents/upload
Content-Type: multipart/form-data

Body:
- file: <file binary>

Response:
{
  "success": true,
  "message": "Dokumen berhasil diupload",
  "document_id": "uuid",
  "filename": "document.pdf",
  "num_chunks": 42
}
```

### List Documents

```bash
GET /v1/documents/list

Response:
[
  {
    "id": "uuid",
    "filename": "document.pdf",
    "file_type": "pdf",
    "file_size": 1024000,
    "uploaded_at": "2025-01-01T00:00:00",
    "num_chunks": 42
  }
]
```

### Delete Document

```bash
DELETE /v1/documents/{document_id}

Response:
{
  "success": true,
  "message": "Dokumen berhasil dihapus"
}
```

### Search Documents

```bash
POST /v1/documents/search
Content-Type: application/json

Body:
{
  "query": "Apa itu beasiswa?",
  "limit": 5
}

Response:
[
  {
    "content": "Beasiswa adalah...",
    "score": 0.89,
    "document_id": "uuid",
    "metadata": {
      "filename": "beasiswa.pdf",
      "file_type": "pdf"
    }
  }
]
```

## 📝 Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Via pypdf |
| Text | `.txt` | Plain text |
| Word | `.docx`, `.doc` | Via python-docx |
| HTML | `.html`, `.htm` | Via beautifulsoup4 |
| Markdown | `.md` | Native support |

## 🔧 Configuration

### Chunk Settings

Edit di `src/adapters/qdrant/llamaindex_rag_adapter.py`:

```python
Settings.chunk_size = 512  # Size of each chunk
Settings.chunk_overlap = 50  # Overlap between chunks
```

### Embedding Model

Untuk ganti embedding model, edit di `.env`:
```env
EMBEDDING_MODEL=nomic-embed-text:latest  # Alternative model
```

Pastikan model sudah di-pull:
```bash
docker exec ollama ollama pull nomic-embed-text:latest
```

## 🧪 Testing

### Test Upload

```bash
curl -X POST "http://localhost:8000/v1/documents/upload" \
  -F "file=@/path/to/document.pdf"
```

### Test Search

```bash
curl -X POST "http://localhost:8000/v1/documents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "apa itu beasiswa?",
    "limit": 3
  }'
```

## 🐛 Troubleshooting

### Error: Model not found

```bash
# Pull embedding model
docker exec ollama ollama pull bge-m3:latest

# Pull LLM model
docker exec ollama ollama pull csalab/sahabatai1:llama3_instruct_Q4_K_M
```

### Error: Cannot connect to Qdrant

Check if Qdrant is running:
```bash
docker ps | grep qdrant
docker logs qdrant
```

### Error: File too large

Max file size is 50MB. Edit `MAX_FILE_SIZE` in `src/use_cases/document_management.py`:

```python
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
```

### Error: Document not indexed

Check logs for embedding/chunking errors:
```bash
docker logs fastapi
```

## 🔍 Integration dengan Chat

RAG results bisa diintegrasikan ke chat LLM via `ChatTextUseCase`:

```python
# Di chat_text.py
if self.rag_adapter:
    results = await self.rag_adapter.search(prompt, limit=3)
    # Add to context
    context_text = "\n".join([r.content for r in results])
```

## 📊 Monitoring

### Check Collection Stats

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6334)
collection_info = client.get_collection("vsa_collection")
print(f"Total points: {collection_info.points_count}")
```

### View Embeddings Dimension

```bash
# Check in logs when RAG initializes
docker logs fastapi | grep "embedding dimension"
```

## 🎯 Best Practices

1. **Pre-process documents**: Remove headers/footers, clean OCR noise
2. **Chunk size**: 512 tokens works well for most cases
3. **Overlap**: 50 tokens prevents context loss
4. **Query expansion**: Rephrase user queries for better results
5. **Metadata**: Add rich metadata for better filtering
6. **Regular cleanup**: Delete outdated documents periodically

## 🔐 Security

1. Validate file extensions before processing
2. Scan uploaded files for malware
3. Implement rate limiting on upload endpoint
4. Add authentication/authorization
5. Sanitize HTML content before indexing

## 📚 Additional Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ollama Models](https://ollama.com/library)

