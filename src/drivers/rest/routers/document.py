"""Document management REST routers for RAG."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from src.use_cases.document_management import (
    UploadDocumentUseCase,
    ListDocumentsUseCase,
    DeleteDocumentUseCase,
)
from src.drivers import dependencies
from src.drivers.rest.schemas.document import UploadDocumentResponse, DocumentResponse, DeleteDocumentResponse

router = APIRouter(prefix="/v1/documents", tags=["documents"])

@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),  # JSON string or comma-separated
    source: str = Form(""),
    document_type: str = Form(""),
    section: str = Form(""),
    language: str = Form("id"),
    year: Optional[int] = Form(None),
    use_case: UploadDocumentUseCase = Depends(dependencies.get_upload_document_use_case),
):
    """Upload a document to RAG.
    
    Supported formats: PDF, TXT, DOCX, HTML, MD
    Maximum file size: 50MB
    
    The document will be:
    1. Saved to file storage
    2. Processed and chunked
    3. Embedded and indexed in vector database
    
    Metadata fields (optional):
    - tags: Comma-separated tags or JSON array
    - source: Source system/repository name
    - document_type: Document function/purpose (not file format)
    - section: Document section
    - language: Language code (default: "id")
    - year: Year of document
    """
    import json
    
    # Parse tags
    tag_list = []
    if tags:
        try:
            # Try parsing as JSON first
            tag_list = json.loads(tags)
            if not isinstance(tag_list, list):
                tag_list = [tags]
        except:
            # If not JSON, treat as comma-separated
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    # Read file content
    file_content = await file.read()
    
    # Process upload with metadata
    result = await use_case.execute(
        file_content=file_content,
        filename=file.filename,
        tags=tag_list,
        source=source,
        document_type=document_type,
        section=section,
        language=language,
        year=year,
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return UploadDocumentResponse(**result)


@router.get("/list", response_model=List[DocumentResponse])
async def list_documents(
    use_case: ListDocumentsUseCase = Depends(dependencies.get_list_documents_use_case),
):
    """List all documents in RAG.
    
    Returns a list of all documents with their metadata.
    """
    documents = await use_case.execute()
    
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.metadata.filename,
            file_type=doc.metadata.file_type.value,
            file_size=doc.metadata.file_size,
            uploaded_at=doc.metadata.uploaded_at.isoformat(),
            num_chunks=doc.metadata.num_chunks,
        )
        for doc in documents
    ]


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str,
    use_case: DeleteDocumentUseCase = Depends(dependencies.get_delete_document_use_case),
):
    """Delete a document from RAG.
    
    Removes the document from:
    1. Vector database
    2. File storage
    """
    result = await use_case.execute(document_id)
    
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    
    return DeleteDocumentResponse(**result)
