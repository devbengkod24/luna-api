"""Document management REST routers for RAG."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
import json
import logging
import re
from pathlib import Path
from fastapi.params import Query
from src.use_cases.document_management import (
    UploadDocumentUseCase,
    ListDocumentsUseCase,
    DeleteDocumentUseCase,
)
from src.drivers import dependencies
from src.drivers.rest.schemas.document import UploadDocumentResponse, DocumentResponse, DeleteDocumentResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"])

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}
MAX_FILENAME_LENGTH = 255
MAX_STRING_FIELD_LENGTH = 500

DANGEROUS_PATTERNS = [
    r"\.\.",  # Path traversal
    r"[/\\]",  # Path separators
    r"[\x00-\x1f]",  # Control characters
]

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues."""
    if not filename:
        return "document"
    
    # Remove dangerous patterns
    sanitized = filename
    for pattern in DANGEROUS_PATTERNS:
        sanitized = re.sub(pattern, "_", sanitized)
    
    # Limit length
    if len(sanitized) > MAX_FILENAME_LENGTH:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_length = MAX_FILENAME_LENGTH - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized.strip() or "document"


def validate_filename(filename: Optional[str]) -> str:
    """Validate and sanitize filename."""
    if not filename or not filename.strip():
        raise HTTPException(
            status_code=400,
            detail="Filename cannot be empty"
        )
    
    sanitized = sanitize_filename(filename)
    
    # Check extension
    ext = Path(sanitized).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File format not supported. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    return sanitized


def validate_string_field(value: str, field_name: str, max_length: int = MAX_STRING_FIELD_LENGTH) -> None:
    """Validate string field length."""
    if value and len(value) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} too long. Maximum {max_length} characters"
        )

@router.post("/upload", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
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
    validated_filename = validate_filename(file.filename)
    
    # Validate metadata fields
    validate_string_field(source, "Source")
    validate_string_field(document_type, "Document type")
    validate_string_field(section, "Section")
    
    # Parse tags
    tag_list = []
    if tags:
        try:
            # Try parsing as JSON first
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                tag_list = [str(tag).strip() for tag in parsed if tag and str(tag).strip()]
            elif isinstance(parsed, str):
                tag_list = [parsed.strip()] if parsed.strip() else []
            else:
                tag_list = []
        except json.JSONDecodeError:
            # If not JSON, treat as comma-separated
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        except Exception as e:
            logger.warning(f"Error parsing tags: {e}")
            raise HTTPException(
                status_code=400,
                detail="Invalid tags format. Use JSON array or comma-separated values"
            )

    for tag in tag_list:
        if len(tag) > MAX_STRING_FIELD_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Tag too long. Maximum {MAX_STRING_FIELD_LENGTH} characters per tag"
            )

    try:
        file_content = await file.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="Error reading file"
        )

    if len(file_content) == 0:
        raise HTTPException(
            status_code=400,
            detail="File cannot be empty"
        )
    
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )
    
    logger.info(
        f"Uploading document - Filename: {validated_filename}, "
        f"Size: {len(file_content)} bytes, "
        f"Tags: {len(tag_list)} tags"
    )
    
    try:
        result = await use_case.execute(
            file_content=file_content,
            filename=validated_filename,
            tags=tag_list if tag_list else None,
            source=source.strip(),
            document_type=document_type.strip(),
            section=section.strip(),
            language=language.lower(),
            year=year,
        )
        
        if not result["success"]:
            logger.warning(f"Upload failed: {result.get('message')}")
            raise HTTPException(status_code=400, detail=result["message"])
        
        logger.info(f"Document uploaded successfully - ID: {result.get('document_id')}")
        return UploadDocumentResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while uploading document"
        )


@router.get("/list", response_model=List[DocumentResponse])
async def list_documents(
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum number of documents to return"),
    offset: Optional[int] = Query(0, ge=0, description="Number of documents to skip"),
    use_case: ListDocumentsUseCase = Depends(dependencies.get_list_documents_use_case),
):
    """List all documents in RAG.
    
    Returns a list of all documents with their metadata.
    """
    try:
        documents = await use_case.execute()
        
        # Apply pagination if requested
        if limit is not None:
            total = len(documents)
            documents = documents[offset:offset + limit]
            logger.info(
                f"Listed {len(documents)} document(s) "
                f"(offset={offset}, limit={limit}, total={total})"
            )
        else:
            logger.info(f"Listed {len(documents)} document(s)")
        
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
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


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
    if not document_id or len(document_id) < 10:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID format"
        )
    
    logger.info(f"Deleting document - ID: {document_id}")
    
    try:
        result = await use_case.execute(document_id)
        
        if not result["success"]:
            logger.warning(f"Delete failed: {result.get('message')}")
            raise HTTPException(status_code=404, detail=result["message"])
        
        logger.info(f"Document deleted successfully - ID: {document_id}")
        return DeleteDocumentResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while deleting document"
        )
