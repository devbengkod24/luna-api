"""Document management REST routers for RAG knowledge base."""

import logging
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.params import Query
from fastapi.responses import Response

from src.drivers import dependencies
from src.drivers.rest.schemas.knowledge import (
    DeleteKnowledgeResponse,
    KnowledgeResponse,
    UploadKnowledgeRequest,
    UploadKnowledgeResponse,
)
from src.use_cases.knowledge import (
    DeleteKnowledgeUseCase,
    DownloadKnowledgeUseCase,
    ListKnowledgeUseCase,
    UploadKnowledgeUseCase,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])

@router.post("/upload", response_model=UploadKnowledgeResponse)
async def upload_knowledge(
    file: UploadFile = File(...),
    use_case: UploadKnowledgeUseCase = Depends(dependencies.get_upload_knowledge_use_case),
):
    """Upload a knowledge to RAG.
    
    Supported formats: PDF, TXT, DOCX, DOC
    Maximum file size: 50MB 
    """
    try:
        file_content = await file.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="Error reading file"
        )
    
    try:
        document_request = UploadKnowledgeRequest(
            filename=file.filename or "document",
            file_content=file_content,
        )
    except Exception as e:
        logger.warning(f"Validation failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    logger.info(
        f"Uploading document - Filename: {document_request.filename}, "
        f"Size: {len(document_request.file_content)} bytes"
    )
    
    try:
        result = await use_case.execute(
            file_content=document_request.file_content,
            filename=document_request.filename,
        )
        
        if not result["success"]:
            logger.warning(f"Upload failed: {result.get('message')}")
            raise HTTPException(status_code=400, detail=result["message"])
        
        logger.info(f"Document uploaded successfully - Filename: {result.get('filename')}")
        return UploadKnowledgeResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while uploading document"
        )


@router.get("/list", response_model=list[KnowledgeResponse])
async def list_knowledge(
    limit: int | None = Query(None, ge=1, le=1000, description="Maximum number of knowledge to return"),
    offset: int | None = Query(0, ge=0, description="Number of knowledge to skip"),
    use_case: ListKnowledgeUseCase = Depends(dependencies.get_list_knowledge_use_case),
):
    """List all knowledge in RAG."""
    try:
        knowledge_list = await use_case.execute()
        
        if limit is not None:
            total = len(knowledge_list)
            knowledge_list = knowledge_list[offset:offset + limit]
            logger.info(
                f"Listed {len(knowledge_list)} knowledge(s) "
                f"(offset={offset}, limit={limit}, total={total})"
            )
        else:
            logger.info(f"Listed {len(knowledge_list)} knowledge(s)")
        
        return [
            KnowledgeResponse(
                document_id=doc.get("document_id", ""),
                file_name=doc.get("file_name", ""),
                file_type=doc.get("file_type", ""),
                file_size=doc.get("file_size", 0),
                creation_date=doc.get("creation_date", ""),
                download_url=doc.get("download_url", ""),
            )
            for doc in knowledge_list
        ]
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/download/{filename:path}")
async def download_knowledge(
    filename: str,
    use_case: DownloadKnowledgeUseCase = Depends(dependencies.get_download_knowledge_use_case),
):
    """Download a knowledge file by filename."""
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    
    logger.info(f"Downloading knowledge - Filename: {filename}")
    
    try:
        result = await use_case.execute(filename)
        
        if not result["success"]:
            logger.warning(f"Download failed: {result.get('message')}")
            raise HTTPException(status_code=404, detail=result["message"])
        
        return Response(
            content=result["file_content"],
            media_type=result["content_type"],
            headers={
                "Content-Disposition": f'attachment; filename="{result["filename"]}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while downloading document"
        )


@router.delete("/{document_id}", response_model=DeleteKnowledgeResponse)
async def delete_knowledge(
    document_id: str,
    use_case: DeleteKnowledgeUseCase = Depends(dependencies.get_delete_knowledge_use_case),
):
    """Delete a knowledge from RAG."""
    if not document_id or len(document_id) < 10:
        raise HTTPException(
            status_code=400,
            detail="Invalid knowledge ID format"
        )
    
    logger.info(f"Deleting knowledge - ID: {document_id}")
    
    try:
        result = await use_case.execute(document_id)
        
        if not result["success"]:
            logger.warning(f"Delete failed: {result.get('message')}")
            raise HTTPException(status_code=404, detail=result["message"])
        
        logger.info(f"Knowledge deleted successfully - ID: {document_id}")
        return DeleteKnowledgeResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while deleting document"
        )
