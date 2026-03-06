"""File management REST router: CRUD + ingest to vector DB."""

import logging
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.params import Query
from fastapi.responses import Response

from src.drivers import dependencies
from src.drivers.rest.schemas.file import (
    FileDeleteResponse,
    FileIngestResponse,
    FileListItem,
    FileUploadResponse,
    UploadFileRequest,
)
from src.use_cases.knowledge import (
    DeleteKnowledgeUseCase,
    DownloadKnowledgeUseCase,
    ListKnowledgeUseCase,
    UploadKnowledgeUseCase,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/files", tags=["files"])


# ----- Create (upload) -----


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    use_case: UploadKnowledgeUseCase = Depends(dependencies.get_upload_knowledge_use_case),
):
    """Upload a file to storage and index to vector DB.
    Supported formats: PDF, TXT, DOCX, DOC. Max size: 50MB.
    """
    try:
        file_content = await file.read()
    except Exception as e:
        logger.error("Error reading file: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail="Error reading file") from e

    try:
        req = UploadFileRequest(
            filename=file.filename or "document",
            file_content=file_content,
        )
    except Exception as e:
        logger.warning("Validation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e

    logger.info(
        "Uploading file: %s, size=%s bytes",
        req.filename,
        len(req.file_content),
    )

    try:
        result = await use_case.execute(
            file_content=req.file_content,
            filename=req.filename,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "Upload failed"))
        return FileUploadResponse(
            success=True,
            message=result.get("message", ""),
            filename=result.get("filename"),
            num_chunks=result.get("num_chunks"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error uploading file: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while uploading file") from e


# ----- Ingest (store to vector DB) -----


@router.post("/ingest", response_model=FileIngestResponse, status_code=202)
async def ingest_file(
    file: UploadFile = File(...),
    use_case: UploadKnowledgeUseCase = Depends(dependencies.get_upload_knowledge_use_case),
):
    """Ingest a file into vector DB (save to storage + index). Same flow as upload."""
    try:
        file_content = await file.read()
    except Exception as e:
        logger.error("Error reading file for ingest: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail="Error reading file") from e

    try:
        req = UploadFileRequest(
            filename=file.filename or "document",
            file_content=file_content,
        )
    except Exception as e:
        logger.warning("Validation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e

    logger.info("Ingesting file to vector DB: %s", req.filename)

    try:
        result = await use_case.execute(
            file_content=req.file_content,
            filename=req.filename,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "Ingest failed"))
        return FileIngestResponse(
            success=True,
            message=result.get("message", "File ingested to vector DB."),
            filename=result.get("filename"),
            num_chunks=result.get("num_chunks"),
            status="indexed",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error ingesting file: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while ingesting file") from e


# ----- Read (list) -----


@router.get("/list", response_model=list[FileListItem])
async def list_files(
    limit: int | None = Query(None, ge=1, le=1000, description="Max items to return"),
    offset: int | None = Query(0, ge=0, description="Number of items to skip"),
    use_case: ListKnowledgeUseCase = Depends(dependencies.get_list_knowledge_use_case),
):
    """List all files in RAG (storage + vector store)."""
    try:
        items = await use_case.execute()
        total = len(items)
        if limit is not None:
            items = items[offset : offset + limit]
        else:
            items = items[offset:]

        out = []
        for doc in items:
            file_name = doc.get("file_name", "")
            out.append(
                FileListItem(
                    document_id=doc.get("document_id", ""),
                    file_name=file_name,
                    file_type=doc.get("file_type", ""),
                    file_size=doc.get("file_size", 0),
                    creation_date=doc.get("creation_date", ""),
                    download_url=f"/files/download/{quote(file_name, safe='')}",
                )
            )
        logger.info("Listed %s file(s) (offset=%s, limit=%s, total=%s)", len(out), offset, limit, total)
        return out
    except Exception as e:
        logger.error("Error listing files: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e


# ----- Read (download) -----


@router.get("/download/{filename:path}")
async def download_file(
    filename: str,
    use_case: DownloadKnowledgeUseCase = Depends(dependencies.get_download_knowledge_use_case),
):
    """Download a file by filename."""
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename is required")

    logger.info("Downloading file: %s", filename)

    try:
        result = await use_case.execute(filename)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message", "Not found"))
        return Response(
            content=result["file_content"],
            media_type=result.get("content_type", "application/octet-stream"),
            headers={
                "Content-Disposition": f'attachment; filename="{result.get("filename", filename)}"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error downloading file: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while downloading file") from e


# ----- Delete -----


@router.delete("/{document_id}", response_model=FileDeleteResponse)
async def delete_file(
    document_id: str,
    use_case: DeleteKnowledgeUseCase = Depends(dependencies.get_delete_knowledge_use_case),
):
    """Delete a file from storage and vector DB by document_id."""
    if not document_id or not document_id.strip():
        raise HTTPException(status_code=400, detail="Document ID is required")

    logger.info("Deleting file: %s", document_id)

    try:
        result = await use_case.execute(document_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message", "Not found"))
        return FileDeleteResponse(
            success=True,
            message=result.get("message", ""),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting file: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while deleting file") from e
