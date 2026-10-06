"""File REST router: CRUD + ingest to vector store."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.config import settings
from src.drivers import dependencies
from src.drivers.rest.schemas.file_schemas import (
    FileResponse,
    IngestRequest,
    IngestResponse,
    UploadFileRequest,
)
from src.use_cases.file_use_case import (
    DeleteFileUseCase,
    IngestFileUseCase,
    ListFilesUseCase,
    UploadFileUseCase,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/file", tags=["file"])


def _to_response(file) -> FileResponse:
    return FileResponse(
        id=file.id or "",
        filename=file.filename,
        size_bytes=file.size_bytes,
        content_type=file.content_type,
        storage_path=file.storage_path,
        status=file.status,
        tags=file.tags or [],
        indexed_at=file.indexed_at.isoformat() if file.indexed_at else None,
        created_at=file.created_at.isoformat() if file.created_at else "",
        updated_at=file.updated_at.isoformat() if file.updated_at else "",
    )


@router.post("", response_model=FileResponse, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    use_case: UploadFileUseCase = Depends(dependencies.get_upload_file_use_case),
):
    """Upload file to MinIO and record metadata. File is NOT yet vectorized."""
    try:
        # One byte past the limit is enough to know it is over, and stops an
        # oversized upload being read whole into memory just to be refused.
        file_content = await file.read(settings.rag_max_file_size + 1)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading file") from e

    try:
        req = UploadFileRequest(filename=file.filename or "document", file_content=file_content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    logger.info("Uploading file: %s (%d bytes)", req.filename, len(req.file_content))
    result = await use_case.execute(file_content=req.file_content, filename=req.filename)
    return _to_response(result)


@router.get("", response_model=list[FileResponse])
async def list_files(
    use_case: ListFilesUseCase = Depends(dependencies.get_list_files_use_case),
):
    """List all files."""
    files = await use_case.execute()
    return [_to_response(f) for f in files]


@router.delete("/{file_id}", status_code=204)
async def delete_file(
    file_id: str,
    use_case: DeleteFileUseCase = Depends(dependencies.get_delete_file_use_case),
):
    """Delete file from MinIO, vector store, and DB."""
    deleted = await use_case.execute(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_files(
    request: IngestRequest,
    use_case: IngestFileUseCase = Depends(dependencies.get_ingest_file_use_case),
):
    """Vectorize file(s) into Qdrant."""
    logger.info("Ingest request: file_id=%s, tags=%s", request.file_id, request.tags)
    result = await use_case.execute(file_id=request.file_id, tags=request.tags)
    if not result["success"] and result.get("indexed", 0) == 0 and result.get("failed", 0) == 0:
        raise HTTPException(status_code=404, detail=result.get("message", "File not found"))
    return IngestResponse(**result)
