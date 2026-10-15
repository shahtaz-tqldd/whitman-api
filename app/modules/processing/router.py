from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select

from app.core.config import settings
from app.core.dependencies import CurrentUser, DbSession
from app.db.models import ProcessingHistory
from app.modules.processing.schemas import ProcessingHistoryPage, ProcessingResult
from app.modules.processing.services.base import build_catalog, parse_upload
from app.schemas.common import Page
from app.services.storage import storage_service

router = APIRouter(prefix="/processing", tags=["Processing"])


@router.post("", response_model=ProcessingResult, status_code=status.HTTP_201_CREATED)
async def process_file(
    db: DbSession,
    user: CurrentUser,
    file: Annotated[UploadFile, File(description="Products data: csv, xlsx or json")],
) -> ProcessingResult:
    raw_name = file.filename or "upload"
    safe_name = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
    if not safe_name or safe_name in {".", ".."} or len(safe_name) > 255:
        raise HTTPException(status_code=400, detail="Invalid filename")

    data = await file.read()
    await file.close()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_size_mb} MB upload limit",
        )

    try:
        rows = parse_upload(safe_name, data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    except Exception:
        raise HTTPException(status_code=422, detail="Unable to parse uploaded file") from None

    output = build_catalog(rows)

    from uuid import uuid4

    object_key = f"users/{user.id}/{uuid4().hex}/{safe_name}"
    content_type = file.content_type or "application/octet-stream"
    try:
        backend, file_url = await storage_service.save(object_key, data, content_type)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Unable to store uploaded file: {e}") from None

    item = ProcessingHistory(
        user_id=user.id,
        filename=safe_name,
        file_type=safe_name.rsplit(".", 1)[-1].lower(),
        file_size=len(data),
        storage=backend,
        object_key=object_key,
        file_url=file_url,
        output=output,
    )
    db.add(item)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        await storage_service.delete(object_key)
        raise HTTPException(status_code=500, detail="Unable to save processing history") from None
    await db.refresh(item)
    return ProcessingResult.model_validate(item)


@router.get("/history", response_model=ProcessingHistoryPage)
async def processing_history(
    db: DbSession,
    user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> Page[ProcessingResult]:
    total = await db.scalar(
        select(func.count())
        .select_from(ProcessingHistory)
        .where(ProcessingHistory.user_id == user.id)
    )
    items = await db.scalars(
        select(ProcessingHistory)
        .where(ProcessingHistory.user_id == user.id)
        .order_by(ProcessingHistory.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return Page(
        items=[ProcessingResult.model_validate(item) for item in items],
        total=total or 0,
        limit=limit,
        offset=offset,
    )



"""
Processing File:
1. Parse Upload: CSV / XLSX / JSON → product records
2. Normalize and Validate: Column names, blank values, duplicates, malformed rows
3. Profile Products: Candidate categories, attributes, value distributions
4. Discover Category Schema: Identify dynamic attributes and their types
5. Build Catalog snapshot: Normalized attribute values, counts, and schema metadata
6. Persist history + result: Save the generated catalog, not the source file

"""