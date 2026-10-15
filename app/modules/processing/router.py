from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Query, UploadFile, status

from app.core.dependencies import CurrentUser
from app.modules.processing.dependencies import ProcessingServiceDep
from app.modules.processing.schemas import (
    ProcessingHistoryPage,
    ProcessingResult,
)

router = APIRouter(prefix="/processing", tags=["Processing"])


@router.post(
    "",
    response_model=ProcessingResult,
    status_code=status.HTTP_201_CREATED,
)
async def process_file(
    user: CurrentUser,
    service: ProcessingServiceDep,
    file: Annotated[
        UploadFile,
        File(description="Products data: csv, xlsx or json"),
    ],
) -> ProcessingResult:
    return await service.process_file(
        user_id=user.id,
        file=file,
    )


@router.post(
    "/{item_id}/reprocess",
    response_model=ProcessingResult,
    status_code=status.HTTP_201_CREATED,
)
async def reprocess_file(
    user: CurrentUser,
    service: ProcessingServiceDep,
    item_id: UUID,
) -> ProcessingResult:
    return await service.reprocess(
        user_id=user.id,
        item_id=item_id,
    )


@router.get(
    "/history",
    response_model=ProcessingHistoryPage,
)
async def processing_history(
    user: CurrentUser,
    service: ProcessingServiceDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ProcessingHistoryPage:
    return await service.get_history(
        user_id=user.id,
        limit=limit,
        offset=offset,
    )
