from typing import Annotated

from fastapi import Depends

from app.core.dependencies import DbSession
from app.modules.processing.service import ProcessingService


def get_processing_service(
    db: DbSession,
) -> ProcessingService:
    return ProcessingService(db)


ProcessingServiceDep = Annotated[
    ProcessingService,
    Depends(get_processing_service),
]
