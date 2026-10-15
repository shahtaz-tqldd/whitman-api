from datetime import datetime
from typing import Any
from uuid import UUID

from app.schemas.common import ORMModel, Page


class ProcessingResult(ORMModel):
    id: UUID
    filename: str
    file_type: str
    file_size: int
    storage: str
    file_url: str
    output: dict[str, Any]
    processed_at: datetime


class ProcessingHistoryPage(Page[ProcessingResult]):
    pass
