from datetime import datetime
from typing import Any
from uuid import UUID

from app.schemas.common import ORMModel, Page


class ProcessingResult(ORMModel):
    id: UUID
    filename: str
    file_type: str
    file_url: str
    processed_at: datetime
    output: dict[str, Any]


class ProcessingHistoryPage(Page[ProcessingResult]):
    pass
