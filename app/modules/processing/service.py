from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.modules.processing.repository import (
    create_processing_history,
    get_processing_history,
    get_processing_history_item,
)
from app.modules.processing.schemas import ProcessingResult
from app.services.storage import storage_service
from whitman.parsing import parse_upload
from whitman.pipeline import build_catalog


class ProcessingService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def process_file(self, *, user_id, file: UploadFile):
        raw_name = file.filename or "upload"
        filename = raw_name.replace("\\", "/").rsplit("/", 1)[-1]

        if (
            not filename
            or filename in {".", ".."}
            or len(filename) > 255
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename",
            )

        data = await file.read()
        await file.close()

        if not data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File exceeds the "
                    f"{settings.max_upload_size_mb} MB upload limit"
                ),
            )

        try:
            rows = parse_upload(filename, data)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from None
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="Unable to parse uploaded file",
            ) from None

        output = build_catalog(rows)

        object_key = (
            f"users/{user_id}/{uuid4().hex}/{filename}"
        )
        content_type = (
            file.content_type or "application/octet-stream"
        )

        try:
            backend, file_url = await storage_service.save(
                object_key, data, content_type
            )
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="Unable to store uploaded file",
            ) from None

        try:
            item = await create_processing_history(
                self.db,
                user_id=user_id,
                filename=filename,
                file_type=filename.rsplit(".", 1)[-1].lower(),
                file_size=len(data),
                storage=backend,
                object_key=object_key,
                file_url=file_url,
                output=output,
            )
        except Exception:
            try:
                await storage_service.delete(object_key)
            except Exception:
                # Log this failure and schedule cleanup.
                pass

            raise HTTPException(
                status_code=500,
                detail="Unable to save processing history",
            ) from None

        return ProcessingResult.model_validate(item)

    async def reprocess(self, *, user_id, item_id: UUID):
        item = await get_processing_history_item(
            self.db,
            item_id=item_id,
            user_id=user_id,
        )

        if item is None:
            raise HTTPException(
                status_code=404,
                detail="Processing history item not found",
            )

        try:
            data = await storage_service.read(item.object_key)
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="Stored file is no longer available",
            ) from None

        try:
            rows = parse_upload(item.filename, data)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from None
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="Unable to parse stored file",
            ) from None

        output = build_catalog(rows)

        try:
            new_item = await create_processing_history(
                self.db,
                user_id=user_id,
                filename=item.filename,
                file_type=item.file_type,
                file_size=len(data),
                storage=item.storage,
                object_key=item.object_key,
                file_url=item.file_url,
                output=output,
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Unable to save processing history",
            ) from None

        return ProcessingResult.model_validate(new_item)

    async def get_history(
        self,
        *,
        user_id,
        limit: int = 20,
        offset: int = 0,
    ):
        items, total = await get_processing_history(
            self.db,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        return {
            "data": [
                ProcessingResult.model_validate(item)
                for item in items
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
