from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ProcessingHistory


async def create_processing_history(
    db: AsyncSession,
    *,
    user_id,
    filename: str,
    file_type: str,
    file_size: int,
    storage: str,
    object_key: str,
    file_url: str | None,
    output: dict,
) -> ProcessingHistory:
    item = ProcessingHistory(
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        storage=storage,
        object_key=object_key,
        file_url=file_url,
        output=output,
    )

    db.add(item)

    try:
        await db.commit()
        await db.refresh(item)
    except Exception:
        await db.rollback()
        raise

    return item


async def get_processing_history(
    db: AsyncSession,
    *,
    user_id,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[ProcessingHistory], int]:
    total = await db.scalar(
        select(func.count())
        .select_from(ProcessingHistory)
        .where(ProcessingHistory.user_id == user_id)
    )

    result = await db.scalars(
        select(ProcessingHistory)
        .where(ProcessingHistory.user_id == user_id)
        .order_by(ProcessingHistory.created_at.desc())
        .limit(limit)
        .offset(offset)
    )

    return list(result.all()), total or 0


async def get_processing_history_item(
    db: AsyncSession,
    *,
    item_id: UUID,
    user_id,
) -> ProcessingHistory | None:
    result = await db.scalars(
        select(ProcessingHistory).where(
            ProcessingHistory.id == item_id,
            ProcessingHistory.user_id == user_id,
        )
    )
    return result.first()
