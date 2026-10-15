from datetime import timedelta

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.core.config import settings
from app.core.dependencies import DbSession
from app.core.security import create_token, verify_password
from app.db.models import User
from app.modules.auth.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DbSession) -> TokenResponse:
    identifier = body.identifier.strip().lower()
    user = await db.scalar(
        select(User).where((User.username == identifier) | (User.email == identifier))
    )
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username/email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return TokenResponse(
        access_token=create_token(
            str(user.id), "access", timedelta(minutes=settings.access_token_expire_minutes)
        ),
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )
