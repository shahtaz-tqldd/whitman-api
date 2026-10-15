from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    identifier: str = Field(
        min_length=3, max_length=320, description="Username or email address"
    )
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
