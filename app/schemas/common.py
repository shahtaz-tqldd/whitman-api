from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Message(BaseModel):
    message: str


class MetaData(BaseModel):
    total: int
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class Page(BaseModel, Generic[T]):
    meta: MetaData
    data: list[T]

    @model_validator(mode="before")
    @classmethod
    def arrange_metadata(cls, data: Any) -> Any:
        if isinstance(data, dict) and "meta" not in data:
            data = data.copy()
            data["meta"] = {
                "total": data.pop("total"),
                "limit": data.pop("limit", 50),
                "offset": data.pop("offset", 0),
            }

        return data