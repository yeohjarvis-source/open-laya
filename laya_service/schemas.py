from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


class Question(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["choice", "score", "noul"]
    instructions: str = Field(min_length=1, max_length=20_000)
    criteria: list[str] | dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_criteria(self):
        kind, value = self.type, self.criteria
        if kind == "choice" and (not value or not isinstance(value, (list, dict))):
            raise ValueError("choice criteria must be a non-empty list or object")
        if kind == "score" and (not value or not isinstance(value, list)):
            raise ValueError("score criteria must be a non-empty list")
        if kind == "noul" and value is not None and not isinstance(value, dict):
            raise ValueError("noul criteria must be an object or omitted")
        if isinstance(value, list) and len(set(value)) != len(value):
            raise ValueError("criteria labels must be unique")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1, max_length=100)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    dataset_eligible: bool = True
    model: str = Field(default="laya", min_length=1, max_length=100)


class PredictResponse(BaseModel):
    request_id: str
    model: str
    result: dict[str, Any]
    usage: dict[str, int]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: Literal["correct", "incorrect", "partially_correct", "unknown"]
    corrected_answers: dict[str, Any] | None = None
    notes: str | None = Field(default=None, max_length=10_000)


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    external_id: str | None = Field(default=None, max_length=200)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class UserPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    active: bool | None = None
    metadata: dict[str, JsonValue] | None = None


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    expires_at: datetime | None = None
    allowed_models: list[str] | None = None


class ApiKeyPatch(BaseModel):
    allowed_models: list[str] = Field(min_length=1)


class LogQuery(BaseModel):
    user_id: str | None = None
    status: str | None = None
    dataset_eligible: bool | None = None
    start: datetime | None = None
    end: datetime | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
