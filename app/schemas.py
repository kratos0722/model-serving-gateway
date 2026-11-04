"""Pydantic v2 request/response models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        ..., description="Feature name -> numeric value. Must cover the model's "
        "feature_order."
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Stable identity used for A/B assignment and canary hashing.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Force a specific model version (bypasses routing). "
        "Unknown versions return a sanitized 404.",
    )


class PredictResponse(BaseModel):
    model_name: str
    version_served: str
    variant: Optional[str] = None
    probability: float
    label: int
    shadowed: bool
    shadow_version: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    version: str
    feature_order: list[str]
    weights: list[float]
    bias: float
    description: str = ""


class HealthResponse(BaseModel):
    status: str
    models_loaded: int


class ErrorResponse(BaseModel):
    error: str
    detail: str
