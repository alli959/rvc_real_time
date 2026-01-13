"""Common request/response models."""

from typing import Optional, List, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    current_model: Optional[str] = Field(default=None, description="Currently loaded model name")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    code: Optional[str] = Field(default=None, description="Error code")


class ModelInfo(BaseModel):
    """Voice model information."""
    name: str = Field(..., description="Model name")
    path: str = Field(..., description="Model file path")
    index_path: Optional[str] = Field(default=None, description="Index file path")
    sample_rate: int = Field(default=32000, description="Model sample rate")
    version: str = Field(default="v2", description="Model version")


class ModelListResponse(BaseModel):
    """Model list response."""
    models: List[ModelInfo] = Field(..., description="Available models")
    total: int = Field(..., description="Total number of models")


class PresetsResponse(BaseModel):
    """Quality presets response."""
    presets: dict = Field(..., description="Available quality presets")


class StatusResponse(BaseModel):
    """Generic status response."""
    success: bool = Field(..., description="Operation success")
    message: str = Field(..., description="Status message")
    data: Optional[Any] = Field(default=None, description="Additional data")
