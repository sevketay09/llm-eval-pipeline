"""Pydantic schemas for model configuration."""
from pydantic import BaseModel, Field
from typing import Optional


class ModelConfig(BaseModel):
    provider: str = Field(
        ...,
        description="Provider: openai, anthropic, ollama, lmstudio, vllm, typesafe, typesafe-mock",
    )
    model_name: str = Field(..., description="Model name or deployment name")
    api_key: Optional[str] = Field(None, description="API key or ${ENV_VAR} reference")
    base_url: Optional[str] = Field(None, description="Base URL for the model endpoint")
    api_version: Optional[str] = Field(None, description="API version (Azure)")
    max_tokens: int = Field(4096, ge=256, le=32768)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    supports_function_calling: bool = False
    supports_streaming: bool = True
    supports_response_format: bool = True
    quirks: Optional[list[str]] = None
    timeout_s: Optional[float] = Field(None, gt=0, le=120, description="Decision model request timeout")
    cost_per_mtok_input: Optional[float] = Field(None, ge=0, description="Decision model input cost, $/MTok")
    cost_per_mtok_output: Optional[float] = Field(None, ge=0, description="Decision model output cost, $/MTok")


class ModelConfigUpdate(BaseModel):
    provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=256, le=32768)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    supports_function_calling: Optional[bool] = None
    supports_streaming: Optional[bool] = None
    supports_response_format: Optional[bool] = None
    quirks: Optional[list[str]] = None
    timeout_s: Optional[float] = Field(None, gt=0, le=120)
    cost_per_mtok_input: Optional[float] = Field(None, ge=0)
    cost_per_mtok_output: Optional[float] = Field(None, ge=0)


class ModelListResponse(BaseModel):
    models: dict[str, ModelConfig]
    total: int
