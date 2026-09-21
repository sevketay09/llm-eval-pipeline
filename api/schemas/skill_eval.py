"""Pydantic schemas for Skill Quality Lab endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple, Union

from pydantic import BaseModel, Field


class SkillLintRequest(BaseModel):
    skill_text: str = Field(..., min_length=1, description="Raw SKILL.md content")


class SkillFitRequest(BaseModel):
    skill_text: str = Field(..., min_length=1)
    task_description: str = Field(..., min_length=1)
    judge_model: str = Field(..., min_length=1, description="Model key from config/models.yaml")


class SkillFullRequest(SkillFitRequest):
    save: bool = Field(True, description="Persist the combined report under reports/")


class TriggerPrompt(BaseModel):
    text: str = Field(..., min_length=1)
    expected: Union[bool, Literal["ambiguous"]] = Field(
        ..., description="true = should trigger, false = should not, 'ambiguous' = borderline"
    )


class SkillTriggerRequest(BaseModel):
    skill_text: str = Field(..., min_length=1)
    judge_model: str = Field(..., min_length=1, description="Model key used to simulate routing")
    prompts: List[TriggerPrompt] = Field(..., min_length=1)
    repeats: int = Field(1, ge=1, le=5, description="Trials per prompt (majority vote)")


class RouteSkillEntry(BaseModel):
    name: str = Field(..., min_length=1)
    skill_text: str = Field(..., min_length=1)


class RoutePrompt(BaseModel):
    text: str = Field(..., min_length=1)
    expected: str = Field(..., min_length=1, description="Skill name, or 'none'")


class SkillRouteRequest(BaseModel):
    skills: List[RouteSkillEntry] = Field(..., min_length=2)
    prompts: List[RoutePrompt] = Field(..., min_length=1)
    decision_model: str = Field(..., min_length=1, description="Jev model key from config/models.yaml")


class SkillRouteResult(BaseModel):
    text: str
    expected: str
    predicted: str
    confidence: float
    ranked: List[Tuple[str, float]]


class SkillRouteResponse(BaseModel):
    skills: List[str]
    metrics: Dict[str, Any]
    results: List[SkillRouteResult]
    none_label: str
