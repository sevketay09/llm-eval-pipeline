"""Pydantic schemas for Skill Quality Lab endpoints."""
from __future__ import annotations

from typing import List, Literal, Union

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
