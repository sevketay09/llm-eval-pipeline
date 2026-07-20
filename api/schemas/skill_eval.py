"""Pydantic schemas for Skill Quality Lab endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SkillLintRequest(BaseModel):
    skill_text: str = Field(..., min_length=1, description="Raw SKILL.md content")


class SkillFitRequest(BaseModel):
    skill_text: str = Field(..., min_length=1)
    task_description: str = Field(..., min_length=1)
    judge_model: str = Field(..., min_length=1, description="Model key from config/models.yaml")


class SkillFullRequest(SkillFitRequest):
    save: bool = Field(True, description="Persist the combined report under reports/")
