"""Pydantic schemas for the Classifier & Guardrail Bench endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

VALID_VERDICTS = {"PASS", "REVIEW", "BLOCK", "OUT_OF_SCOPE"}


class BenchCase(BaseModel):
    id: str = Field(..., min_length=1, max_length=120)
    text: str = Field(..., min_length=1, max_length=8000)
    label: Optional[str] = None
    expected_verdict: Optional[str] = None
    labels: List[str] = Field(default_factory=list)


class GuardrailCategoryConfig(BaseModel):
    instructions: str = Field(..., min_length=1)
    kind: Literal["risk", "scope"] = "risk"
    block: Optional[float] = Field(None, ge=0, le=1)
    review: Optional[float] = Field(None, ge=0, le=1)


class CreateBenchRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    mode: Literal["choice", "guardrail"]
    criteria: Dict[str, str] = Field(default_factory=dict)
    instructions: str = "Which option best describes the primary intent of the text in `state`?"
    guardrail_categories: Dict[str, GuardrailCategoryConfig] = Field(default_factory=dict)
    cases: List[BenchCase] = Field(..., min_length=1, max_length=2000)
    decision_model: str = Field(..., min_length=1)
    llm_models: List[str] = Field(default_factory=list)
    repeats: int = Field(1, ge=1, le=5)

    @model_validator(mode="after")
    def _validate_mode_shape(self) -> "CreateBenchRequest":
        if self.mode == "choice":
            if not (2 <= len(self.criteria) <= 255):
                raise ValueError("choice mode requires between 2 and 255 criteria")
            for case in self.cases:
                if case.label not in self.criteria:
                    raise ValueError(f"case '{case.id}': label '{case.label}' is not a criteria key")
        else:
            if not self.guardrail_categories:
                raise ValueError("guardrail mode requires at least one guardrail_categories entry")
            for case in self.cases:
                if case.expected_verdict not in VALID_VERDICTS:
                    raise ValueError(
                        f"case '{case.id}': expected_verdict must be one of {sorted(VALID_VERDICTS)}"
                    )
        return self


class ClassifierResult(BaseModel):
    name: str
    kind: Literal["decision", "llm"]
    model_key: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class BenchSummary(BaseModel):
    bench_id: str
    name: str
    mode: str
    status: str
    case_count: int
    classifier_names: List[str] = Field(default_factory=list)
    created_at: float
    finished_at: Optional[float] = None
    error: str = ""


class BenchDetail(BenchSummary):
    criteria: Dict[str, str] = Field(default_factory=dict)
    instructions: str = ""
    guardrail_categories: Dict[str, GuardrailCategoryConfig] = Field(default_factory=dict)
    decision_model: str = ""
    llm_models: List[str] = Field(default_factory=list)
    repeats: int = 1
    results: List[ClassifierResult] = Field(default_factory=list)


class ImportJsonlRequest(BaseModel):
    jsonl_text: str = Field(..., min_length=1, max_length=2_000_000)


class ImportJsonlResponse(BaseModel):
    cases: List[BenchCase]
