"""Skill Quality Lab API — lint / task-fit / combined skill evaluation."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.schemas.skill_eval import SkillFitRequest, SkillFullRequest, SkillLintRequest
from api.services.skill_eval_service import SkillEvalService

router = APIRouter(prefix="/skill-eval", tags=["skill-eval"])

_service = SkillEvalService()


@router.post("/lint")
def lint_skill(req: SkillLintRequest):
    return _service.lint(req.skill_text)


@router.post("/fit")
def fit_skill(req: SkillFitRequest):
    try:
        result = _service.fit(req.skill_text, req.task_description, req.judge_model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if result is None:
        raise HTTPException(status_code=502, detail="Judge output unusable (parse failed or empty input)")
    return result


@router.post("/full")
def full_skill_eval(req: SkillFullRequest):
    try:
        return _service.full(req.skill_text, req.task_description, req.judge_model, save=req.save)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/reports")
def list_skill_reports(limit: int = Query(20, ge=1, le=100)):
    return {"reports": _service.list_reports(limit=limit)}


@router.get("/reports/{filename}")
def get_skill_report(filename: str):
    report = _service.get_report(filename)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found")
    return report
