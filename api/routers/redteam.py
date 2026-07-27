"""Red-team API."""
from __future__ import annotations

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException

from api.schemas.redteam import CreateSessionRequest, SessionDetail, SessionSummary
from api.services.redteam_service import RedTeamService

router = APIRouter(prefix="/redteam", tags=["redteam"])

_service = RedTeamService()


def get_service() -> RedTeamService:
    return _service


@router.post("", response_model=SessionSummary, status_code=201)
def create_session(
    req: CreateSessionRequest,
    svc: Annotated[RedTeamService, Depends(get_service)],
):
    session = svc.create(system_prompt=req.system_prompt, categories=req.categories, model_key=req.model_key)
    return svc.to_summary(session)


@router.get("", response_model=List[SessionSummary])
def list_sessions(svc: Annotated[RedTeamService, Depends(get_service)]):
    return [svc.to_summary(s) for s in svc.list()]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(
    session_id: str,
    svc: Annotated[RedTeamService, Depends(get_service)],
):
    session = svc.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return svc.to_detail(session)


@router.post("/{session_id}/run", status_code=202, response_model=SessionSummary)
async def run_session(
    session_id: str,
    svc: Annotated[RedTeamService, Depends(get_service)],
):
    session = svc.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    if session.status == "running":
        raise HTTPException(409, "Session is already running")
    session = await svc.run(session_id)
    return svc.to_summary(session)


@router.get("/{session_id}/results", response_model=SessionDetail)
def get_results(
    session_id: str,
    svc: Annotated[RedTeamService, Depends(get_service)],
):
    session = svc.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    if session.status not in ("done", "error"):
        raise HTTPException(409, f"Session not finished (status={session.status})")
    return svc.to_detail(session)
