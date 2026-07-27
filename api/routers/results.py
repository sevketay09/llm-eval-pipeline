"""Results/Reports API."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse, PlainTextResponse

from api.schemas.evaluations import ReportCompareEntry, ReportListItem, ReportSummary
from api.schemas.rag_eval import RagReportEvalResponse
from api.services.report_service import ReportService
from api.services.rag_eval_service import RagEvalService
# Same singleton as api/routers/rag_eval.py — reused here so its embedding
# adapter cache (expensive to build for HuggingFace-backed models) is shared
# rather than rebuilt per request, unlike get_report_service() below which
# is intentionally stateless/fresh-per-request.
from api.routers.rag_eval import _service as _shared_rag_eval_service

router = APIRouter(prefix="/results", tags=["results"])


def get_rag_eval_service() -> RagEvalService:
    return _shared_rag_eval_service


def get_report_service() -> ReportService:
    return ReportService()


@router.get("/reports", response_model=list[ReportListItem])
def list_reports(
    limit: int = 50,
    svc: ReportService = Depends(get_report_service),
):
    return svc.list_reports(limit)


@router.get("/reports/{filename}", response_model=ReportSummary)
def get_report(filename: str, svc: ReportService = Depends(get_report_service)):
    report = svc.get_report(filename)
    if not report:
        raise HTTPException(404, f"Report '{filename}' not found")
    return report


@router.get("/reports/{filename}/raw")
def get_report_raw(filename: str, svc: ReportService = Depends(get_report_service)):
    data = svc.get_report_raw(filename)
    if not data:
        raise HTTPException(404, f"Report '{filename}' not found")
    return data


@router.get("/reports/{filename}/rag-eval", response_model=RagReportEvalResponse)
def get_report_rag_eval(
    filename: str,
    embedding_model: Optional[str] = None,
    svc: ReportService = Depends(get_report_service),
    rag_svc: RagEvalService = Depends(get_rag_eval_service),
):
    """Batch-score every RAG-shaped case already recorded in this report,
    aggregated per model — see analysis.rag_eval.evaluate_rag_report."""
    report = svc.get_report_raw(filename)
    if not report:
        raise HTTPException(404, f"Report '{filename}' not found")
    try:
        return rag_svc.evaluate_report(report, embedding_model=embedding_model)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Embedding scoring failed: {exc}") from exc


@router.get("/reports/{filename}/markdown", response_class=PlainTextResponse)
def get_report_markdown(filename: str, svc: ReportService = Depends(get_report_service)):
    content = svc.get_report_markdown(filename)
    if content is None:
        raise HTTPException(404, f"Report '{filename}' not found")
    return PlainTextResponse(content)


@router.get("/reports/{filename}/html", response_class=HTMLResponse)
def get_report_html(filename: str, svc: ReportService = Depends(get_report_service)):
    content = svc.get_report_html(filename)
    if content is None:
        raise HTTPException(404, f"Report '{filename}' not found")
    return HTMLResponse(content)


@router.post("/compare", response_model=dict[str, ReportCompareEntry])
def compare_reports(
    filenames: list[str],
    svc: ReportService = Depends(get_report_service),
):
    if len(filenames) < 2:
        raise HTTPException(400, "Need at least 2 reports to compare")
    result = svc.compare_reports(filenames)
    if len(result) < 2:
        raise HTTPException(400, "At least two comparable eval reports are required")
    return result
