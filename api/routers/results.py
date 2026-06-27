"""Results/Reports API."""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse, PlainTextResponse

from api.schemas.evaluations import ReportCompareEntry, ReportListItem, ReportSummary
from api.services.report_service import ReportService

router = APIRouter(prefix="/results", tags=["results"])


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
    return svc.compare_reports(filenames)
