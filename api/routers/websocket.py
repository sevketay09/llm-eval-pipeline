"""WebSocket endpoint for real-time evaluation progress."""
from __future__ import annotations
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.config import get_settings
from api.services.eval_service import EvalService

router = APIRouter(tags=["websocket"])


def _ws_token_ok(websocket: WebSocket) -> bool:
    """Same opt-in guard as the HTTP bearer-token middleware, via ?token= query param
    since browser WebSocket clients cannot set a custom Authorization header. A no-op
    (always True) unless EVAL_API_AUTH_TOKEN is configured, so default behavior is unchanged.
    """
    token = get_settings().api_auth_token
    if not token:
        return True
    return websocket.query_params.get("token") == token

# Same singleton as evaluations router
_eval_service: EvalService | None = None


def get_eval_service() -> EvalService:
    global _eval_service
    if _eval_service is None:
        _eval_service = EvalService()
    return _eval_service


def set_eval_service(svc: EvalService):
    """Allow sharing the same EvalService instance with the evaluations router."""
    global _eval_service
    _eval_service = svc


@router.websocket("/ws/progress/{run_id}")
async def ws_progress(websocket: WebSocket, run_id: str):
    """Stream evaluation progress for a specific run."""
    if not _ws_token_ok(websocket):
        await websocket.close(code=4401, reason="Unauthorized")
        return

    svc = get_eval_service()
    run = svc.get_run(run_id)
    if not run:
        await websocket.close(code=4004, reason="Run not found")
        return

    await websocket.accept()

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    svc.subscribe_progress(run_id, queue)

    try:
        # Send current state immediately
        progress = run.to_progress()
        await websocket.send_json(progress.model_dump(mode="json"))

        # Stream updates
        while run.status == "running":
            try:
                update = await asyncio.wait_for(queue.get(), timeout=2.0)
                await websocket.send_json(update.model_dump(mode="json"))
            except asyncio.TimeoutError:
                # Send heartbeat / current state
                progress = run.to_progress()
                await websocket.send_json(progress.model_dump(mode="json"))

        # Send final state
        final = run.to_progress()
        await websocket.send_json(final.model_dump(mode="json"))

    except WebSocketDisconnect:
        pass
    finally:
        svc.unsubscribe_progress(run_id, queue)


@router.websocket("/ws/runs")
async def ws_all_runs(websocket: WebSocket):
    """Stream status updates for all active runs."""
    if not _ws_token_ok(websocket):
        await websocket.close(code=4401, reason="Unauthorized")
        return

    svc = get_eval_service()
    await websocket.accept()

    try:
        while True:
            active = svc.active_runs
            await websocket.send_json({
                "active_runs": [r.model_dump(mode="json") for r in active],
                "total_active": len(active),
            })
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
