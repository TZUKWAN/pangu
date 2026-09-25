"""Pangu 3.0 决策 API（Phase 11：Web 降级为辅助界面）。

默认界面只保留 今日Top20 / 个股详情 / 系统状态；研究能力（执行/策略/研究
登记）保留在 Advanced 分组，不删除。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class AnalyzeBody(BaseModel):
    code: str
    asof: str | None = None
    limit: int = 20


def _svc():
    from engine.decision.service import PanguDecisionService
    return PanguDecisionService()


@router.get("/api/decision/status")
async def api_decision_status():
    """/pangu status 的 Web 版：行情/新闻/公告/PIT/MCP。"""
    try:
        return _svc().status()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"decision status unavailable: {e!r}"[:200])


@router.post("/api/decision/analyze")
async def api_decision_analyze(body: AnalyzeBody):
    try:
        run = _svc().analyze_stock(body.code, asof=body.asof)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"analyze failed: {e!r}"[:200])
    out = run.to_dict()
    # 附带极简渲染文本，前端直接展示
    from engine.decision.render import render_single
    out["rendered"] = render_single(run, body.code)
    return out


@router.post("/api/decision/recommend")
async def api_decision_recommend(limit: int = 20, asof: str | None = None):
    try:
        run = _svc().recommend_next_session(limit=limit, asof=asof)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"recommend failed: {e!r}"[:200])
    from engine.decision.render import render_top20
    out = run.to_dict()
    out["rendered"] = render_top20(run)
    return out
