"""Pangu 3.0 MCP Server（Phase 8）。

基于官方 mcp SDK（stdio transport）。Pangu 的全部业务能力经此暴露给宿主
Agent：pangu_health / pangu_refresh / pangu_recommend / pangu_analyze_stock /
pangu_explain / pangu_sources / pangu_run_get + 3 个 resource + 1 个 prompt。

启动：python -m engine.mcp.server   （宿主以 stdio 子进程方式拉起）
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from engine.decision.service import PanguDecisionService

server = MCPServer(name="pangu", version="3.0.0",
                   instructions=(
                       "Pangu 是 A 股研究与下一交易日决策中间层。"
                       "调用 pangu_recommend 生成下一交易日 Top20 决策候选；"
                       "所有输出为结构化证据，禁止在无数据时编造。"))

_service: PanguDecisionService | None = None


def get_service() -> PanguDecisionService:
    global _service
    if _service is None:
        _service = PanguDecisionService()
    return _service


def set_service(svc: PanguDecisionService) -> None:
    """测试注入。"""
    global _service
    _service = svc


@server.tool()
def pangu_health() -> dict:
    """返回 Pangu 版本、数据健康、最近市场日、新闻新鲜度、PIT 状态。"""
    st = get_service().status()
    return {"version": "pangu-3.0.0", **st}


@server.tool()
def pangu_refresh() -> dict:
    """强制刷新查询时上下文（清空源缓存并增量拉取）。"""
    svc = get_service()
    run = svc.recommend_next_session(limit=1, force_refresh=True, persist=False)
    return {"refreshed": True,
            "data_status": run.recommendations.data_status
            if run.recommendations else "unknown",
            "asof": run.asof_timestamp}


@server.tool()
def pangu_recommend(asof: str | None = None, limit: int = 20,
                    force_refresh: bool = False) -> dict:
    """生成下一交易日 Top20 决策候选（默认 limit=20，返回完整 DecisionRun）。"""
    run = get_service().recommend_next_session(asof=asof, limit=limit,
                                               force_refresh=force_refresh)
    return run.to_dict()


@server.tool()
def pangu_analyze_stock(code: str, asof: str | None = None) -> dict:
    """单票完整分析（code 支持 600519 / sh.600519 形式）。"""
    run = get_service().analyze_stock(code, asof=asof)
    return run.to_dict()


@server.tool()
def pangu_explain(run_id: str, code: str) -> dict:
    """按 run_id + 代码返回当时的完整结构化证据链。"""
    out = get_service().explain(run_id, code)
    if out is None:
        return {"error": "run 或代码未找到", "run_id": run_id, "code": code}
    return out


@server.tool()
def pangu_sources() -> dict:
    """查看数据源状态与 freshness（行情/新闻/公告/PIT）。"""
    return get_service().status()["sources"]


@server.tool()
def pangu_run_get(run_id: str) -> dict:
    """按 run_id 取回已生成的推荐（不重新计算）。"""
    run = get_service().runstore.load(run_id)
    if run is None:
        return {"error": "run not found", "run_id": run_id}
    return run.to_dict()


# ---------------------------------------------------------------- resources
@server.resource("pangu://recommendation/latest")
def latest_recommendation() -> str:
    run = get_service().runstore.latest()
    if run is None:
        return json.dumps({"note": "尚无已保存的推荐 run；调用 pangu_recommend 生成"},
                          ensure_ascii=False)
    return json.dumps(run.to_dict(), ensure_ascii=False)


@server.resource("pangu://sources/health")
def sources_health() -> str:
    return json.dumps(get_service().status(), ensure_ascii=False)


@server.resource("pangu://methodology")
def methodology() -> str:
    return json.dumps({
        "model": "pangu_ranker_v1",
        "factors": "经研究窗 OOS 验证的反转/非流动性/低换手/低波方向（见 docs/pangu2/STRATEGY_VALIDATION_REPORT.md）",
        "decision_time": "T 日 15:05 盘后决策，T+1 开盘执行",
        "costs": "佣金 3bp(最低5元) + 印花 5bp(卖出) + 滑点 10bp",
        "discipline": "未校准分数不称概率；数据失败禁止 BUY；所有决策可按 run_id 复原",
    }, ensure_ascii=False)


# ---------------------------------------------------------------- prompt
@server.prompt()
def pangu(limit: int = 20) -> str:
    """统一工作流：使用当前最新数据，生成下一交易日 Top20 决策候选，并以简洁格式展示。"""
    return (
        "调用 pangu_recommend（limit=20）。然后按以下格式输出，不要展开全部原始因子：\n"
        "分析时点 / 目标交易日 / 市场状态 / 数据状态（正常|DEGRADED）\n"
        "Top20 表：# | 股票 | 决策(BUY/WATCH/AVOID) | 入场区 | 持有 | 止损 | 核心理由 | 最大风险\n"
        "BUY 少于 20 是正常且诚实的；数据 DEGRADED 时如实标注。"
    )


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
