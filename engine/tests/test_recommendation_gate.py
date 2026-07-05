"""推荐闸门单元测试：覆盖新闻证据审计、反追涨、追价买点拦截。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from engine.quant_guard import GuardResult
from engine.recommendation_gate import RecommendationGate
from engine.strategy_pools import StrategySignal
from engine.trend_scanner import StockCandidate


def _candidate(code: str = "000001", name: str = "测试", board: str = "机器人") -> StockCandidate:
    return StockCandidate(
        code=code, name=name, board=board, close=10.0, pct_change=2.0,
        turnover_rate=5.0, circ_mv_yi=100.0, rps=90.0, rps_mode="real",
        fund_inflow_days=3, fund_flow_status="available", score=80.0,
    )


def _signal(code: str = "000001", score: float = 80.0, strategy: str = "趋势突破") -> StrategySignal:
    return StrategySignal(
        strategy_name=strategy, code=code, name="测试", board="机器人",
        trigger_reason="突破", score=score,
    )


def _entry_exit() -> dict:
    return {
        "buy_points": [{"is_primary": True, "price": 10.0, "type": "低吸", "condition": "回踩均线"}],
        "stop_loss": {"price": 9.0, "method": "均线"},
        "take_profit": [{"price": 11.0, "method": "1:1"}],
        "warnings": [],
    }


def _gate(
    candidates: list[dict[str, object]] | None = None,
    pooled: dict[str, list[StrategySignal]] | None = None,
    phase: dict[str, object] | None = None,
    recommendation_allowed: bool = True,
) -> RecommendationGate:
    dl = MagicMock()
    guard = GuardResult(kept=[_candidate()], watch=[], rejected=[])
    return RecommendationGate(
        dl=dl,
        guard_result=guard,
        market_phase=phase or {"market_phase": "震荡", "allowed_strategies": ["趋势突破"], "forbidden_strategies": []},
        cfg={},
        recommendation_allowed=recommendation_allowed,
    )


def test_news_evidence_bearish_rejects() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {
            "sentiment_label": "bearish",
            "verdict_reason": "业绩变脸",
            "risk_events": ["业绩变脸"],
        },
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.rejected)
    assert any(log["gate"] == "news_evidence" for log in res.gate_log)


def test_news_evidence_mixed_with_risk_watch() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {
            "sentiment_label": "mixed",
            "verdict_reason": "多空交织",
            "risk_events": ["减持公告"],
        },
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.watchlist)


def test_news_evidence_bullish_passes() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {
            "sentiment_label": "bullish",
            "verdict_reason": "利多",
            "support_count": 3,
        },
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.final_recommendations)


def test_anti_chase_blocked_rejects() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "blocked", "reason": "已加速 3 天"},
        "entry_plan": {"is_chasing": False},
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.rejected)


def test_entry_plan_chasing_watch() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": True, "trigger_condition": "突破追高"},
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" for i in res.watchlist)


def test_breakout_confirm_deviation_watch() -> None:
    gate = _gate()
    item = {
        "code": "000001", "name": "测试", "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_style": "breakout_confirm",
        "entry_plan": {"is_chasing": False, "trigger_price": 10.0, "current_price": 10.2},
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])
    assert any(i["code"] == "000001" and i.get("watch_reason", "").startswith("突破确认") for i in res.watchlist)


def test_volume_audit_missing_goes_watch() -> None:
    gate = _gate()
    item = {
        "code": "000001",
        "name": "测试",
        "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "volume_audit": {
            "status": "missing",
            "price_volume_pattern": "missing",
            "reason": "量能数据缺失，不能进入正式推荐",
        },
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    pooled = {"趋势突破": [_signal()]}
    cand_map = {"000001": _candidate()}
    res = gate.pass_gate(pooled, cand_map, candidates=[item])

    assert res.final_recommendations == []
    assert any(i["code"] == "000001" and "量能审计不足" in i.get("watch_reason", "") for i in res.watchlist)


def test_fund_flow_unavailable_does_not_block_normal_strategy() -> None:
    gate = _gate()
    cand = _candidate()
    cand.fund_flow_status = "unavailable"
    item = {
        "code": "000001",
        "name": "测试",
        "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "volume_audit": {"status": "ok", "price_volume_pattern": "pullback_shrink"},
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    res = gate.pass_gate({"趋势突破": [_signal(strategy="趋势突破")]}, {"000001": cand}, candidates=[item])

    assert any(i["code"] == "000001" for i in res.final_recommendations)
    assert res.final_recommendations[0]["fund_flow_risk"].startswith("资金流状态 unavailable")


def test_fund_flow_unavailable_blocks_fund_flow_strategy() -> None:
    gate = _gate(phase={"market_phase": "震荡", "allowed_strategies": ["主力资金"], "forbidden_strategies": []})
    cand = _candidate()
    cand.fund_flow_status = "unavailable"
    item = {
        "code": "000001",
        "name": "测试",
        "entry_exit": _entry_exit(),
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "volume_audit": {"status": "ok", "price_volume_pattern": "pullback_shrink"},
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    res = gate.pass_gate({"主力资金": [_signal(strategy="主力资金")]}, {"000001": cand}, candidates=[item])

    assert res.final_recommendations == []
    assert any("策略强依赖资金流" in i.get("watch_reason", "") for i in res.watchlist)
