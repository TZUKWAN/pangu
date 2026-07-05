from __future__ import annotations

from unittest.mock import MagicMock

from engine.pipeline import Pipeline, PipelineResult
from engine.quant_guard import GuardResult
from engine.recommendation_gate import RecommendationGate
from engine.strategy_pools import StrategySignal
from engine.trend_scanner import StockCandidate


class FakeDataLoader:
    pass


def test_data_ok_without_final_pick_is_not_tradable() -> None:
    pipe = Pipeline(dl=FakeDataLoader(), full_cfg={})
    data_quality, reasons = pipe._compute_data_quality(
        {
            "all_spot": {"status": "ok"},
            "daily_kline": {"status": "ok"},
            "rps": {"status": "ok"},
            "fund_flow": {"status": "ok"},
            "entry_exit": {"status": "ok"},
            "quant_guard": {"status": "ok"},
            "volume_audit": {"status": "ok"},
        },
        {"components": {"limit_up_count": 10}},
        True,
        [],
    )
    result = PipelineResult(
        date="20260703",
        sentiment={},
        boards=[],
        candidates=[],
        rejected=[],
        posture_advice="",
        final_recommendations=[],
        data_quality=data_quality,
        tradable=(data_quality == "ok" and False),
        no_trade_reason="data ok, no low-risk entry",
        block_reasons=reasons,
    )

    payload = result.to_dict()
    assert payload["data_quality"] == "ok"
    assert payload["tradable"] is False
    assert payload["no_trade_reason"] == "data ok, no low-risk entry"


def test_complete_evidence_chain_can_enter_final() -> None:
    candidate = StockCandidate(
        code="000001",
        name="Test",
        board="AI",
        close=10.0,
        pct_change=2.0,
        turnover_rate=5.0,
        circ_mv_yi=100.0,
        rps=90.0,
        rps_mode="real",
        fund_inflow_days=3,
        fund_flow_status="available",
        score=80.0,
    )
    signal = StrategySignal(
        strategy_name="trend_pullback",
        code="000001",
        name="Test",
        board="AI",
        trigger_reason="pullback support",
        score=82.0,
    )
    guard = GuardResult(kept=[candidate], watch=[], rejected=[])
    gate = RecommendationGate(
        dl=MagicMock(),
        guard_result=guard,
        market_phase={
            "market_phase": "normal",
            "allowed_strategies": ["trend_pullback"],
            "forbidden_strategies": [],
        },
        cfg={},
        recommendation_allowed=True,
    )
    item = {
        "code": "000001",
        "name": "Test",
        "entry_exit": {
            "buy_points": [{"is_primary": True, "price": 9.8, "type": "ma_pullback"}],
            "warnings": [],
        },
        "entry_style": "ma_pullback",
        "entry_plan": {
            "is_chasing": False,
            "trigger_condition": "pullback near MA20",
            "invalid_condition": "break below support",
        },
        "volume_audit": {
            "status": "ok",
            "price_volume_pattern": "pullback_shrink",
            "turnover_status": "ok",
        },
        "anti_chase": {"status": "ok", "reason": "not extended"},
        "news_evidence": {
            "sentiment_label": "bullish",
            "support_count": 2,
            "verdict_reason": "theme and stock evidence aligned",
        },
    }

    result = gate.pass_gate({"trend_pullback": [signal]}, {"000001": candidate}, candidates=[item])

    assert len(result.final_recommendations) == 1
    final = result.final_recommendations[0]
    assert final["code"] == "000001"
    assert final["gate_status"] == "final"


def test_trend_only_candidate_cannot_enter_final() -> None:
    """场景17：仅有 trend 扫描补充、无策略池信号的候选不能进入 final。"""
    candidate = StockCandidate(
        code="600000",
        name="TrendOnly",
        board="测试",
        close=10.0,
        pct_change=1.5,
        turnover_rate=4.0,
        circ_mv_yi=80.0,
        rps=85.0,
        rps_mode="real",
        fund_inflow_days=2,
        fund_flow_status="available",
        score=60.0,
    )
    guard = GuardResult(kept=[candidate], watch=[], rejected=[])
    gate = RecommendationGate(
        dl=MagicMock(),
        guard_result=guard,
        market_phase={
            "market_phase": "normal",
            "allowed_strategies": [],
            "forbidden_strategies": [],
        },
        cfg={},
        recommendation_allowed=True,
    )
    # 不传入任何 pooled_signals → 走 _judge_trend_only 分支
    item = {
        "code": "600000",
        "name": "TrendOnly",
        "entry_exit": {
            "buy_points": [{"is_primary": True, "price": 9.8}],
            "warnings": [],
        },
        "volume_audit": {"status": "ok", "price_volume_pattern": "normal"},
        "anti_chase": {"status": "ok"},
        "entry_plan": {"is_chasing": False},
        "news_evidence": {"sentiment_label": "bullish", "support_count": 2},
    }
    result = gate.pass_gate({}, {"600000": candidate}, candidates=[item])

    # 即使证据完整，trend-only 也只能进 watchlist，不能进 final
    assert not any(i["code"] == "600000" for i in result.final_recommendations)
    assert any(i["code"] == "600000" for i in result.watchlist)
    watch_item = next(i for i in result.watchlist if i["code"] == "600000")
    assert watch_item["gate_status"] == "watch"
    # gate_log 应记录 strategy_signal 拦截
    assert any(
        log.get("code") == "600000" and log.get("gate") == "strategy_signal"
        for log in result.gate_log
    )


def test_gate_result_emits_block_reasons() -> None:
    """GateResult 必须输出 block_reasons 字段，记录被拦截原因。"""
    candidate = StockCandidate(
        code="000003",
        name="Blocked",
        board="测试",
        close=10.0,
        pct_change=2.0,
        turnover_rate=5.0,
        circ_mv_yi=100.0,
        rps=90.0,
        rps_mode="real",
        fund_inflow_days=3,
        fund_flow_status="available",
        score=80.0,
    )
    signal = StrategySignal(
        strategy_name="trend_pullback",
        code="000003",
        name="Blocked",
        board="测试",
        trigger_reason="pullback",
        score=80.0,
    )
    guard = GuardResult(kept=[candidate], watch=[], rejected=[])
    gate = RecommendationGate(
        dl=MagicMock(),
        guard_result=guard,
        market_phase={
            "market_phase": "normal",
            "allowed_strategies": ["trend_pullback"],
            "forbidden_strategies": [],
        },
        cfg={},
        recommendation_allowed=True,
    )
    # 触发 anti_chase 拦截
    item = {
        "code": "000003",
        "name": "Blocked",
        "entry_exit": {"buy_points": [{"is_primary": True, "price": 9.8}], "warnings": []},
        "anti_chase": {"status": "blocked", "reason": "5 日涨幅过大"},
        "entry_plan": {"is_chasing": False},
    }
    result = gate.pass_gate({"trend_pullback": [signal]}, {"000003": candidate}, candidates=[item])

    payload = result.to_dict()
    assert "block_reasons" in payload
    assert any("anti_chase" in r for r in payload["block_reasons"])
