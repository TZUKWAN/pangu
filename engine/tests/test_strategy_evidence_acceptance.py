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
