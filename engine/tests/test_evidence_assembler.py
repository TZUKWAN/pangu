from __future__ import annotations

from engine.evidence_assembler import EvidenceAssembler
from engine.pipeline import PipelineResult


def test_evidence_assembler_builds_required_sections():
    assembler = EvidenceAssembler()
    evidence = assembler.assemble(
        candidates=[{"code": "000001", "name": "测试股", "close": 10.0, "pct_change": 1.0}],
        final_recommendations=[],
        watchlist=[{"code": "000001", "gate_status": "watch", "watch_reason": "无量突破"}],
        rejected=[],
        strategy_signals={"trend_pullback": [{"code": "000001", "score": 80, "theme": "算力"}]},
        source_status={"all_spot": {"status": "ok"}, "daily_kline": {"status": "ok"}},
        news_evidence={"000001": {"sentiment_label": "neutral", "verdict_reason": "无新闻证据"}},
        market_phase={"market_phase": "震荡"},
        data_quality="ok",
    )

    ev = evidence["000001"]
    for key in (
        "strategy",
        "data_quality",
        "price_action",
        "volume_audit",
        "liquidity",
        "news_evidence",
        "anti_chase",
        "entry_plan",
        "decision",
        "raw",
    ):
        assert key in ev
    assert ev["strategy"]["strategy_name"] == "trend_pullback"
    assert ev["decision"]["status"] == "watch"
    assert "无量突破" in ev["decision"]["reasons"]


def test_pipeline_result_exposes_counts_and_candidate_evidence():
    result = PipelineResult(
        date="20260703",
        sentiment={},
        boards=[],
        candidates=[{"code": "000001"}],
        rejected=[],
        posture_advice="",
        final_recommendations=[],
        watchlist=[{"code": "000001"}],
        candidate_evidence={"000001": {"code": "000001", "decision": {"status": "watch"}}},
        data_quality="ok",
        tradable=False,
        no_trade_reason="数据完整，但无低风险买点",
    )
    payload = result.to_dict()

    assert payload["final_count"] == 0
    assert payload["watch_count"] == 1
    assert payload["raw_candidate_count"] == 1
    assert payload["candidate_evidence"]["000001"]["decision"]["status"] == "watch"
