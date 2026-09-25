"""Phase 10 渲染器测试。"""
from __future__ import annotations

import pytest

from engine.decision.contracts import (DecisionAction, ConfidenceType,
                                       MarketStatus, DecisionRun,
                                       StockDecision, Top20DecisionSet)
from engine.decision.render import render_status, render_top20, render_single, render_why


def _run():
    dec = StockDecision(
        rank=1, code="600519", name="贵州茅台", decision=DecisionAction.BUY,
        score=71.2, confidence=0.71, confidence_type=ConfidenceType.EVIDENCE,
        execution_date="2026-09-28", entry_condition="开盘入区",
        entry_zone=[1650.0, 1680.0], invalid_condition="跌破1620",
        stop_loss=1620.0, target_zone=[1760.0], expected_holding_days=3,
        holding_range=[2, 5], exit_conditions=["hard_stop@1620.00",
                                               "time_stop@5d"],
        primary_strategy="reversal_rev5",
        reasons=["风险调整分 71 ≥ 65", "事件催化强度 0.31"],
        risks=["板块扩散退潮"],
        score_breakdown={"factor_alpha": 60.1, "event_alpha": 58.3})
    tset = Top20DecisionSet(asof="2026-09-25T15:05:00+08:00",
                            execution_date="2026-09-28", decisions=[dec],
                            market_status=MarketStatus.CLOSED_AFTER,
                            market_conclusion="震荡，反转方向占优",
                            data_status="ok")
    return DecisionRun(run_id="pangu-20260928-abc", query_timestamp="q",
                       decision_date="2026-09-25", execution_date="2026-09-28",
                       asof_timestamp="2026-09-25T15:05:00+08:00",
                       market_status=MarketStatus.CLOSED_AFTER,
                       recommendations=tset,
                       data_freshness={"daily_kline": {"stale": False}},
                       source_health={"all_spot": "ok"},
                       model_version="pangu_ranker_v1")


class TestRender:
    def test_top20_first_screen_is_minimal(self):
        text = render_top20(_run())
        assert "目标交易日：2026-09-28" in text
        assert "市场状态：震荡，反转方向占优" in text
        assert "数据状态：正常" in text
        assert "600519 贵州茅台" in text
        assert "BUY" in text and "1650.00-1680.00" in text and "1620.00" in text
        # 禁止信息过载：不应出现 OMS/实验登记/原始因子名
        for banned in ("OMS", "order", "experiment", "score_breakdown"):
            assert banned not in text

    def test_single_seven_elements(self):
        text = render_single(_run(), "600519")
        for key in ("决策：BUY", "下一交易日计划", "建议持有：3", "最重要理由",
                    "最大风险", "止损：1620.0", "退出条件"):
            assert key in text

    def test_why_full_evidence(self):
        run = _run()
        d = run.recommendations.decisions[0].to_dict()
        text = render_why(run.to_dict(), d)
        assert "分数分解" in text and "factor_alpha" in text
        assert "pangu_ranker_v1" in text
        assert "confidence_type" in text or "evidence" in text

    def test_status_lists_sources(self):
        text = render_status({"asof": "a", "sources": {
            "realtime_quote": {"fetched_at": "t", "age_seconds": 12,
                               "stale": False, "quality": "ok"}},
            "pit_archive": {"max_date": "2026-08-13"},
            "mcp": {"transport": "stdio"}})
        assert "realtime_quote" in text and "stdio" in text
