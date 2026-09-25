"""Decision contracts 单元测试（Phase 1 Task 1.1 验收）。"""
from __future__ import annotations

import json

import pytest

from engine.decision.contracts import (
    SCHEMA_VERSION,
    ConfidenceType,
    DecisionAction,
    DecisionRequest,
    DecisionRun,
    MarketStatus,
    StockDecision,
    Top20DecisionSet,
)


def _dec(code="600519", rank=1, action=DecisionAction.BUY, **kw):
    base = dict(
        rank=rank, code=code, name="贵州茅台", decision=action, score=72.5,
        confidence=0.62, confidence_type=ConfidenceType.EVIDENCE,
        execution_date="2026-09-28",
        entry_condition="开盘价进入入场区且不低于止损",
        entry_zone=[1650.0, 1680.0],
        invalid_condition="跌破 1620 或利好证伪",
        stop_loss=1620.0,
        target_zone=[1760.0, 1800.0],
        expected_holding_days=3,
        holding_range=[2, 5],
        exit_conditions=["hard_stop", "time_stop"],
        primary_strategy="reversal_rev5",
        risks=["板块扩散退潮", "流动性收缩"],
    )
    base.update(kw)
    return StockDecision(**base)


class TestDecisionRequest:
    def test_roundtrip(self):
        r = DecisionRequest(limit=10, codes=["600519"], force_refresh=True)
        r2 = DecisionRequest.from_dict(json.loads(json.dumps(r.to_dict())))
        assert r2 == r

    def test_defaults_and_validation(self):
        r = DecisionRequest()
        assert r.limit == 20 and r.market == "CN" and r.include_watchlist is True
        with pytest.raises(ValueError):
            DecisionRequest(limit=0)
        with pytest.raises(ValueError):
            DecisionRequest(limit=101)
        with pytest.raises(ValueError):
            DecisionRequest(market="US")

    def test_unknown_fields_ignored(self):
        r = DecisionRequest.from_dict({"limit": 5, "future_field": 1})
        assert r.limit == 5


class TestStockDecision:
    def test_roundtrip_full(self):
        d = _dec()
        s = json.dumps(d.to_dict(), ensure_ascii=False)
        d2 = StockDecision.from_dict(json.loads(s))
        assert d2 == d
        assert json.loads(s)["schema_version"] == SCHEMA_VERSION

    def test_illegal_action_rejected(self):
        with pytest.raises(ValueError):
            _dec(action="final")          # 旧语义状态禁止出现
        with pytest.raises(ValueError):
            _dec(action="recommend")
        with pytest.raises(ValueError):
            _dec(action=DecisionAction.BUY, rank=0)

    def test_none_and_missing_optional(self):
        d = _dec(confidence=None, entry_zone=None, stop_loss=None,
                 target_zone=None, expected_holding_days=None, holding_range=None)
        d2 = StockDecision.from_dict(d.to_dict())
        assert d2.confidence is None and d2.stop_loss is None


class TestTop20DecisionSet:
    def test_counts_and_rank_continuity(self):
        ds = Top20DecisionSet(
            asof="2026-09-26T15:05:00+08:00", execution_date="2026-09-28",
            decisions=[_dec("600519", 1), _dec("000001", 2, DecisionAction.WATCH),
                       _dec("300750", 3, DecisionAction.AVOID)],
            market_status=MarketStatus.CLOSED_AFTER,
            market_conclusion="震荡，反转方向占优")
        assert ds.counts() == {"BUY": 1, "WATCH": 1, "AVOID": 1}
        s = json.dumps(ds.to_dict(), ensure_ascii=False)
        ds2 = Top20DecisionSet.from_dict(json.loads(s))
        assert [d.code for d in ds2.decisions] == ["600519", "000001", "300750"]

    def test_gap_rank_rejected(self):
        with pytest.raises(ValueError):
            Top20DecisionSet(asof="x", execution_date="y",
                             decisions=[_dec("600519", 1), _dec("000001", 3)])

    def test_duplicate_rejected(self):
        with pytest.raises(ValueError):
            Top20DecisionSet(asof="x", execution_date="y",
                             decisions=[_dec("600519", 1), _dec("600519", 2)])

    def test_no_buy_when_data_failed(self):
        with pytest.raises(ValueError):
            Top20DecisionSet(asof="x", execution_date="y",
                             decisions=[_dec("600519", 1, DecisionAction.BUY)],
                             data_status="failed")
        # WATCH 在 failed 时仍允许（诚实降级，不编造 BUY）
        Top20DecisionSet(asof="x", execution_date="y",
                         decisions=[_dec("600519", 1, DecisionAction.WATCH)],
                         data_status="failed")

    def test_fewer_than_limit_allowed(self):
        ds = Top20DecisionSet(asof="x", execution_date="y",
                              decisions=[_dec("600519", 1)],
                              note="有效候选不足 20，如实返回 1 只")
        assert len(ds.decisions) == 1 and ds.note


class TestDecisionRun:
    def test_roundtrip(self):
        ds = Top20DecisionSet(asof="a", execution_date="b",
                              decisions=[_dec("600519", 1)])
        run = DecisionRun(
            run_id="r-20260926-001", query_timestamp="2026-09-26T15:05:00+08:00",
            decision_date="2026-09-26", execution_date="2026-09-28",
            asof_timestamp="2026-09-26T15:05:00+08:00",
            market_status=MarketStatus.CLOSED_AFTER,
            data_freshness={"daily_kline": {"age_seconds": 3600, "stale": False}},
            source_health={"all_spot": "ok"},
            recommendations=ds, warnings=["RPS 表陈旧"],
            model_version="pangu_ranker_v1", evidence_version="evidence.v1",
            latency={"total_s": 42.5},
            request=DecisionRequest(limit=20))
        s = json.dumps(run.to_dict(), ensure_ascii=False)
        run2 = DecisionRun.from_dict(json.loads(s))
        assert run2.run_id == run.run_id
        assert run2.recommendations.counts() == {"BUY": 1}
        assert run2.market_status == MarketStatus.CLOSED_AFTER
        assert run2.warnings == ["RPS 表陈旧"]

    def test_backward_compat_unknown_fields(self):
        d = _dec().to_dict()
        d["legacy_final_flag"] = True          # 未来新增字段不破坏解析
        d2 = StockDecision.from_dict(d)
        assert d2.code == "600519"
