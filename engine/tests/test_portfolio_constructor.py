"""Portfolio constructor tests (offline, pure logic)."""
from __future__ import annotations

import pytest

from engine.contracts import PortfolioTarget
from engine.portfolio_engine import (
    PortfolioConfig,
    PortfolioConstructor,
    PortfolioPlanContext,
)

EPS = 1e-9


def make_target(symbol: str, weight: float = 0.2, score: float = 1.0,
                calibrated: bool = False, **kw) -> PortfolioTarget:
    return PortfolioTarget(
        strategy_id=kw.pop("strategy_id", "测试池"),
        symbol=symbol,
        target_weight=weight,
        expected_return=kw.pop("expected_return", None),
        confidence=kw.pop("confidence", None),
        calibrated=calibrated,
        raw_score=kw.pop("raw_score", score),
        meta=kw.pop("meta", {}),
    )


def three_industries(symbol: str):
    return {"000001": "银行", "600000": "地产", "300750": "电池"}.get(symbol)


def many_targets(n: int = 30) -> list:
    symbols = [f"{600000 + i}" for i in range(n)]
    return [make_target(s, weight=0.2, score=1.0 + i * 0.001) for i, s in enumerate(symbols)]


def ten_industries(symbol: str) -> str:
    idx = int(symbol) - 600000
    return f"行业{idx % 10}"


CTX = PortfolioPlanContext(decision_date="2026-09-21", equity=1_000_000.0)


# ---------------------------------------------------------------------- #
def test_caps_respected_30_targets_10_industries():
    cfg = PortfolioConfig(industry_of=ten_industries)
    plan = PortfolioConstructor(cfg).build(many_targets(30), CTX)

    assert len(plan.allocations) <= cfg.max_positions
    for a in plan.allocations:
        assert a["weight"] <= cfg.max_weight_per_stock + EPS
        assert a["strategy_id"] == "测试池"
        assert a["calibrated"] is False

    per_bucket = {}
    for a in plan.allocations:
        b = ten_industries(a["symbol"])
        per_bucket[b] = per_bucket.get(b, 0.0) + a["weight"]
    for bucket, w in per_bucket.items():
        assert w <= cfg.max_industry_weight + 1e-6, f"bucket {bucket} breached: {w}"

    assert plan.gross_weight <= cfg.base_exposure + 1e-6
    assert abs(plan.gross_weight + plan.cash_weight - 1.0) < 1e-6
    assert plan.rule_hash


def test_max_positions_cap_binds():
    cfg = PortfolioConfig(max_positions=5, industry_of=None)
    plan = PortfolioConstructor(cfg).build(many_targets(30), CTX)
    assert len(plan.allocations) == 5
    assert any("max_positions_cap_hit=5" in n for n in plan.notes)


def test_cash_only_plan_when_no_targets():
    plan = PortfolioConstructor().build([], CTX)
    assert plan.allocations == []
    assert plan.cash_weight == 1.0
    assert plan.no_trade_reason
    assert any("cash_only" in n for n in plan.notes)


def test_drawdown_scaling_monotonic():
    ctor = PortfolioConstructor(PortfolioConfig(industry_of=three_industries))
    targets = many_targets(30)
    gross = {}
    for dd in (0.0, 0.2, 0.4, 0.6):
        ctx = PortfolioPlanContext(decision_date="2026-09-21", equity=1_000_000.0,
                                   current_drawdown=dd)
        gross[dd] = ctor.build(targets, ctx).gross_weight
    values = [gross[dd] for dd in (0.0, 0.2, 0.4, 0.6)]
    assert all(a > b for a, b in zip(values, values[1:])), gross
    # sanity: dd=0 gross == base_exposure (cap ceiling is not binding here)
    assert gross[0.0] == pytest.approx(0.8, abs=1e-6)


def test_industry_cap_skipped_note_when_no_industry_mapping():
    plan = PortfolioConstructor(PortfolioConfig(industry_of=None)).build(many_targets(5), CTX)
    assert any("industry_cap_skipped" in n for n in plan.notes)
    # still respects per-stock cap and position count
    assert all(a["weight"] <= 0.10 + EPS for a in plan.allocations)
    assert len(plan.allocations) == 5


def test_zero_and_negative_expected_return_dropped():
    targets = [
        make_target("600000", expected_return=0.02),
        make_target("600001", expected_return=0.0),
        make_target("600002", expected_return=-0.01),
        make_target("600003"),  # None -> kept
    ]
    plan = PortfolioConstructor(PortfolioConfig(max_positions=10)).build(targets, CTX)
    got = {a["symbol"] for a in plan.allocations}
    assert got == {"600000", "600003"}
    assert any("dropped_zero_or_negative_expected_return=2" in n for n in plan.notes)


def test_uncalibrated_ranked_after_calibrated_at_equal_score():
    # budget for only 2 positions; 4 targets all at score 1.0 (equal score)
    targets = [
        make_target("700001", score=1.0, calibrated=False),
        make_target("700002", score=1.0, calibrated=True, confidence=1.0),
        make_target("700003", score=1.0, calibrated=True, confidence=1.0),
        make_target("700004", score=1.0, calibrated=False),
    ]
    cfg = PortfolioConfig(max_positions=2, max_weight_per_stock=0.1)
    plan = PortfolioConstructor(cfg).build(targets, CTX)
    assert [a["symbol"] for a in plan.allocations] == ["700002", "700003"]
    assert all(a["calibrated"] for a in plan.allocations)
    assert any("uncalibrated_targets_ranked_after_calibrated" in n for n in plan.notes)


def test_min_confidence_filter_and_rule_hash_changes():
    targets = [make_target("600000", calibrated=True, confidence=0.3),
               make_target("600001", calibrated=True, confidence=0.05)]
    cfg = PortfolioConfig(min_confidence=0.2)
    plan = PortfolioConstructor(cfg).build(targets, CTX)
    assert [a["symbol"] for a in plan.allocations] == ["600000"]
    assert any("dropped_below_min_confidence=1" in n for n in plan.notes)

    base = PortfolioConfig().rule_hash()
    tighter = PortfolioConfig(max_weight_per_stock=0.05).rule_hash()
    assert base != tighter
    assert PortfolioConfig().rule_hash() == base  # deterministic


def test_all_filtered_out_is_cash_only():
    targets = [make_target("600000", expected_return=-1.0)]
    plan = PortfolioConstructor().build(targets, CTX)
    assert plan.allocations == []
    assert plan.cash_weight == 1.0
    assert plan.no_trade_reason == "all_targets_filtered"
