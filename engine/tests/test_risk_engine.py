"""Risk engine + portfolio state tests (offline, tmp SQLite db)."""
from __future__ import annotations

import pytest

from engine.contracts import ExecutionMode, RiskDecision
from engine.portfolio_engine import (
    PortfolioPlan,
    PortfolioPlanContext,
    PortfolioConstructor,
    PortfolioConfig,
    PortfolioState,
    RiskContext,
    RiskEngine,
    RiskEngineConfig,
)

BASE_TARGET = {"symbol": "600000", "side": "BUY", "weight": 0.05, "notional": 50_000.0,
               "strategy_id": "测试池"}
BASE_ADV = {"600000": 50_000_000.0}


def make_ctx(**over) -> RiskContext:
    vals = dict(daily_pnl=0.0, equity=1_000_000.0, drawdown=0.0,
                adv=BASE_ADV, tradable=True, mode=ExecutionMode.PAPER)
    vals.update(over)
    return RiskContext(**vals)


def make_engine(**cfg_over) -> RiskEngine:
    return RiskEngine(RiskEngineConfig(**cfg_over))


def check(decision: RiskDecision, name: str) -> dict:
    assert name in decision.checks, f"missing check '{name}': {decision.checks}"
    return decision.checks[name]


# ---------------------------------------------------------------------- #
def test_notional_cap_blocks():
    d = make_engine(max_order_notional=100_000.0).assess_target(
        {**BASE_TARGET, "notional": 150_000.0}, make_ctx())
    assert not d.approved
    assert "notional_cap" in d.reason
    assert check(d, "notional_cap")["passed"] is False
    assert check(d, "weight_cap")["passed"] is True


def test_weight_cap_blocks():
    d = make_engine(max_weight_per_stock=0.10).assess_target(
        {**BASE_TARGET, "weight": 0.25, "notional": 50_000.0}, make_ctx())
    assert not d.approved
    assert check(d, "weight_cap")["passed"] is False


def test_liquidity_floor_blocks_thin_names():
    eng = make_engine(min_liquidity_amount=10_000_000.0)
    d = eng.assess_target(BASE_TARGET, make_ctx(adv={"600000": 5_000_000.0}))
    assert not d.approved
    assert check(d, "liquidity_floor")["passed"] is False
    d_ok = eng.assess_target(BASE_TARGET, make_ctx(adv={"600000": 10_000_000.0}))
    assert check(d_ok, "liquidity_floor")["passed"] is True


def test_participation_cap_blocks_too_big_slice():
    eng = make_engine(participation_cap=0.05, min_liquidity_amount=100_000.0)
    # adv 200k passes the floor, but 5% * 200k = 10k ceiling < 50k order.
    d = eng.assess_target(BASE_TARGET, make_ctx(adv={"600000": 200_000.0}))
    assert not d.approved
    assert check(d, "participation_cap")["passed"] is False
    assert check(d, "liquidity_floor")["passed"] is True


def test_daily_loss_soft_blocks_buys_allows_sells():
    eng = make_engine(daily_loss_soft=-0.02, daily_loss_hard=-0.04)
    ctx = make_ctx(daily_pnl=-0.03)
    buy = eng.assess_target(BASE_TARGET, ctx)
    assert not buy.approved
    assert check(buy, "daily_loss_soft")["passed"] is False
    assert check(buy, "daily_loss_hard")["passed"] is True
    sell = eng.assess_target({**BASE_TARGET, "side": "SELL"}, ctx)
    assert sell.approved, sell.reason
    assert check(sell, "daily_loss_soft")["passed"] is True


def test_daily_loss_hard_blocks_everything():
    eng = make_engine(daily_loss_hard=-0.04)
    ctx = make_ctx(daily_pnl=-0.05)
    buy = eng.assess_target(BASE_TARGET, ctx)
    sell = eng.assess_target({**BASE_TARGET, "side": "SELL"}, ctx)
    assert not buy.approved and not sell.approved
    assert check(buy, "daily_loss_hard")["passed"] is False
    assert check(sell, "daily_loss_hard")["passed"] is False


def test_drawdown_cap_blocks_buys_allows_sells():
    eng = make_engine(max_drawdown=0.15)
    ctx = make_ctx(drawdown=0.20)
    buy = eng.assess_target(BASE_TARGET, ctx)
    assert not buy.approved
    assert check(buy, "drawdown_cap")["passed"] is False
    sell = eng.assess_target({**BASE_TARGET, "side": "SELL"}, ctx)
    assert sell.approved, sell.reason


def test_market_not_tradable_vetoes_all():
    eng = make_engine()
    ctx = make_ctx(tradable=False)
    for side in ("BUY", "SELL"):
        d = eng.assess_target({**BASE_TARGET, "side": side}, ctx)
        assert not d.approved
        assert check(d, "market_tradable")["passed"] is False


def test_mode_disabled_blocks_all():
    eng = make_engine()
    ctx = make_ctx(mode=ExecutionMode.DISABLED)
    d = eng.assess_target(BASE_TARGET, ctx)
    assert not d.approved
    assert check(d, "mode_enabled")["passed"] is False


def test_allow_new_positions_false_blocks_buys_only():
    eng = make_engine(allow_new_positions=False)
    buy = eng.assess_target(BASE_TARGET, make_ctx())
    assert not buy.approved
    assert check(buy, "new_positions_allowed")["passed"] is False
    sell = eng.assess_target({**BASE_TARGET, "side": "SELL"}, make_ctx())
    assert sell.approved, sell.reason


def test_clean_target_passes_with_full_audit_trail():
    d = make_engine().assess_target(BASE_TARGET, make_ctx())
    assert d.approved and d.reason == ""
    for name in ("mode_enabled", "market_tradable", "daily_loss_hard",
                 "new_positions_allowed", "drawdown_cap", "daily_loss_soft",
                 "notional_cap", "weight_cap", "liquidity_floor", "participation_cap"):
        assert check(d, name)["passed"] is True


def test_assess_accepts_portfolio_target_dataclass():
    from engine.contracts import PortfolioTarget
    t = PortfolioTarget(strategy_id="池", symbol="600000", target_weight=0.05,
                        meta={"side": "BUY", "notional": 50_000.0})
    d = make_engine().assess_target(t, make_ctx())
    assert d.approved


# ---------------------------------------------------------------------- #
# PortfolioState persistence roundtrips
# ---------------------------------------------------------------------- #
def test_state_plan_and_risk_decision_roundtrip(tmp_path):
    state = PortfolioState(tmp_path / "p.db")
    ctx = PortfolioPlanContext(decision_date="2026-09-21", equity=1_000_000.0)
    plan = PortfolioConstructor(PortfolioConfig(industry_of=None)).build(
        [], ctx)
    row_id = state.record_plan(plan)
    assert row_id >= 1
    with state._connect() as conn:
        row = conn.execute(
            "SELECT decision_date, rule_hash, cash_weight, plan_json "
            "FROM portfolio_plans WHERE id = ?", (row_id,)).fetchone()
    assert row[0] == "2026-09-21"
    assert row[2] == 1.0
    assert "cash_only" in row[3]

    decision = RiskEngine().assess_target(BASE_TARGET, make_ctx())
    dec_id = state.record_risk_decision(decision, decision_date="2026-09-21",
                                        symbol="600000", strategy_id="测试池")
    with state._connect() as conn:
        drow = conn.execute(
            "SELECT symbol, approved, decision_json FROM risk_decisions WHERE id = ?",
            (dec_id,)).fetchone()
    assert drow[0] == "600000" and drow[1] == 1
    assert '"participation_cap"' in drow[2]


def test_state_daily_pnl_roundtrip(tmp_path):
    state = PortfolioState(tmp_path / "p.db")
    assert state.get_daily_pnl("2026-09-21") is None
    state.update_daily_pnl("2026-09-21", realized=-1200.0, unrealized=300.0)
    pnl = state.get_daily_pnl("2026-09-21")
    assert pnl == {"realized": -1200.0, "unrealized": 300.0, "total": -900.0}
    # same-day update overwrites (upsert)
    state.update_daily_pnl("2026-09-21", realized=-100.0, unrealized=0.0)
    assert state.get_daily_pnl("2026-09-21")["total"] == -100.0


def test_state_high_water_and_drawdown_persist(tmp_path):
    db = tmp_path / "p.db"
    s1 = PortfolioState(db)
    assert s1.update_equity(1_000_000.0) == 0.0
    dd = s1.update_equity(900_000.0)
    assert dd == pytest.approx(0.10)
    # persisted: a fresh instance over the same db sees the high water
    s2 = PortfolioState(db)
    assert s2.high_water() == pytest.approx(1_000_000.0)
    assert s2.current_drawdown() == pytest.approx(0.10)
    assert s2.current_drawdown(800_000.0) == pytest.approx(0.20)
    # equity recovery lifts drawdown but never lowers high water
    s2.update_equity(1_100_000.0)
    assert s2.high_water() == pytest.approx(1_100_000.0)
    assert s2.current_drawdown() == 0.0
