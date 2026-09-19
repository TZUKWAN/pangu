"""BacktestV2 引擎性质测试：账户恒等式、手数、T+1、涨跌停/停牌/容量事件、确定性。"""

from __future__ import annotations

import json

import pytest

from engine.validation.backtest_v2 import (
    BacktestConfig,
    BacktestV2,
    LookaheadError,
    TargetOrder,
)
from engine.validation.data_interface import SyntheticValidationData

LOT = 100


class PlanStrategy:
    """plan: {decision_date: [TargetOrder, ...]}"""

    def __init__(self, plan):
        self.plan = plan

    def rebalance(self, decision_date, view):
        return list(self.plan.get(str(decision_date), []))


@pytest.fixture
def data():
    return SyntheticValidationData(n_symbols=8, n_days=10, seed=42)


def _buy_all(data, weight=0.05):
    return [TargetOrder(symbol=s, side="BUY", weight=weight) for s in data.codes]


def test_equity_never_negative_and_identity(data):
    plan = {data.dates[0]: _buy_all(data, 0.10)}
    res = BacktestV2(data, BacktestConfig()).run(
        PlanStrategy(plan), data.dates[0], data.dates[-1])
    eq = res.equity_curve
    assert (eq["equity"] >= 0).all(), "equity must never go below 0"
    assert (eq["cash"] >= -1e-6).all()
    identity = (eq["cash"] + eq["market_value"] - eq["equity"]).abs()
    assert (identity < 1e-6).all(), "cash + market_value == equity on every row"


def test_buy_qty_lot_multiple(data):
    plan = {data.dates[0]: _buy_all(data, 0.05),
            data.dates[4]: [TargetOrder(symbol=s, side="SELL", weight=0.0)
                            for s in data.codes]}
    res = BacktestV2(data, BacktestConfig()).run(
        PlanStrategy(plan), data.dates[0], data.dates[-1])
    assert res.trades
    for tr in res.trades:
        assert tr["qty"] > 0 and tr["qty"] % LOT == 0


def test_t1_same_day_buy_not_sellable(data):
    """d1 决策同票 BUY+SELL：d2 买入当日不可卖 → unsold_t1；d3 卖出成交。"""
    sym = data.codes[0]
    plan = {
        data.dates[0]: [TargetOrder(symbol=sym, side="BUY", weight=0.10),
                        TargetOrder(symbol=sym, side="SELL", weight=0.0)],
        data.dates[1]: [TargetOrder(symbol=sym, side="SELL", weight=0.0)],
    }
    res = BacktestV2(data, BacktestConfig()).run(
        PlanStrategy(plan), data.dates[0], data.dates[-1])
    unsold = [e for e in res.events if e["type"] == "unsold_t1"]
    assert unsold and unsold[0]["date"] == data.dates[1]
    assert unsold[0]["detail"]["qty"] > 0
    sells = [tr for tr in res.trades if tr["close_reason"] == "sell_target"]
    assert sells and all(tr["exit_date"] == data.dates[2] for tr in sells)
    assert all(tr["entry_date"] == data.dates[1] for tr in sells)


def test_limit_up_buy_never_fills(data):
    sym = data.codes[0]
    data.limit_up_open(sym, data.dates[1])
    plan = {data.dates[0]: [TargetOrder(symbol=sym, side="BUY", weight=0.10)]}
    res = BacktestV2(data, BacktestConfig()).run(
        PlanStrategy(plan), data.dates[0], data.dates[-1])
    ev = [e for e in res.events if e["type"] == "unfilled_limit_up"]
    assert ev and ev[0]["symbol"] == sym
    assert not [t for t in res.trades if t["symbol"] == sym
                and t["entry_date"] == data.dates[1]]


def test_limit_up_one_word_board_even_with_allow(data):
    sym = data.codes[0]
    data.limit_up_open(sym, data.dates[1])
    plan = {data.dates[0]: [TargetOrder(symbol=sym, side="BUY", weight=0.10)]}
    cfg = BacktestConfig(allow_buy_at_limit_up=True)
    res = BacktestV2(data, cfg).run(PlanStrategy(plan),
                                    data.dates[0], data.dates[-1])
    assert any(e["type"] == "unfilled_limit_up" for e in res.events)


def test_suspended_no_trade(data):
    sym = data.codes[0]
    data.suspend(sym, [data.dates[1]])
    plan = {data.dates[0]: [TargetOrder(symbol=sym, side="BUY", weight=0.10)]}
    res = BacktestV2(data, BacktestConfig()).run(
        PlanStrategy(plan), data.dates[0], data.dates[-1])
    ev = [e for e in res.events if e["type"] == "unfilled_suspended"]
    assert ev and ev[0]["date"] == data.dates[1]
    assert not res.trades


def test_capacity_cut(data):
    sym = data.codes[0]
    cfg = BacktestConfig(participation_cap=1e-12)
    plan = {data.dates[0]: [TargetOrder(symbol=sym, side="BUY", weight=0.10)]}
    res = BacktestV2(data, cfg).run(PlanStrategy(plan),
                                    data.dates[0], data.dates[-1])
    ev = [e for e in res.events if e["type"] == "unfilled_capacity"]
    assert ev, "capacity excess must be recorded, never silently dropped"
    assert not [t for t in res.trades if t["entry_date"] == data.dates[1]]


def test_max_positions_drop_recorded(data):
    cfg = BacktestConfig(max_positions=3)
    plan = {data.dates[0]: _buy_all(data, 0.05)}
    res = BacktestV2(data, cfg).run(PlanStrategy(plan),
                                    data.dates[0], data.dates[-1])
    drops = [e for e in res.events if e["type"] == "dropped_max_positions"]
    assert drops and len(drops[0]["detail"]["symbols"]) == 5
    held_syms = {t["symbol"] for t in res.trades}
    assert len(held_syms) == 3


def test_determinism_byte_identical(data):
    plan = {data.dates[0]: _buy_all(data, 0.05),
            data.dates[3]: [TargetOrder(symbol=data.codes[1], side="SELL",
                                        weight=0.0)]}
    cfg = BacktestConfig()
    r1 = BacktestV2(data, cfg).run(PlanStrategy(plan),
                                   data.dates[0], data.dates[-1]).to_dict()
    r2 = BacktestV2(data, cfg).run(PlanStrategy(plan),
                                   data.dates[0], data.dates[-1]).to_dict()
    b1 = json.dumps(r1, sort_keys=True, ensure_ascii=False).encode("utf-8")
    b2 = json.dumps(r2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    assert b1 == b2


def test_history_view_blocks_future(data):
    naughty_sym = data.codes[0]

    class Naughty:
        def __init__(self):
            self.seen = []

        def rebalance(self, t, view):
            assert view.asof == t
            nxt = data.dates[data.dates.index(t) + 1]
            self.seen.append(t)
            # 任何未来访问都必须抛 LookaheadError
            with pytest.raises(LookaheadError):
                view.daily_frame(nxt)
            with pytest.raises(LookaheadError):
                view.panel_asof(nxt)
            with pytest.raises(LookaheadError):
                view.trading_days(t, nxt)
            with pytest.raises(LookaheadError):
                view.universe(nxt)
            with pytest.raises(LookaheadError):
                view.asof_view(nxt)
            # asof 当日访问正常
            assert len(view.daily_frame(t)) >= 0
            return [TargetOrder(symbol=naughty_sym, side="BUY", weight=0.05)]

    s = Naughty()
    res = BacktestV2(data, BacktestConfig()).run(s, data.dates[0], data.dates[-1])
    assert len(s.seen) == len(data.dates) - 1
    assert res.trades


def test_forced_close_at_end(data):
    """无 SELL 目标 → 期末按末日收盘强平，close_reason=force_close，空仓收尾。"""
    plan = {data.dates[0]: [TargetOrder(symbol=data.codes[2], side="BUY",
                                        weight=0.10)]}
    res = BacktestV2(data, BacktestConfig()).run(
        PlanStrategy(plan), data.dates[0], data.dates[-1])
    assert res.trades and all(
        tr["close_reason"] == "force_close" for tr in res.trades)
    assert res.trades[0]["exit_date"] == data.dates[-1]
    assert float(res.equity_curve.iloc[-1]["market_value"]) == 0.0
    assert int(res.equity_curve.iloc[-1]["n_positions"]) == 0


def test_ex_div_pnl_uses_true_return(data):
    """除权日：pct_change 保持真实收益，引擎按 bar 价格成交不受影响。"""
    sym = data.codes[1]
    data.ex_div(sym, data.dates[2], pct=0.10)
    plan = {data.dates[0]: [TargetOrder(symbol=sym, side="BUY", weight=0.10)],
            data.dates[4]: [TargetOrder(symbol=sym, side="SELL", weight=0.0)]}
    res = BacktestV2(data, BacktestConfig()).run(
        PlanStrategy(plan), data.dates[0], data.dates[-1])
    assert res.trades
    # 权益恒等式仍逐行成立
    eq = res.equity_curve
    identity = (eq["cash"] + eq["market_value"] - eq["equity"]).abs()
    assert (identity < 1e-6).all()
