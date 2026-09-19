"""盘前风控检查器单元测试（block + allow 两例）。"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from engine.contracts import BrokerBalance, BrokerPosition, Order, OrderSide, OrderType
from engine.execution.risk_controls import (
    CashCheckGuard,
    DailyLossCircuitBreaker,
    DuplicateOrderGuard,
    PositionCheckGuard,
    RateLimitGuard,
    TradeContext,
    PriceDeviationGuard,
)

NOW = datetime(2026, 9, 15, 9, 35, 0)


def make_order(symbol="600000", side=OrderSide.BUY, qty=100, price=10.0, strategy="mom",
               created=NOW):
    o = Order(client_order_id=f"{strategy}-{symbol}-{side.value}-001", strategy_id=strategy,
              decision_id="d1", symbol=symbol, side=side, order_type=OrderType.LIMIT,
              qty=qty, limit_price=price, created_at=created.isoformat(timespec="seconds"))
    o.meta["reference_price"] = 10.0
    return o


def ctx(**kw):
    base = dict(balance=None, positions=[], recent_orders=[], reference_price=10.0,
                now=NOW, daily_pnl=0.0, config={"equity": 1_000_000})
    base.update(kw)
    return TradeContext(**base)


class TestDuplicateOrderGuard:
    def test_blocks_recent_same_symbol_side_strategy(self):
        prev = make_order()
        res = DuplicateOrderGuard(window_minutes=5).check(make_order(), ctx(recent_orders=[prev]))
        assert res.approved is False and "duplicate_order" in res.reason

    def test_allows_after_window(self):
        prev = make_order(created=NOW - timedelta(minutes=10))
        res = DuplicateOrderGuard(window_minutes=5).check(make_order(), ctx(recent_orders=[prev]))
        assert res.approved is True

    def test_allows_other_side(self):
        prev = make_order()
        res = DuplicateOrderGuard().check(make_order(side=OrderSide.SELL), ctx(recent_orders=[prev]))
        assert res.approved is True


class TestPriceDeviationGuard:
    def test_blocks_deviation(self):
        res = PriceDeviationGuard(max_deviation_pct=0.03).check(
            make_order(price=10.5), ctx(reference_price=10.0))
        assert res.approved is False and "price_deviation" in res.reason

    def test_allows_within_band(self):
        res = PriceDeviationGuard(max_deviation_pct=0.03).check(
            make_order(price=10.2), ctx(reference_price=10.0))
        assert res.approved is True

    def test_inert_without_reference(self):
        res = PriceDeviationGuard().check(make_order(price=99.0), ctx(reference_price=None))
        assert res.approved is True


class TestCashCheckGuard:
    def _bal(self, cash):
        return BrokerBalance(total_asset=cash, available_cash=cash, frozen_cash=0.0,
                             market_value=0.0, asof="20260915")

    def test_blocks_buy_beyond_cash(self):
        res = CashCheckGuard().check(make_order(qty=1000, price=10.0), ctx(balance=self._bal(5000)))
        # need = 10000 + max(3, 5) = 10005 > 5000
        assert res.approved is False and "insufficient_cash" in res.reason

    def test_allows_buy_within_cash(self):
        res = CashCheckGuard().check(make_order(qty=100, price=10.0), ctx(balance=self._bal(5000)))
        # need = 1000 + 5 <= 5000
        assert res.approved is True

    def test_sell_never_cash_blocked(self):
        res = CashCheckGuard().check(make_order(side=OrderSide.SELL, qty=10**9),
                                     ctx(balance=self._bal(0)))
        assert res.approved is True


class TestPositionCheckGuard:
    def _pos(self, symbol="600000", sellable=100):
        return BrokerPosition(symbol=symbol, qty=sellable, sellable_qty=sellable,
                              avg_cost=10.0, market_value=sellable * 10, asof="20260915")

    def test_blocks_sell_without_position(self):
        res = PositionCheckGuard().check(make_order(side=OrderSide.SELL, qty=100), ctx())
        assert res.approved is False and "insufficient_position" in res.reason

    def test_blocks_sell_beyond_sellable_t1(self):
        res = PositionCheckGuard().check(make_order(side=OrderSide.SELL, qty=200),
                                         ctx(positions=[self._pos(sellable=100)]))
        assert res.approved is False

    def test_allows_sell_within_sellable(self):
        res = PositionCheckGuard().check(make_order(side=OrderSide.SELL, qty=100),
                                         ctx(positions=[self._pos(sellable=100)]))
        assert res.approved is True


class TestRateLimitGuard:
    def test_blocks_daily_cap(self):
        many = [make_order(created=NOW - timedelta(hours=1), strategy=f"s{i}")
                for i in range(200)]
        res = RateLimitGuard(max_per_day=200).check(make_order(), ctx(recent_orders=many))
        assert res.approved is False and "rate_limit" in res.reason

    def test_blocks_min_interval(self):
        prev = make_order(created=NOW - timedelta(seconds=1))
        res = RateLimitGuard(min_interval_seconds=2).check(make_order(), ctx(recent_orders=[prev]))
        assert res.approved is False and "last order" in res.reason

    def test_allows_spaced_orders(self):
        prev = make_order(created=NOW - timedelta(seconds=30))
        res = RateLimitGuard(max_per_day=200, min_interval_seconds=2).check(
            make_order(), ctx(recent_orders=[prev]))
        assert res.approved is True


class TestDailyLossCircuitBreaker:
    def test_hard_stop_blocks_all_and_flags(self):
        c = {"equity": 1_000_000}
        res = DailyLossCircuitBreaker().check(make_order(side=OrderSide.SELL), ctx(daily_pnl=-60_000, config=c))
        assert res.approved is False and "hard_stop" in res.reason
        assert c["account_stop"] is True

    def test_soft_stop_blocks_new_positions_only(self):
        br = DailyLossCircuitBreaker(soft_stop=-0.02, hard_stop=-0.05)
        buy = br.check(make_order(side=OrderSide.BUY), ctx(daily_pnl=-25_000))
        sell = br.check(make_order(side=OrderSide.SELL), ctx(daily_pnl=-25_000))
        assert buy.approved is False and "soft_stop" in buy.reason
        assert sell.approved is True  # 减仓允许

    def test_allows_normal_pnl(self):
        res = DailyLossCircuitBreaker().check(make_order(), ctx(daily_pnl=5_000))
        assert res.approved is True

    def test_inert_without_equity(self):
        res = DailyLossCircuitBreaker().check(make_order(), ctx(daily_pnl=-999_999, config={}))
        assert res.approved is True
