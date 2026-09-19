"""PaperBroker 测试：tmp SQLite + 注入式行情，全部离线。"""
from __future__ import annotations

import pytest

from engine.contracts import (
    BrokerUnavailable,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from engine.execution.paper import PaperBroker

D0, D1, D2 = "20260914", "20260915", "20260916"


def make_quote(st_symbols=()):
    """symbol -> {open,high,low,close,preclose,is_st}，preclose 固定 10.0。"""
    def provider(symbol, date_str):
        return {"open": 10.0, "high": 12.9, "low": 8.1, "close": 10.0,
                "preclose": 10.0, "is_st": symbol in st_symbols}
    return provider


def make_order(symbol="600000", side=OrderSide.BUY, qty=100, price=10.0, cid="t-d-s-001"):
    return Order(client_order_id=cid, strategy_id="t", decision_id="d", symbol=symbol,
                 side=side, order_type=OrderType.LIMIT, qty=qty, limit_price=price)


@pytest.fixture()
def broker(tmp_path):
    b = PaperBroker(db_path=str(tmp_path / "paper.db"), quote_provider=make_quote(),
                    initial_cash=100_000.0)
    b.connect()
    b.set_date(D0)
    return b


class TestConnectivity:
    def test_not_connected_raises(self, tmp_path):
        b = PaperBroker(db_path=str(tmp_path / "p.db"), quote_provider=make_quote())
        with pytest.raises(BrokerUnavailable):
            b.get_balance()
        with pytest.raises(BrokerUnavailable):
            b.submit_order(make_order())

    def test_health_after_connect(self, broker):
        h = broker.health()
        assert h["ok"] is True and h["date"] == D0


class TestBuyFills:
    def test_buy_fill_with_slippage_and_fees(self, broker):
        broker.submit_order(make_order(price=10.0))
        trades = broker.get_trades(D0)
        assert len(trades) == 1
        assert trades[0].price == 10.01  # 10.0 * (1 + 10bp)
        bal = broker.get_balance()
        assert bal.available_cash == pytest.approx(100_000 - 100 * 10.01 - 5.0, abs=0.01)

    def test_lot_rounding_down(self, broker):
        broker.submit_order(make_order(qty=333, price=10.0))
        assert broker.get_trades()[0].qty == 300

    def test_limit_up_unbuyable(self, broker):
        rec = broker.query_order(broker.submit_order(make_order(price=11.0)))  # limit_up == 11.0
        assert rec.status == OrderStatus.REJECTED.value
        assert rec.raw["reject_reason"] == "limit_up_unbuyable"
        assert broker.get_trades() == []

    def test_chinext_star_20pct_band(self, broker):
        for sym in ("sz.300750", "sh.688001"):  # 创业板/科创板 20% 带：11.9 < 12.0 可买
            rec = broker.query_order(
                broker.submit_order(make_order(symbol=sym, qty=100, price=11.9)))
            assert rec.status == OrderStatus.FILLED.value, sym

    def test_st_band_5pct(self, tmp_path):
        # 红队修复后正确语义：ST 涨跌停带宽 5%（原测试错误地断言 20%）
        b = PaperBroker(db_path=str(tmp_path / "p.db"), quote_provider=make_quote(st_symbols=("sh.600100",)),
                        initial_cash=100_000.0)
        b.connect()
        b.set_date(D0)
        ok = b.query_order(b.submit_order(make_order(symbol="sh.600100", price=10.4)))
        assert ok.status == OrderStatus.FILLED.value          # ST limit_up=10.5
        bad = b.query_order(b.submit_order(make_order(symbol="sh.600100", price=10.5, cid="t-d-s-002")))
        assert bad.status == OrderStatus.REJECTED.value
        assert bad.raw["reject_reason"] == "limit_up_unbuyable"

    def test_non_marketable_limit_rejected(self, tmp_path):
        # 限价低于当日最低价的买单不可成交，保守拒绝而非按限价立即成交
        b = PaperBroker(db_path=str(tmp_path / "p.db"), quote_provider=make_quote(),
                        initial_cash=100_000.0)
        b.connect()
        b.set_date(D0)
        rec = b.query_order(b.submit_order(make_order(symbol="sh.600000", price=7.0)))
        assert rec.status == OrderStatus.REJECTED.value
        assert rec.raw["reject_reason"] == "not_marketable_below_low"

    def test_insufficient_cash_rejected(self, broker):
        rec = broker.query_order(broker.submit_order(make_order(qty=100_000, price=10.0)))
        assert rec.status == OrderStatus.REJECTED.value
        assert rec.raw["reject_reason"] == "insufficient_cash"

    def test_partial_fill_ratio(self, tmp_path):
        b = PaperBroker(db_path=str(tmp_path / "p.db"), quote_provider=make_quote(),
                        initial_cash=100_000.0, fill_ratio=0.5)
        b.connect()
        b.set_date(D0)
        rec = b.query_order(b.submit_order(make_order(qty=200, price=10.0)))
        assert rec.status == OrderStatus.PARTIALLY_FILLED.value
        assert rec.filled_qty == 100


class TestSellAndT1:
    def test_sell_same_day_blocked_by_t1(self, broker):
        broker.submit_order(make_order(qty=200, price=10.0))
        rec = broker.query_order(broker.submit_order(make_order(side=OrderSide.SELL, qty=100)))
        assert rec.status == OrderStatus.REJECTED.value
        assert rec.raw["reject_reason"] == "insufficient_position"

    def test_sell_next_day_ok_t1_release(self, broker):
        broker.submit_order(make_order(qty=200, price=10.0))          # fill 10.01, fees 5
        broker.set_date(D1)
        rec = broker.query_order(
            broker.submit_order(make_order(side=OrderSide.SELL, qty=100, price=10.0)))  # fill 9.99
        assert rec.status == OrderStatus.FILLED.value
        assert rec.filled_qty == 100
        pos = [p for p in broker.get_positions() if p.symbol == "600000"][0]
        assert pos.qty == 100 and pos.sellable_qty == 100
        # cash: 100000 - (200*10.01+5) + (100*9.99 - max(5, 0.0003*999) - 0.0005*999)
        expected = 100_000 - 2007.0 + (999.0 - 5.0 - round(0.0005 * 999, 2))
        assert broker.get_balance().available_cash == pytest.approx(expected, abs=0.02)

    def test_limit_down_unsellable(self, broker):
        broker.submit_order(make_order(qty=200, price=10.0))
        broker.set_date(D2)
        rec = broker.query_order(broker.submit_order(make_order(side=OrderSide.SELL, price=8.9)))
        assert rec.status == OrderStatus.REJECTED.value
        assert rec.raw["reject_reason"] == "limit_down_unsellable"    # limit_down == 9.0

    def test_no_quote_rejected_conservatively(self, broker):
        broker._quote_provider = lambda s, d: None
        rec = broker.query_order(broker.submit_order(make_order(price=10.0)))
        assert rec.status == OrderStatus.REJECTED.value
        assert "no quote" in rec.raw["reject_reason"]

    def test_get_orders_and_trades_filter_by_date(self, broker):
        broker.submit_order(make_order(qty=100, price=10.0))
        broker.set_date(D1)
        broker.submit_order(make_order(side=OrderSide.SELL, qty=100, price=10.0, cid="t-d-s-002"))
        assert len(broker.get_orders(D0)) == 1
        assert len(broker.get_orders(D1)) == 1
        assert len(broker.get_trades(D1)) == 1
        assert broker.get_orders(D1)[0].side == OrderSide.SELL

    def test_cancel_before_fill(self, broker):
        b = broker
        b._fill_ratio = 0.0  # engineered: no fill at all
        bid = b.submit_order(make_order(qty=200, price=10.0))
        assert b.query_order(bid).status == OrderStatus.SUBMITTED.value
        assert b.cancel_order(bid) is True
        assert b.query_order(bid).status == OrderStatus.CANCELLED.value
        assert b.cancel_order(bid) is False  # already terminal
