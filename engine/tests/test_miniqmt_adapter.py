"""miniQMT (xtquant) 适配器测试：假 XtQuantTrader 注入；离线。"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from engine.contracts import (
    BrokerUnavailable,
    CaptchaRequired,
    Order,
    OrderSide,
    OrderType,
)
from engine.execution.miniqmt import MiniQMTAdapter, STATUS_MAP


@dataclass
class FakeXtOrder:
    order_id: int
    stock_code: str
    order_type: int
    order_volume: int
    traded_volume: int
    price: float
    order_status: int
    order_time: str = "20260915 09:31:00"
    strategy_name: str = "mom"
    order_remark: str = "d1"


@dataclass
class FakeXtTrade:
    traded_id: str
    order_id: int
    stock_code: str
    order_type: int
    traded_volume: int
    traded_price: float
    traded_time: str = "20260915 09:31:01"


@dataclass
class FakeXtAsset:
    total_asset: float = 150000.0
    cash: float = 90000.0
    frozen_cash: float = 1000.0
    market_value: float = 50000.0


@dataclass
class FakeXtPosition:
    stock_code: str
    volume: int
    can_use_volume: int
    open_price: float
    market_value: float


class FakeXtQuantTrader:
    STOCK_BUY, STOCK_SELL = 23, 24

    def __init__(self, fail_connect=False):
        self.fail_connect = fail_connect
        self.subscribed = []
        self.orders: list[FakeXtOrder] = []
        self.trades: list[FakeXtTrade] = []
        self.cancelled = []
        self._next = 3001

    def connect(self):
        return 1 if self.fail_connect else 0

    def subscribe(self, account_id):
        self.subscribed.append(account_id)
        return 0

    def order_stock(self, account_id, code, order_type, volume, price_type, price,
                    strategy_name="", order_remark=""):
        if code == "000000":
            return -1  # 模拟柜台拒绝
        oid = self._next
        self._next += 1
        status = 51 if oid % 2 else 50  # FILLED / PART_SUCC
        self.orders.append(FakeXtOrder(order_id=oid, stock_code=code, order_type=order_type,
                                       order_volume=volume, traded_volume=volume, price=price,
                                       order_status=status))
        self.trades.append(FakeXtTrade(traded_id=f"X{oid}", order_id=oid, stock_code=code,
                                       order_type=order_type, traded_volume=volume,
                                       traded_price=price))
        return oid

    def cancel_order_stock(self, account_id, order_id):
        self.cancelled.append(order_id)
        self.orders = [o for o in self.orders if o.order_id != order_id]
        return 0

    def query_stock_orders(self, account_id):
        return list(self.orders)

    def query_stock_trades(self, account_id):
        return list(self.trades)

    def query_stock_asset(self, account_id):
        return FakeXtAsset()

    def query_stock_positions(self, account_id):
        return [FakeXtPosition("600000", 1000, 800, 10.0, 10500.0)]


def make_order(symbol="600000", side=OrderSide.BUY, qty=100, price=10.01):
    return Order(client_order_id="mom-d1-600000-BUY-001", strategy_id="mom", decision_id="d1",
                 symbol=symbol, side=side, order_type=OrderType.LIMIT, qty=qty,
                 limit_price=price)


def adapter_with(client, account="888000011"):
    a = MiniQMTAdapter(client_factory=lambda: client, account_id=account)
    a.connect()
    return a


class TestHonestAvailability:
    def test_xtquant_not_installed_reports_honestly(self):
        a = MiniQMTAdapter()  # xtquant 不在环境里
        a.connect()
        h = a.health()
        assert h["ok"] is False
        assert "xtquant not installed" in h["reason"]

    def test_qmt_terminal_down_reports_honestly(self):
        a = adapter_with(FakeXtQuantTrader(fail_connect=True))
        assert a.health()["ok"] is False
        with pytest.raises(BrokerUnavailable):
            a.get_balance()


class TestFlows:
    def test_connect_subscribes_account(self):
        client = FakeXtQuantTrader()
        a = adapter_with(client, account="ACC1")
        assert a.health()["ok"] is True
        assert client.subscribed == ["ACC1"]

    def test_submit_returns_broker_id_and_query_maps(self):
        a = adapter_with(FakeXtQuantTrader())
        bid = a.submit_order(make_order())
        rec = a.query_order(bid)
        assert rec is not None and rec.broker_order_id == bid
        assert rec.symbol == "600000" and rec.side == OrderSide.BUY
        assert rec.qty == 100 and rec.filled_qty == 100
        assert rec.status == STATUS_MAP[51] == "FILLED"

    def test_partial_fill_maps(self):
        client = FakeXtQuantTrader()
        a = adapter_with(client)
        first = a.submit_order(make_order())
        second = a.submit_order(make_order(qty=200, symbol="600001"))
        # _next: 3001(奇)→FILLED, 3002(偶)→PART_SUCC
        assert a.query_order(first).status == "FILLED"
        assert a.query_order(second).status == "PARTIALLY_FILLED"

    def test_broker_reject_returns_unavailable(self):
        a = adapter_with(FakeXtQuantTrader())
        with pytest.raises(BrokerUnavailable):
            a.submit_order(make_order(symbol="000000"))

    def test_balance_and_positions_mapping(self):
        a = adapter_with(FakeXtQuantTrader())
        bal = a.get_balance()
        assert (bal.total_asset, bal.available_cash, bal.frozen_cash, bal.market_value) == \
            (150000.0, 90000.0, 1000.0, 50000.0)
        pos = a.get_positions()
        assert pos[0].symbol == "600000" and pos[0].sellable_qty == 800

    def test_trades_and_cancel(self):
        a = adapter_with(FakeXtQuantTrader())
        bid = a.submit_order(make_order())
        trades = a.get_trades()
        assert trades[0].broker_order_id == bid and trades[0].price == pytest.approx(10.01)
        assert a.cancel_order(bid) is True
        assert a.query_order(bid) is None

    def test_captcha_keyword_maps(self):
        class CaptchaClient(FakeXtQuantTrader):
            def order_stock(self, *a, **k):
                raise RuntimeError("客户端未登录，请输入密码")

        adapter = adapter_with(CaptchaClient())
        with pytest.raises(CaptchaRequired):
            adapter.submit_order(make_order())

    def test_only_limit_orders(self):
        a = adapter_with(FakeXtQuantTrader())
        o = make_order()
        o.order_type = OrderType.MARKET
        with pytest.raises(ValueError):
            a.submit_order(o)
