"""同花顺 easytrader 适配器测试：假 client 注入；离线。"""
from __future__ import annotations

import pytest

from engine.contracts import (
    BrokerOrderRecord,
    BrokerUnavailable,
    CaptchaRequired,
    Order,
    OrderSide,
    OrderType,
)
from engine.execution.ths_easytrader import TongHuaShunAdapter, map_row, side_of


class FakeTHSClient:
    """形似 easytrader ths universal client 的假对象（中文列）。"""

    def __init__(self, verify_submit=True, raise_on_buy=None):
        self.entrusts = []
        self.trades = []
        self.buy_calls = []
        self.verify_submit = verify_submit
        self.raise_on_buy = raise_on_buy
        self._next = 20260915001

    def connect(self):
        self.connected = True
        return True

    def balance(self):
        return [{"资金余额": 100000.0, "可用余额": 90000.0, "冻结金额": 1000.0,
                 "证券市值": 50000.0, "总资产": 150000.0}]

    def position(self):
        return [{"证券代码": "600000", "证券数量": 1000, "可用余额": 800,
                 "成本价": 10.0, "市值": 10500.0}]

    def buy(self, symbol, price, qty):
        self.buy_calls.append((symbol, price, qty))
        if self.raise_on_buy:
            raise self.raise_on_buy
        if self.verify_submit:
            no = str(self._next)
            self._next += 1
            self.entrusts.append({"合同编号": no, "证券代码": symbol, "操作": "买入",
                                  "委托数量": qty, "成交数量": qty, "委托价格": price,
                                  "状态": "全部成交", "委托时间": "20260915 09:31:00"})
            return no
        return "0"  # click 'succeeded' but nothing visible

    def sell(self, symbol, price, qty):
        no = str(self._next)
        self._next += 1
        self.entrusts.append({"合同编号": no, "证券代码": symbol, "操作": "卖出",
                              "委托数量": qty, "成交数量": qty, "委托价格": price,
                              "状态": "全部成交", "委托时间": "20260915 09:31:00"})
        return no

    def today_entrusts(self):
        return list(self.entrusts)

    def today_trades(self):
        return [{"成交编号": "T1", "合同编号": self.entrusts[0]["合同编号"],
                 "证券代码": self.entrusts[0]["证券代码"], "操作": "买入",
                 "成交数量": 100, "成交价格": 10.01, "成交时间": "20260915 09:31:01"}] \
            if self.entrusts else []

    def cancel_entrust(self, entrust_no):
        self.entrusts = [e for e in self.entrusts if e["合同编号"] != entrust_no]
        return True


def make_order(side=OrderSide.BUY, qty=100, price=10.01):
    return Order(client_order_id="mom-d1-600000-BUY-001", strategy_id="mom", decision_id="d1",
                 symbol="600000", side=side, order_type=OrderType.LIMIT, qty=qty,
                 limit_price=price)


def adapter_with(client, timeout=0.5):
    a = TongHuaShunAdapter(client_factory=lambda: client, ack_timeout_s=timeout)
    a.connect()
    return a


class TestHonestAvailability:
    def test_easytrader_not_installed_reports_honestly(self):
        a = TongHuaShunAdapter()  # easytrader 不在环境里
        a.connect()
        h = a.health()
        assert h["ok"] is False
        assert "easytrader not installed" in h["reason"]
        for call in (a.get_balance, a.get_positions, a.get_orders, a.get_trades,
                     lambda: a.submit_order(make_order()),
                     lambda: a.cancel_order("1"), lambda: a.query_order("1")):
            with pytest.raises(BrokerUnavailable):
                call()


class TestSubmitVerification:
    def test_buy_verified_via_today_entrusts(self):
        a = adapter_with(FakeTHSClient())
        bid = a.submit_order(make_order())
        assert bid.isdigit() and len(bid) >= 3
        rec = a.query_order(bid)
        assert rec is not None
        assert isinstance(rec, BrokerOrderRecord)
        assert rec.status == "FILLED" and rec.filled_qty == 100

    def test_click_success_without_entrust_raises_unavailable(self):
        a = adapter_with(FakeTHSClient(verify_submit=False), timeout=0.15)
        with pytest.raises(BrokerUnavailable) as ei:
            a.submit_order(make_order())
        assert "UNKNOWN" in str(ei.value)

    def test_captcha_message_maps_to_captcha_required(self):
        a = adapter_with(FakeTHSClient(raise_on_buy=RuntimeError("请输入验证码")))
        with pytest.raises(CaptchaRequired):
            a.submit_order(make_order())

    def test_only_limit_orders_accepted(self):
        a = adapter_with(FakeTHSClient())
        o = make_order()
        o.order_type = OrderType.MARKET
        with pytest.raises(ValueError):
            a.submit_order(o)

    def test_price_and_qty_normalized(self):
        client = FakeTHSClient()
        a = adapter_with(client)
        a.submit_order(make_order(price=10.006, qty=100.0))
        assert client.buy_calls[0][1] == 10.01  # round 2dp
        assert client.buy_calls[0][2] == 100    # int


class TestMappings:
    def test_balance_from_chinese_columns(self):
        a = adapter_with(FakeTHSClient())
        bal = a.get_balance()
        assert bal.available_cash == 90000.0
        assert bal.total_asset == 150000.0
        assert bal.frozen_cash == 1000.0
        assert bal.market_value == 50000.0

    def test_positions_map_sellable(self):
        a = adapter_with(FakeTHSClient())
        pos = a.get_positions()
        assert len(pos) == 1
        assert pos[0].symbol == "600000"
        assert pos[0].qty == 1000 and pos[0].sellable_qty == 800  # 可用余额 → T+1 sellable

    def test_get_orders_and_trades(self):
        a = adapter_with(FakeTHSClient())
        bid = a.submit_order(make_order())
        orders = a.get_orders()
        assert orders[0].broker_order_id == bid
        assert orders[0].side == OrderSide.BUY
        trades = a.get_trades()
        assert trades[0].broker_order_id == bid
        assert trades[0].qty == 100 and trades[0].price == 10.01

    def test_sell_side_and_cancel(self):
        a = adapter_with(FakeTHSClient())
        bid = a.submit_order(make_order(side=OrderSide.SELL))
        assert a.get_orders()[0].side == OrderSide.SELL
        assert a.cancel_order(bid) is True
        assert a.query_order(bid) is None

    def test_column_map_and_side_helper(self):
        row = map_row({"证券代码": "600000", "可用余额": 800, "证券数量": 1000})
        assert row == {"symbol": "600000", "available": 800, "qty": 1000}
        assert side_of("买入") is OrderSide.BUY
        assert side_of("卖出") is OrderSide.SELL
        assert side_of(" ?? ") is None
