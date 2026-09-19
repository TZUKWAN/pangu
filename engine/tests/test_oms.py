"""OMS 流程测试：staging 幂等、风控拒绝、UNKNOWN→reconcile、kill switch。"""
from __future__ import annotations

import sqlite3

import pytest

from engine.contracts import (
    BrokerBalance,
    BrokerOrderRecord,
    CaptchaRequired,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from engine.execution.broker import BrokerAdapter
from engine.execution.oms import NeedsReconciliation, OrderManagementSystem
from engine.execution.risk_controls import (
    CheckResult,
    DuplicateOrderGuard,
    PreTradeChecker,
    TradeContext,
)


class BlockingChecker(PreTradeChecker):
    def __init__(self, reason="nope"):
        self.reason = reason

    def check(self, order: Order, ctx: TradeContext) -> CheckResult:
        return CheckResult(False, self.reason)


class FakeAdapter(BrokerAdapter):
    name = "fake"

    def __init__(self, fail_submit_once=None, visible_after_fail=False,
                 captcha_submit=False, cancel_ok=True):
        super().__init__()
        self.fail_submit_once = fail_submit_once
        self.visible_after_fail = visible_after_fail
        self.captcha_submit = captcha_submit
        self.cancel_ok = cancel_ok
        self.cancel_calls = []
        self.orders = {}
        self.next_id = 1000

    def connect(self):
        self._connected = True

    def get_balance(self):
        return BrokerBalance(total_asset=1e6, available_cash=1e6, frozen_cash=0.0,
                             market_value=0.0, asof="20260915")

    def get_positions(self):
        return []

    def _record(self, order):
        bid = f"FAKE-{self.next_id}"
        self.next_id += 1
        self.orders[bid] = BrokerOrderRecord(
            broker_order_id=bid, symbol=order.symbol, side=order.side, qty=order.qty,
            filled_qty=order.qty, price=order.limit_price, status="FILLED",
            submitted_at=order.created_at)
        return bid

    def submit_order(self, order):
        self._require_connected()
        if self.captcha_submit:
            raise CaptchaRequired("请输入验证码")
        if self.fail_submit_once is not None:
            exc, self.fail_submit_once = self.fail_submit_once, None
            if self.visible_after_fail:  # 单子其实到了交易所，只是响应超时
                self._record(order)
            raise exc
        return self._record(order)

    def cancel_order(self, broker_order_id):
        self._require_connected()
        self.cancel_calls.append(broker_order_id)
        if broker_order_id in self.orders and self.cancel_ok:
            self.orders[broker_order_id].status = "CANCELLED"
        return self.cancel_ok

    def query_order(self, broker_order_id):
        self._require_connected()
        return self.orders.get(broker_order_id)

    def get_orders(self, trade_date=None):
        return list(self.orders.values())

    def get_trades(self, trade_date=None):
        return []


@pytest.fixture()
def env(tmp_path):
    adapter = FakeAdapter()
    adapter.connect()
    oms = OrderManagementSystem(adapter, db_path=str(tmp_path / "exec.db"),
                                ack_timeout_s=1.0)
    return adapter, oms


def stage(oms, **kw):
    args = dict(strategy_id="mom", decision_id="d1", symbol="600000",
                side=OrderSide.BUY.value, qty=100, limit_price=10.0)
    args.update(kw)
    return oms.stage_order(**args)


class TestStaging:
    def test_stage_creates_created_order(self, env):
        _, oms = env
        o = stage(oms)
        assert o.status == OrderStatus.CREATED
        assert o.client_order_id == "mom-d1-600000-BUY-001"
        assert o.order_type == OrderType.LIMIT

    def test_restage_same_key_is_idempotent(self, env):
        _, oms = env
        a = stage(oms)
        b = stage(oms)  # same strategy/decision/symbol/side/seq
        assert a.client_order_id == b.client_order_id
        assert a.to_dict() == b.to_dict()
        assert len(oms.list_orders()) == 1

    def test_restage_after_submit_returns_unchanged(self, env):
        adapter, oms = env
        o = stage(oms)
        submitted = oms.submit(o.client_order_id)
        again = stage(oms)  # same key → existing persisted order, no new order, no reset
        assert again.client_order_id == o.client_order_id
        assert again.to_dict() == oms.get_order(o.client_order_id).to_dict()
        assert again.status == OrderStatus.ACKNOWLEDGED
        assert again.broker_order_id == submitted.broker_order_id == "FAKE-1000"
        assert len(oms.list_orders()) == 1


class TestSubmitAck:
    def test_stage_submit_ack_flow(self, env):
        _, oms = env
        o = stage(oms)
        out = oms.submit(o.client_order_id)
        assert out.status == OrderStatus.ACKNOWLEDGED
        assert out.broker_order_id == "FAKE-1000"
        statuses = [e.status for e in out.timeline]
        assert statuses == ["CREATED", "PENDING_SUBMIT", "SUBMITTED", "ACKNOWLEDGED"]

    def test_risk_rejected_when_check_fails(self, tmp_path):
        adapter = FakeAdapter()
        adapter.connect()
        oms = OrderManagementSystem(adapter, checks=[BlockingChecker("deviation too big")],
                                    db_path=str(tmp_path / "e.db"))
        o = stage(oms)
        out = oms.submit(o.client_order_id)
        assert out.status == OrderStatus.RISK_REJECTED
        assert "deviation too big" in out.reject_reason
        with pytest.raises(NeedsReconciliation):
            oms.submit(out.client_order_id)  # non-CREATED cannot resubmit

    def test_duplicate_guard_blocks_second_order(self, tmp_path):
        adapter = FakeAdapter()
        adapter.connect()
        oms = OrderManagementSystem(adapter, checks=[DuplicateOrderGuard(window_minutes=5)],
                                    db_path=str(tmp_path / "e.db"))
        first = oms.submit(stage(oms).client_order_id)
        assert first.status == OrderStatus.ACKNOWLEDGED
        second = oms.submit(stage(oms, decision_id="d2").client_order_id)
        assert second.status == OrderStatus.RISK_REJECTED
        assert "duplicate_order" in second.reject_reason

    def test_submit_unknown_cid_raises(self, env):
        _, oms = env
        with pytest.raises(KeyError):
            oms.submit("ghost-000")


class TestUnknownAndReconcile:
    def test_submit_timeout_goes_unknown_and_blocks(self, tmp_path):
        adapter = FakeAdapter(fail_submit_once=TimeoutError("read timed out"))
        adapter.connect()
        oms = OrderManagementSystem(adapter, db_path=str(tmp_path / "e.db"))
        o = oms.submit(stage(oms).client_order_id)
        assert o.status == OrderStatus.UNKNOWN
        assert oms.trading_allowed() is False
        with pytest.raises(NeedsReconciliation):
            oms.submit(o.client_order_id)  # resubmit forbidden before reconcile

    def test_reconcile_finds_lost_order_by_key(self, tmp_path):
        adapter = FakeAdapter(fail_submit_once=TimeoutError("read timed out"),
                              visible_after_fail=True)
        adapter.connect()
        oms = OrderManagementSystem(adapter, db_path=str(tmp_path / "e.db"))
        o = oms.submit(stage(oms).client_order_id)
        assert o.status == OrderStatus.UNKNOWN
        report = oms.reconcile()
        assert report.matched == 1 and report.reconcile_ok is True
        assert report.still_unknown == []
        resolved = oms.get_order(o.client_order_id)
        assert resolved.status == OrderStatus.FILLED       # broker shows FILLED
        assert resolved.broker_order_id == "FAKE-1000"
        assert resolved.avg_price == pytest.approx(10.0)
        assert oms.trading_allowed() is True

    def test_reconcile_still_unknown_blocks_trading(self, tmp_path):
        adapter = FakeAdapter(fail_submit_once=TimeoutError("read timed out"),
                              visible_after_fail=False)
        adapter.connect()
        oms = OrderManagementSystem(adapter, db_path=str(tmp_path / "e.db"))
        o = oms.submit(stage(oms).client_order_id)
        assert o.status == OrderStatus.UNKNOWN
        report = oms.reconcile()
        assert report.matched == 0 and report.still_unknown == [o.client_order_id]
        assert report.reconcile_ok is False
        assert oms.trading_allowed() is False
        assert oms.get_order(o.client_order_id).status == OrderStatus.UNKNOWN

    def test_reconcile_applies_fills_to_submitted_order(self, env):
        _, oms = env
        o = oms.submit(stage(oms).client_order_id)
        assert o.status == OrderStatus.ACKNOWLEDGED
        report = oms.reconcile()
        assert report.matched == 1
        assert oms.get_order(o.client_order_id).status == OrderStatus.FILLED

    def test_captcha_goes_unknown_manual_intervention(self, tmp_path):
        adapter = FakeAdapter(captcha_submit=True)
        adapter.connect()
        oms = OrderManagementSystem(adapter, db_path=str(tmp_path / "e.db"))
        o = oms.submit(stage(oms).client_order_id)
        assert o.status == OrderStatus.UNKNOWN
        assert oms.trading_allowed() is False
        row = sqlite3.connect(oms._db_path).execute(
            "SELECT value FROM system_state WHERE key='manual_intervention_required'").fetchone()
        assert "验证码" in row[0]
        oms.clear_manual_intervention("op", "人工已处理验证码")
        assert oms.trading_allowed() is True


class TestCancel:
    def test_cancel_created_is_local(self, env):
        _, oms = env
        o = stage(oms)
        out = oms.cancel(o.client_order_id)
        assert out.status == OrderStatus.CANCELLED

    def test_cancel_acknowledged_via_broker(self, env):
        adapter, oms = env
        o = oms.submit(stage(oms).client_order_id)
        out = oms.cancel(o.client_order_id)
        assert out.status == OrderStatus.CANCELLED
        assert adapter.cancel_calls == [o.broker_order_id]
        statuses = [e.status for e in out.timeline]
        assert statuses[-2:] == ["CANCEL_PENDING", "CANCELLED"]

    def test_cancel_unknown_requires_reconcile(self, tmp_path):
        adapter = FakeAdapter(fail_submit_once=TimeoutError("t"))
        adapter.connect()
        oms = OrderManagementSystem(adapter, db_path=str(tmp_path / "e.db"))
        o = oms.submit(stage(oms).client_order_id)
        with pytest.raises(NeedsReconciliation):
            oms.cancel(o.client_order_id)

    def test_cancel_broker_failure_goes_unknown(self, env):
        adapter, oms = env
        adapter.cancel_ok = False
        o = oms.submit(stage(oms).client_order_id)
        out = oms.cancel(o.client_order_id)
        assert out.status == OrderStatus.UNKNOWN


class TestKillSwitch:
    def test_engage_cancels_and_blocks_then_disengage(self, env):
        adapter, oms = env
        o = oms.submit(stage(oms).client_order_id)
        assert oms.trading_allowed() is True
        oms.engage_kill_switch("盘中异常波动", operator="human")
        assert oms.get_order(o.client_order_id).status == OrderStatus.CANCELLED
        assert adapter.cancel_calls == [o.broker_order_id]
        assert oms.trading_allowed() is False
        blocked = oms.submit(stage(oms, decision_id="d3").client_order_id)
        assert blocked.status == OrderStatus.RISK_REJECTED  # fail closed
        with pytest.raises(ValueError):
            oms.disengage_kill_switch("human", "")
        oms.disengage_kill_switch("human", "波动恢复，复盘完成")
        assert oms.trading_allowed() is True

    def test_engage_local_cancel_for_unsubmitted(self, env):
        adapter, oms = env
        o = stage(oms)
        oms.engage_kill_switch("收盘", operator="op")
        assert oms.get_order(o.client_order_id).status == OrderStatus.CANCELLED
        assert adapter.cancel_calls == []

    def test_audit_trail_persisted(self, env):
        _, oms = env
        o = oms.submit(stage(oms).client_order_id)
        oms.engage_kill_switch("test", operator="op")
        rows = sqlite3.connect(oms._db_path).execute(
            "SELECT kind, status FROM order_events ORDER BY id").fetchall()
        kinds = {r[0] for r in rows}
        assert {"order", "kill_switch"} <= kinds
        assert ("kill_switch", "ENGAGED") in rows
