"""contracts.py 订单状态机测试（离线，无 IO）。"""
from __future__ import annotations

import pytest

from engine.contracts import (
    ORDER_TERMINAL_STATES,
    InvalidOrderTransition,
    assert_transition,
    new_client_order_id,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)


class TestAssertTransition:
    def test_legal_created_to_pending_submit(self):
        assert_transition(OrderStatus.CREATED, OrderStatus.PENDING_SUBMIT)  # no raise

    def test_legal_submitted_to_acknowledged(self):
        assert_transition(OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED)

    def test_legal_unknown_exits(self):
        for nxt in (OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED,
                    OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
            assert_transition(OrderStatus.UNKNOWN, nxt)

    def test_illegal_created_to_filled(self):
        with pytest.raises(InvalidOrderTransition):
            assert_transition(OrderStatus.CREATED, OrderStatus.FILLED)

    def test_illegal_filled_reopen(self):
        with pytest.raises(InvalidOrderTransition):
            assert_transition(OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)

    def test_illegal_risk_rejected_any(self):
        with pytest.raises(InvalidOrderTransition):
            assert_transition(OrderStatus.RISK_REJECTED, OrderStatus.PENDING_SUBMIT)

    def test_partial_fill_idempotent(self):
        assert_transition(OrderStatus.PARTIALLY_FILLED, OrderStatus.PARTIALLY_FILLED)  # ok


class TestOrderLifecycle:
    def _order(self) -> Order:
        return Order(client_order_id="s-d-600000-BUY-001", strategy_id="s", decision_id="d",
                     symbol="600000", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                     qty=100, limit_price=10.0)

    def test_transition_appends_timeline(self):
        o = self._order()
        o.transition(OrderStatus.PENDING_SUBMIT, "checks ok")
        o.transition(OrderStatus.SUBMITTED, "sent")
        o.transition(OrderStatus.ACKNOWLEDGED, "confirmed")
        assert o.status == OrderStatus.ACKNOWLEDGED
        assert [e.status for e in o.timeline] == ["PENDING_SUBMIT", "SUBMITTED", "ACKNOWLEDGED"]
        assert not o.is_terminal()

    def test_terminal_states(self):
        for st in ORDER_TERMINAL_STATES:
            assert st in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
        o = self._order()
        o.transition(OrderStatus.PENDING_SUBMIT, "ok")
        o.transition(OrderStatus.SUBMITTED, "sent")
        o.transition(OrderStatus.FILLED, "done")
        assert o.is_terminal()
        with pytest.raises(InvalidOrderTransition):
            o.transition(OrderStatus.PARTIALLY_FILLED)

    def test_risk_rejected_is_absorbing(self):
        o = self._order()
        o.transition(OrderStatus.RISK_REJECTED, "blocked")
        assert o.status == OrderStatus.RISK_REJECTED
        with pytest.raises(InvalidOrderTransition):
            o.transition(OrderStatus.PENDING_SUBMIT)

    def test_json_roundtrip_keeps_timeline_and_status(self):
        o = self._order()
        o.transition(OrderStatus.PENDING_SUBMIT, "ok")
        o.transition(OrderStatus.SUBMITTED, "sent")
        o.broker_order_id = "B-1"
        clone = Order.from_dict(o.to_dict())
        assert clone.status == OrderStatus.SUBMITTED
        assert clone.broker_order_id == "B-1"
        assert [e.status for e in clone.timeline] == ["PENDING_SUBMIT", "SUBMITTED"]

    def test_unknown_is_first_class_from_any_nonterminal(self):
        for start in (OrderStatus.PENDING_SUBMIT, OrderStatus.SUBMITTED,
                      OrderStatus.ACKNOWLEDGED, OrderStatus.CANCEL_PENDING):
            assert_transition(start, OrderStatus.UNKNOWN)

    def test_new_client_order_id_deterministic(self):
        a = new_client_order_id("mom", "d1", "600000", "BUY", 3)
        b = new_client_order_id("mom", "d1", "600000", "BUY", 3)
        assert a == b == "mom-d1-600000-BUY-003"
