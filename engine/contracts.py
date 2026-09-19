"""Pangu 2.0 shared contracts.

Single source of truth for types shared across research / portfolio /
execution / compliance / web.  All sub-systems MUST import from here
instead of redefining their own enums/dataclasses, so that the OMS,
strategy registry, risk engine and web layer never drift apart.

Design rules (from docs/Pangu_2.0 task list):
- ExecutionMode default is PAPER.  LIVE is impossible unless every gate
  (strategy validated, paper, shadow, broker ok, compliance CONFIRMED,
  kill switch armed) passes; the gates live in engine/strategies,
  engine/compliance and engine/execution.
- Order lifecycle is a strict state machine.  UNKNOWN is a first-class
  state: after any submit timeout the order MUST go UNKNOWN and the OMS
  must reconcile before resubmitting (idempotency by client_order_id).
- Illegal transitions raise InvalidOrderTransition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Execution / compliance / strategy lifecycle
# ---------------------------------------------------------------------------

class ExecutionMode(str, Enum):
    DISABLED = "DISABLED"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    MANUAL_CONFIRM = "MANUAL_CONFIRM"
    LIVE = "LIVE"


DEFAULT_EXECUTION_MODE = ExecutionMode.PAPER


class ComplianceState(str, Enum):
    UNKNOWN = "UNKNOWN"
    NOT_REQUIRED_CONFIRMED = "NOT_REQUIRED_CONFIRMED"
    REPORT_REQUIRED = "REPORT_REQUIRED"
    SUBMITTED = "SUBMITTED"
    CONFIRMED = "CONFIRMED"
    EXPIRED_OR_CHANGED = "EXPIRED_OR_CHANGED"


COMPLIANCE_LIVE_OK = {ComplianceState.CONFIRMED, ComplianceState.NOT_REQUIRED_CONFIRMED}


class StrategyStatus(str, Enum):
    IDEA = "idea"
    RESEARCH = "research"
    VALIDATED = "validated"
    PAPER = "paper"
    SHADOW_LIVE = "shadow_live"
    LIMITED_LIVE = "limited_live"
    LIVE = "live"
    SUSPENDED = "suspended"
    RETIRED = "retired"


# A strategy may only emit executable orders when its status is here AND
# the global execution mode allows it.
EXECUTABLE_STRATEGY_STATUS = {StrategyStatus.PAPER, StrategyStatus.SHADOW_LIVE,
                              StrategyStatus.LIMITED_LIVE, StrategyStatus.LIVE}


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"      # market-like (limit with price protection upstream)
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    RISK_REJECTED = "RISK_REJECTED"
    PENDING_SUBMIT = "PENDING_SUBMIT"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


# Terminal states (no further transitions allowed except audit notes).
ORDER_TERMINAL_STATES = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}

# Allowed transitions of the order state machine.  Anything not listed is
# illegal and must raise.  UNKNOWN may be reached from any non-terminal
# state (lost contact with broker) and may only leave via reconciliation
# (to ACKNOWLEDGED / PARTIALLY_FILLED / FILLED / CANCELLED / REJECTED).
_ORDER_T: Dict[OrderStatus, set] = {
    OrderStatus.CREATED: {OrderStatus.RISK_REJECTED, OrderStatus.PENDING_SUBMIT, OrderStatus.CANCELLED},
    OrderStatus.PENDING_SUBMIT: {OrderStatus.SUBMITTED, OrderStatus.UNKNOWN, OrderStatus.CANCELLED, OrderStatus.REJECTED},
    OrderStatus.SUBMITTED: {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
                            OrderStatus.CANCEL_PENDING, OrderStatus.CANCELLED, OrderStatus.REJECTED,
                            OrderStatus.UNKNOWN},
    OrderStatus.ACKNOWLEDGED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCEL_PENDING,
                               OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.UNKNOWN},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCEL_PENDING,
                                   OrderStatus.CANCELLED, OrderStatus.UNKNOWN},
    OrderStatus.CANCEL_PENDING: {OrderStatus.CANCELLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
                                 OrderStatus.UNKNOWN},
    OrderStatus.UNKNOWN: {OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
                          OrderStatus.CANCELLED, OrderStatus.REJECTED},
    OrderStatus.RISK_REJECTED: set(),
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
}


class InvalidOrderTransition(ValueError):
    pass


def assert_transition(current: OrderStatus, nxt: OrderStatus) -> None:
    if current == nxt and current == OrderStatus.PARTIALLY_FILLED:
        return  # idempotent partial fill updates
    if nxt not in _ORDER_T.get(current, set()):
        raise InvalidOrderTransition(f"illegal order transition {current.value} -> {nxt.value}")


@dataclass
class OrderEvent:
    ts: str
    status: str
    detail: str = ""


@dataclass
class Order:
    """One client order.  client_order_id is globally unique and idempotent."""
    client_order_id: str
    strategy_id: str
    decision_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: int
    limit_price: Optional[float] = None
    created_at: Optional[str] = None
    submitted_at: Optional[str] = None
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.CREATED
    filled_qty: int = 0
    avg_price: Optional[float] = None
    reject_reason: Optional[str] = None
    last_reconciled_at: Optional[str] = None
    timeline: List[OrderEvent] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def transition(self, nxt: OrderStatus, detail: str = "", ts: Optional[str] = None) -> None:
        assert_transition(self.status, nxt)
        self.status = nxt
        self.timeline.append(OrderEvent(ts=ts or _now_iso(), status=nxt.value, detail=detail))

    def is_terminal(self) -> bool:
        return self.status in ORDER_TERMINAL_STATES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "strategy_id": self.strategy_id,
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "qty": self.qty,
            "limit_price": self.limit_price,
            "created_at": self.created_at,
            "submitted_at": self.submitted_at,
            "broker_order_id": self.broker_order_id,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "avg_price": self.avg_price,
            "reject_reason": self.reject_reason,
            "last_reconciled_at": self.last_reconciled_at,
            "timeline": [{"ts": e.ts, "status": e.status, "detail": e.detail} for e in self.timeline],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Order":
        return cls(
            client_order_id=d["client_order_id"],
            strategy_id=d.get("strategy_id", ""),
            decision_id=d.get("decision_id", ""),
            symbol=d["symbol"],
            side=OrderSide(d.get("side", "BUY")),
            order_type=OrderType(d.get("order_type", "LIMIT")),
            qty=int(d.get("qty", 0)),
            limit_price=d.get("limit_price"),
            created_at=d.get("created_at"),
            submitted_at=d.get("submitted_at"),
            broker_order_id=d.get("broker_order_id"),
            status=OrderStatus(d.get("status", "CREATED")),
            filled_qty=int(d.get("filled_qty", 0)),
            avg_price=d.get("avg_price"),
            reject_reason=d.get("reject_reason"),
            last_reconciled_at=d.get("last_reconciled_at"),
            timeline=[OrderEvent(ts=e.get("ts", ""), status=e.get("status", ""),
                                 detail=e.get("detail", "")) for e in d.get("timeline", [])],
            meta=dict(d.get("meta", {})),
        )


# ---------------------------------------------------------------------------
# Broker-facing snapshots (what a BrokerAdapter returns)
# ---------------------------------------------------------------------------

@dataclass
class BrokerBalance:
    total_asset: float
    available_cash: float
    frozen_cash: float
    market_value: float
    asof: str


@dataclass
class BrokerPosition:
    symbol: str
    qty: int
    sellable_qty: int          # respects T+1
    avg_cost: float
    market_value: float
    asof: str


@dataclass
class BrokerOrderRecord:
    broker_order_id: str
    symbol: str
    side: OrderSide
    qty: int
    filled_qty: int
    price: Optional[float]
    status: str                # broker-native string, adapter maps where possible
    submitted_at: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrokerTradeRecord:
    broker_trade_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    qty: int
    price: float
    traded_at: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class BrokerError(RuntimeError):
    pass


class BrokerUnavailable(BrokerError):
    """Adapter cannot reach the broker right now."""


class CaptchaRequired(BrokerError):
    """Broker/client requires manual human intervention; live loop must pause."""


# ---------------------------------------------------------------------------
# Portfolio / risk
# ---------------------------------------------------------------------------

@dataclass
class PortfolioTarget:
    strategy_id: str
    symbol: str
    target_weight: float        # fraction of portfolio equity, may be 0..max
    expected_return: Optional[float] = None
    confidence: Optional[float] = None   # calibrated only; None means raw
    calibrated: bool = False
    risk_score: Optional[float] = None
    raw_score: Optional[float] = None    # uncalibrated heuristic score
    decision_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDecision:
    approved: bool
    reason: str = ""
    checks: Dict[str, Any] = field(default_factory=dict)
    adjusted_target: Optional[PortfolioTarget] = None


class RiskBlocked(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_client_order_id(strategy_id: str, decision_id: str, symbol: str, side: str, seq: int) -> str:
    """Deterministic, collision-free id: retries reuse the same id (idempotency)."""
    return f"{strategy_id}-{decision_id}-{symbol}-{side}-{seq:03d}"


def is_live_mode(mode: ExecutionMode) -> bool:
    return mode == ExecutionMode.LIVE
