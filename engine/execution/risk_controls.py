"""Pre-trade risk checks run by the OMS before any submit (fail closed)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from engine.contracts import BrokerBalance, BrokerPosition, Order, OrderSide

DEFAULT_FEES = {"commission_rate": 0.0003, "min_commission": 5.0, "stamp_duty_rate": 0.0005}


@dataclass
class TradeContext:
    balance: Optional[BrokerBalance] = None
    positions: List[BrokerPosition] = field(default_factory=list)
    recent_orders: List[Order] = field(default_factory=list)
    reference_price: Optional[float] = None
    now: Optional[datetime] = None
    daily_pnl: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    approved: bool
    reason: str = ""


@runtime_checkable
class PreTradeChecker(Protocol):
    def check(self, order: Order, ctx: TradeContext) -> CheckResult: ...


def _order_ts(order: Order, ctx: TradeContext) -> Optional[datetime]:
    for raw in (order.submitted_at, order.created_at):
        if raw:
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                continue
    return ctx.now


def _est_fees(order: Order, cfg: Dict[str, Any]) -> float:
    notional = (order.limit_price or 0.0) * order.qty
    commission = max(cfg.get("commission_rate", DEFAULT_FEES["commission_rate"]) * notional,
                     cfg.get("min_commission", DEFAULT_FEES["min_commission"]))
    stamp = cfg.get("stamp_duty_rate", DEFAULT_FEES["stamp_duty_rate"]) * notional \
        if order.side == OrderSide.SELL else 0.0
    return commission + stamp


class DuplicateOrderGuard:
    """Block same symbol+side+strategy re-entry inside the window."""

    def __init__(self, window_minutes: int = 5) -> None:
        self.window_minutes = window_minutes

    def check(self, order: Order, ctx: TradeContext) -> CheckResult:
        now = ctx.now or datetime.now()
        for prev in ctx.recent_orders:
            if (prev.symbol, prev.side, prev.strategy_id) != (order.symbol, order.side, order.strategy_id):
                continue
            ts = _order_ts(prev, ctx)
            if ts is None:
                return CheckResult(False, f"duplicate_order: {prev.client_order_id} (no ts, conservative)")
            if (now - ts).total_seconds() <= self.window_minutes * 60:
                return CheckResult(False, f"duplicate_order: {prev.client_order_id} within {self.window_minutes}min")
        return CheckResult(True)


class PriceDeviationGuard:
    def __init__(self, max_deviation_pct: float = 0.03) -> None:
        self.max_deviation_pct = max_deviation_pct

    def check(self, order: Order, ctx: TradeContext) -> CheckResult:
        ref = ctx.reference_price
        if not ref or ref <= 0 or order.limit_price is None:
            return CheckResult(True, "no reference price; guard inert")
        dev = abs(order.limit_price - ref) / ref
        if dev > self.max_deviation_pct:
            return CheckResult(False, f"price_deviation {dev:.4f} > {self.max_deviation_pct}")
        return CheckResult(True)


class CashCheckGuard:
    """BUY: notional + estimated fees must fit available cash."""

    def check(self, order: Order, ctx: TradeContext) -> CheckResult:
        if order.side != OrderSide.BUY or order.limit_price is None:
            return CheckResult(True)
        if ctx.balance is None:
            return CheckResult(True, "no balance snapshot; guard inert")
        cfg = {**DEFAULT_FEES, **(ctx.config or {})}
        need = order.limit_price * order.qty + _est_fees(order, cfg)
        if need > ctx.balance.available_cash:
            return CheckResult(False, f"insufficient_cash: need {need:.2f} > available {ctx.balance.available_cash:.2f}")
        return CheckResult(True)


class PositionCheckGuard:
    """SELL: qty must be <= sellable (T+1 aware) position."""

    def check(self, order: Order, ctx: TradeContext) -> CheckResult:
        if order.side != OrderSide.SELL:
            return CheckResult(True)
        for pos in ctx.positions:
            if pos.symbol == order.symbol:
                if order.qty <= pos.sellable_qty:
                    return CheckResult(True)
                return CheckResult(False, f"insufficient_position: sell {order.qty} > sellable {pos.sellable_qty}")
        return CheckResult(False, f"insufficient_position: no position for {order.symbol}")


class RateLimitGuard:
    def __init__(self, max_per_day: int = 200, min_interval_seconds: int = 2) -> None:
        self.max_per_day = max_per_day
        self.min_interval_seconds = min_interval_seconds

    def check(self, order: Order, ctx: TradeContext) -> CheckResult:
        now = ctx.now or datetime.now()
        todays = 0
        for prev in ctx.recent_orders:
            ts = _order_ts(prev, ctx)
            if ts is not None and ts.date() == now.date():
                todays += 1
        if todays >= self.max_per_day:
            return CheckResult(False, f"rate_limit: {todays} orders today >= {self.max_per_day}")
        for prev in ctx.recent_orders:
            ts = _order_ts(prev, ctx)
            if ts is not None and (now - ts).total_seconds() < self.min_interval_seconds:
                return CheckResult(False, f"rate_limit: last order {prev.client_order_id} < {self.min_interval_seconds}s ago")
        return CheckResult(True)


class DailyLossCircuitBreaker:
    """soft_stop: block NEW positions (BUY) only; hard_stop: block all + flag account_stop."""

    def __init__(self, soft_stop: float = -0.02, hard_stop: float = -0.05) -> None:
        self.soft_stop = soft_stop
        self.hard_stop = hard_stop

    def check(self, order: Order, ctx: TradeContext) -> CheckResult:
        equity = (ctx.config or {}).get("equity", 0)
        if not equity or equity <= 0:
            return CheckResult(True, "no equity reference; guard inert")
        loss_frac = ctx.daily_pnl / float(equity)
        if loss_frac <= self.hard_stop:
            ctx.config["account_stop"] = True
            return CheckResult(False, f"hard_stop: daily_pnl {loss_frac:.4f} <= {self.hard_stop}; account stopped")
        if loss_frac <= self.soft_stop and order.side == OrderSide.BUY:
            return CheckResult(False, f"soft_stop: daily_pnl {loss_frac:.4f} <= {self.soft_stop}; new positions blocked")
        return CheckResult(True)
