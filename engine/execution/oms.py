"""OrderManagementSystem: idempotent staging, pre-trade checks, strict state
machine, UNKNOWN-first failure handling, reconciliation, kill switch.

Core rules (Pangu 2.0 Phase 9):
- click-success != success: SUBMITTED only becomes ACKNOWLEDGED after the
  adapter confirms the order via query_order within ack_timeout_s.
- any submit raise/timeout -> UNKNOWN + manual_intervention_required; resubmit
  is FORBIDDEN until reconcile() resolves the order (NeedsReconciliation).
- reconcile matches by broker_order_id first, then by (symbol, side, qty,
  same-day); orders still UNKNOWN after reconcile keep the OMS trading_blocked.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

from engine.contracts import (
    CaptchaRequired,
    BrokerError,
    BrokerOrderRecord,
    BrokerPosition,
    BrokerTradeRecord,
    InvalidOrderTransition,
    Order,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    new_client_order_id,
)
from engine.execution.broker import BrokerAdapter
from engine.execution.risk_controls import CheckResult, PreTradeChecker, TradeContext

_NON_TERMINAL_FOR_SUBMIT = {OrderStatus.CREATED}
# local states that may still be alive at the broker and need reconciliation
_RECONCILABLE = {OrderStatus.PENDING_SUBMIT, OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED,
                 OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCEL_PENDING, OrderStatus.UNKNOWN}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_order_id TEXT PRIMARY KEY,
    json TEXT NOT NULL,
    status TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS order_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'order',
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS reconcile_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class NeedsReconciliation(RuntimeError):
    """Submit/cancel attempted while an order is UNKNOWN or not CREATED."""


@dataclass
class ReconcileReport:
    ts: str
    matched: int = 0
    mismatches: List[str] = field(default_factory=list)
    still_unknown: List[str] = field(default_factory=list)
    reconcile_ok: bool = True
    details: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"ts": self.ts, "matched": self.matched, "mismatches": self.mismatches,
                "still_unknown": self.still_unknown, "reconcile_ok": self.reconcile_ok,
                "details": self.details}


class OrderManagementSystem:
    def __init__(
        self,
        adapter: BrokerAdapter,
        checks: Optional[List[PreTradeChecker]] = None,
        db_path: str = "data/execution.db",
        ack_timeout_s: float = 5.0,
        config: Optional[Dict] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        self.adapter = adapter
        self.checks: List[PreTradeChecker] = list(checks or [])
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ack_timeout_s = float(ack_timeout_s)
        self._config = dict(config or {})
        self._now_fn = now_fn or datetime.now
        self._poll = float(poll_interval_s)
        self._init_db()

    # -- persistence --------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            for key, value in (("manual_intervention_required", ""),
                               ("trading_blocked", ""), ("kill_switch_engaged", ""),
                               ("kill_switch_reason", ""), ("kill_switch_operator", "")):
                conn.execute("INSERT OR IGNORE INTO system_state (key, value) VALUES (?, ?)",
                             (key, value))

    def _now(self) -> datetime:
        return self._now_fn()

    def _ts(self) -> str:
        return self._now().isoformat(timespec="seconds")

    def _audit(self, conn: sqlite3.Connection, kind: str, subject: str, status: str, detail: str = "") -> None:
        conn.execute("INSERT INTO order_events (ts, client_order_id, kind, status, detail)"
                     " VALUES (?, ?, ?, ?, ?)", (self._ts(), subject, kind, status, detail))

    def _save_order(self, conn: sqlite3.Connection, order: Order) -> None:
        conn.execute(
            "INSERT INTO orders (client_order_id, json, status, symbol, side, strategy_id,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(client_order_id) DO UPDATE SET json = excluded.json,"
            " status = excluded.status, updated_at = excluded.updated_at",
            (order.client_order_id, json.dumps(order.to_dict(), ensure_ascii=False),
             order.status.value, order.symbol, order.side.value, order.strategy_id,
             order.created_at, self._ts()))

    def _set_state(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO system_state (key, value) VALUES (?, ?)"
                         " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
            self._audit(conn, "system", key, value)

    def _get_state(self, key: str) -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM system_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else ""

    # -- order store --------------------------------------------------------
    def get_order(self, client_order_id: str) -> Order:
        with self._conn() as conn:
            row = conn.execute("SELECT json FROM orders WHERE client_order_id = ?",
                               (client_order_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown client_order_id {client_order_id}")
        return Order.from_dict(json.loads(row["json"]))

    def list_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        sql = "SELECT json FROM orders"
        args: tuple = ()
        if status is not None:
            sql += " WHERE status = ?"
            args = (status.value,)
        sql += " ORDER BY client_order_id"
        with self._conn() as conn:
            return [Order.from_dict(json.loads(r["json"])) for r in conn.execute(sql, args)]

    @staticmethod
    def _order_last_ts(order: Order) -> str:
        return order.timeline[-1].ts if order.timeline else (order.created_at or "")

    def _recent_orders(self, window_minutes: int = 60, exclude: str = "") -> List[Order]:
        cutoff = self._now() - timedelta(minutes=window_minutes)
        out = []
        for order in self.list_orders():
            if order.client_order_id == exclude:
                continue
            for raw in (self._order_last_ts(order), order.created_at):
                if raw:
                    try:
                        if datetime.fromisoformat(raw) >= cutoff:
                            out.append(order)
                    except ValueError:
                        pass
                    break
        return out

    # -- staging ------------------------------------------------------------
    def stage_order(self, strategy_id: str, decision_id: str, symbol: str, side: str,
                    qty: int, limit_price: Optional[float],
                    order_type: OrderType = OrderType.LIMIT, seq: int = 1) -> Order:
        """Deterministic client_order_id; re-staging the SAME key returns the
        existing order unchanged (idempotency)."""
        cid = new_client_order_id(strategy_id, decision_id, symbol, OrderSide(side).value, seq)
        try:
            return self.get_order(cid)
        except KeyError:
            pass
        order = Order(
            client_order_id=cid, strategy_id=strategy_id, decision_id=decision_id,
            symbol=symbol, side=OrderSide(side), order_type=order_type, qty=int(qty),
            limit_price=limit_price, created_at=self._ts())
        order.timeline.append(OrderEvent(ts=order.created_at, status="CREATED", detail="staged"))
        with self._conn() as conn:
            self._save_order(conn, order)
            self._audit(conn, "order", cid, "CREATED", "staged")
        return order

    # -- submit -------------------------------------------------------------
    def _build_ctx(self, order: Order) -> TradeContext:
        balance = None
        positions: List[BrokerPosition] = []
        try:
            balance = self.adapter.get_balance()
        except BrokerError:
            pass
        try:
            positions = list(self.adapter.get_positions() or [])
        except BrokerError:
            pass
        return TradeContext(
            balance=balance, positions=positions,
            recent_orders=self._recent_orders(exclude=order.client_order_id),
            reference_price=order.meta.get("reference_price"),
            now=self._now(), daily_pnl=float(self._config.get("daily_pnl", 0.0)),
            config=dict(self._config))

    def submit(self, client_order_id: str) -> Order:
        order = self.get_order(client_order_id)
        if order.status != OrderStatus.CREATED:
            raise NeedsReconciliation(
                f"order {client_order_id} is {order.status.value}; only CREATED orders may be submitted"
                + (" — run reconcile() first" if order.status == OrderStatus.UNKNOWN else ""))
        if not self.trading_allowed():
            reason = "trading blocked: " + "; ".join(
                f"{k}={v}" for k, v in (
                    ("kill_switch", self._get_state("kill_switch_engaged")),
                    ("manual_intervention", self._get_state("manual_intervention_required")),
                    ("trading_blocked", self._get_state("trading_blocked"))) if v)
            self._transition(order, OrderStatus.RISK_REJECTED, reason)
            return order

        ctx = self._build_ctx(order)
        for checker in self.checks:
            result: CheckResult = checker.check(order, ctx)
            if not result.approved:
                self._transition(order, OrderStatus.RISK_REJECTED,
                                 f"{type(checker).__name__}: {result.reason}")
                return order

        self._transition(order, OrderStatus.PENDING_SUBMIT, "checks passed")
        try:
            broker_order_id = self.adapter.submit_order(order)
        except CaptchaRequired as exc:  # P9-006: human must intervene
            self._go_unknown(order, f"captcha_required: {exc}")
            return order
        except (BrokerError, TimeoutError) as exc:
            self._go_unknown(order, f"submit_failed: {type(exc).__name__}: {exc}")
            return order

        order.broker_order_id = broker_order_id
        order.submitted_at = self._ts()
        self._transition(order, OrderStatus.SUBMITTED, f"broker_order_id={broker_order_id}")
        self._await_ack(order)
        return order

    def _await_ack(self, order: Order) -> None:
        """click-success != success: confirm the live order before ACKNOWLEDGED."""
        deadline = time.monotonic() + self._ack_timeout_s
        record = None
        while time.monotonic() < deadline and record is None:
            try:
                record = self.adapter.query_order(order.broker_order_id or "")
            except (BrokerError, TimeoutError) as exc:
                self._go_unknown(order, f"ack_query_failed: {type(exc).__name__}: {exc}")
                return
            if record is None:
                time.sleep(self._poll)
        if record is None:
            self._go_unknown(order,
                             f"ack_timeout: broker_order_id {order.broker_order_id} not confirmed "
                             f"within {self._ack_timeout_s}s")
            return
        self._transition(order, OrderStatus.ACKNOWLEDGED,
                         f"confirmed by broker: {record.status}")

    def _go_unknown(self, order: Order, reason: str) -> None:
        try:
            order.transition(OrderStatus.UNKNOWN, reason, ts=self._ts())
        except InvalidOrderTransition:
            return  # terminal already; leave untouched
        order.reject_reason = reason
        with self._conn() as conn:
            self._save_order(conn, order)
            self._audit(conn, "order", order.client_order_id, "UNKNOWN", reason)
            self._set_state_kv(conn, "manual_intervention_required", reason)

    def _transition(self, order: Order, nxt: OrderStatus, detail: str = "") -> None:
        order.transition(nxt, detail, ts=self._ts())
        if nxt in (OrderStatus.RISK_REJECTED, OrderStatus.REJECTED):
            order.reject_reason = detail or order.reject_reason
        with self._conn() as conn:
            self._save_order(conn, order)
            self._audit(conn, "order", order.client_order_id, nxt.value, detail)

    # -- cancel -------------------------------------------------------------
    def cancel(self, client_order_id: str) -> Order:
        order = self.get_order(client_order_id)
        if order.status == OrderStatus.CREATED:
            self._transition(order, OrderStatus.CANCELLED, "local cancel before submit")
            return order
        if order.status == OrderStatus.UNKNOWN:
            raise NeedsReconciliation(f"order {client_order_id} UNKNOWN; reconcile() first")
        if order.status not in {OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED,
                                OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_SUBMIT,
                                OrderStatus.CANCEL_PENDING}:
            raise InvalidOrderTransition(f"cannot cancel order in {order.status.value}")
        if order.status != OrderStatus.CANCEL_PENDING:
            self._transition(order, OrderStatus.CANCEL_PENDING, "cancel requested")
        try:
            ok = self.adapter.cancel_order(order.broker_order_id or "")
        except (BrokerError, TimeoutError) as exc:
            self._go_unknown(order, f"cancel_failed: {type(exc).__name__}: {exc}")
            return order
        if ok:
            self._transition(order, OrderStatus.CANCELLED, "cancel confirmed by broker")
        else:
            self._go_unknown(order, "cancel rejected by broker")
        return order

    # -- reconciliation -----------------------------------------------------
    def reconcile(self, trade_date: Optional[str] = None) -> ReconcileReport:
        report = ReconcileReport(ts=self._ts())
        self._set_state("trading_blocked", "")
        try:
            broker_orders = self.adapter.get_orders(trade_date)
            broker_trades = self.adapter.get_trades(trade_date)
        except BrokerError as exc:
            report.mismatches.append(f"adapter_unavailable: {exc}")
            report.reconcile_ok = False
            self._persist_report(report)
            return report

        trades_by_id: Dict[str, List[BrokerTradeRecord]] = {}
        for t in broker_trades:
            trades_by_id.setdefault(t.broker_order_id, []).append(t)
        used_broker_ids: set = {o.broker_order_id for o in self.list_orders() if o.broker_order_id}

        for order in self.list_orders():
            if order.status not in _RECONCILABLE:
                continue
            record = None
            if order.broker_order_id:
                for rec in broker_orders:
                    if rec.broker_order_id == order.broker_order_id:
                        record = rec
                        break
            else:
                record = self._match_by_key(order, broker_orders, used_broker_ids)
                if record is not None:
                    used_broker_ids.add(record.broker_order_id)
                    if order.status == OrderStatus.UNKNOWN:
                        self._transition(order, OrderStatus.ACKNOWLEDGED,
                                         f"reconcile: adopted broker_order_id={record.broker_order_id}")
                    else:
                        order.broker_order_id = record.broker_order_id
                        with self._conn() as conn:
                            self._save_order(conn, order)
            if record is None:
                if order.status == OrderStatus.UNKNOWN:
                    report.still_unknown.append(order.client_order_id)
                    with self._conn() as conn:
                        self._audit(conn, "reconcile", order.client_order_id, "UNKNOWN",
                                    "still unknown after reconcile")
                continue
            self._apply_record(order, record, trades_by_id.get(record.broker_order_id, []), report)

        if report.still_unknown:
            self._set_state("trading_blocked", f"unknown orders: {report.still_unknown}")
            report.reconcile_ok = False
        elif self._get_state("manual_intervention_required") and report.matched:
            self._set_state("manual_intervention_required", "")  # cause resolved
        self._persist_report(report)
        return report

    def _match_by_key(self, order: Order, broker_orders: List[BrokerOrderRecord],
                      used: set) -> Optional[BrokerOrderRecord]:
        day = (order.created_at or self._ts())[:10]
        for rec in broker_orders:
            if rec.broker_order_id in used or rec.symbol != order.symbol or rec.side != order.side \
                    or rec.qty != order.qty:
                continue
            rec_day = (rec.submitted_at or self._ts())[:10]
            if rec_day == day:
                return rec
        return None

    def _apply_record(self, order: Order, record: BrokerOrderRecord,
                      fills: List[BrokerTradeRecord], report: ReconcileReport) -> None:
        target = self._target_status(record, order)
        if fills:
            notional = sum(f.qty * f.price for f in fills)
            total_qty = sum(f.qty for f in fills)
            if total_qty > 0:
                order.avg_price = round(notional / total_qty, 4)
            order.filled_qty = max(order.filled_qty, record.filled_qty, total_qty)
        else:
            order.filled_qty = max(order.filled_qty, record.filled_qty)
            if record.filled_qty > 0 and record.price:
                order.avg_price = record.price
        if order.broker_order_id != record.broker_order_id:
            order.broker_order_id = record.broker_order_id
        if target == OrderStatus.REJECTED:
            order.reject_reason = (record.raw or {}).get("reject_reason") or record.status
        order.last_reconciled_at = self._ts()
        try:
            order.transition(target, f"reconcile: broker={record.status}", ts=self._ts())
        except InvalidOrderTransition as exc:
            report.mismatches.append(f"{order.client_order_id}: {exc}")
            with self._conn() as conn:
                self._save_order(conn, order)
            return
        report.matched += 1
        report.details.append({"client_order_id": order.client_order_id,
                               "broker_order_id": record.broker_order_id,
                               "status": target.value, "filled_qty": order.filled_qty,
                               "avg_price": order.avg_price})
        with self._conn() as conn:
            self._save_order(conn, order)
            self._audit(conn, "reconcile", order.client_order_id, target.value, record.status)

    @staticmethod
    def _target_status(record: BrokerOrderRecord, order: Order) -> OrderStatus:
        native = (record.status or "")
        upper = native.upper()
        filled = max(record.filled_qty, 0)
        if "REJECT" in upper or "废单" in native:
            return OrderStatus.REJECTED
        if "CANCEL" in upper or "撤" in native:
            if 0 < filled < order.qty:
                return OrderStatus.PARTIALLY_FILLED
            if filled >= order.qty > 0:
                return OrderStatus.FILLED
            return OrderStatus.CANCELLED
        if filled >= order.qty and filled > 0:
            return OrderStatus.FILLED
        if filled > 0:
            return OrderStatus.PARTIALLY_FILLED
        return OrderStatus.ACKNOWLEDGED

    def _persist_report(self, report: ReconcileReport) -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO reconcile_runs (ts, report_json) VALUES (?, ?)",
                         (report.ts, json.dumps(report.to_dict(), ensure_ascii=False)))

    # -- kill switch / gating ----------------------------------------------
    def trading_allowed(self) -> bool:
        if self._get_state("kill_switch_engaged") == "true":
            return False
        if self._get_state("manual_intervention_required"):
            return False
        if self._get_state("trading_blocked"):
            return False
        return True

    def engage_kill_switch(self, reason: str, operator: str) -> None:
        """Best-effort cancel of all non-terminal orders, then hard block."""
        for order in self.list_orders():
            if order.status in {OrderStatus.CREATED, OrderStatus.PENDING_SUBMIT}:
                self._transition(order, OrderStatus.CANCELLED, "kill_switch: local cancel")
            elif order.status in {OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED,
                                  OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCEL_PENDING} \
                    and order.broker_order_id:
                try:
                    ok = self.adapter.cancel_order(order.broker_order_id)
                except BrokerError:
                    ok = False
                try:
                    if ok:
                        self._transition(order, OrderStatus.CANCELLED, "kill_switch: broker cancel")
                    else:
                        self._go_unknown(order, "kill_switch: broker cancel unconfirmed")
                except InvalidOrderTransition:
                    pass
        with self._conn() as conn:
            conn.execute("INSERT INTO system_state (key, value) VALUES ('kill_switch_engaged', 'true')"
                         " ON CONFLICT(key) DO UPDATE SET value = 'true'")
            conn.execute("INSERT INTO system_state (key, value) VALUES ('kill_switch_reason', ?)"
                         " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (reason,))
            conn.execute("INSERT INTO system_state (key, value) VALUES ('kill_switch_operator', ?)"
                         " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (operator,))
            conn.execute("INSERT INTO system_state (key, value) VALUES ('kill_switch_ts', ?)"
                         " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (self._ts(),))
            self._audit(conn, "kill_switch", "kill_switch", "ENGAGED", f"{operator}: {reason}")

    def disengage_kill_switch(self, operator: str, reason: str) -> None:
        if not (reason or "").strip():
            raise ValueError("disengage_kill_switch requires a non-empty reason")
        with self._conn() as conn:
            conn.execute("INSERT INTO system_state (key, value) VALUES ('kill_switch_engaged', 'false')"
                         " ON CONFLICT(key) DO UPDATE SET value = 'false'")
            self._audit(conn, "kill_switch", "kill_switch", "DISENGAGED", f"{operator}: {reason}")

    def clear_manual_intervention(self, operator: str, reason: str) -> None:
        if not (reason or "").strip():
            raise ValueError("clear_manual_intervention requires a non-empty reason")
        self._set_state("manual_intervention_required", "")
        with self._conn() as conn:
            self._audit(conn, "system", "manual_intervention_required", "CLEARED",
                        f"{operator}: {reason}")

    # helper used inside an open connection
    def _set_state_kv(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute("INSERT INTO system_state (key, value) VALUES (?, ?)"
                     " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
