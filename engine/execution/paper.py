"""PaperBroker: SQLite-backed simulated broker for PAPER/SHADOW execution.

Fill model (deliberately conservative):
- LIMIT only; quote_provider(symbol, date) -> {open,high,low,close,preclose,is_st}|None.
- BUY:  limit_price < limit_up  (10% / 20% ST·创业板(3x)·科创板(68x))  else REJECTED limit_up_unbuyable.
- SELL: limit_price > limit_down else REJECTED limit_down_unsellable.
- fill price = limit_price * (1 ± slippage); BUY qty rounded down to 100-lot;
  cash check (insufficient_cash); SELL qty <= T+1 available (insufficient_position).
- partial fills via fill_ratio; cash/positions/trades updated in one transaction.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from engine.contracts import (
    BrokerBalance,
    BrokerOrderRecord,
    BrokerPosition,
    BrokerTradeRecord,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from engine.execution.broker import BrokerAdapter

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    qty INTEGER NOT NULL DEFAULT 0,
    avg_cost REAL NOT NULL DEFAULT 0,
    available_qty INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS lots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    qty INTEGER NOT NULL,
    price REAL NOT NULL,
    buy_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    broker_order_id TEXT PRIMARY KEY,
    client_order_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty INTEGER NOT NULL,
    price REAL,
    filled_qty INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    reject_reason TEXT,
    trade_date TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty INTEGER NOT NULL,
    price REAL NOT NULL,
    fees REAL NOT NULL DEFAULT 0,
    trade_date TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cash_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    broker_order_id TEXT,
    delta REAL NOT NULL,
    reason TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS seq (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    next_id INTEGER NOT NULL
);
INSERT OR IGNORE INTO seq (id, next_id) VALUES (1, 1);
"""


class PaperBroker(BrokerAdapter):
    name = "paper"

    def __init__(
        self,
        db_path: str = "data/paper_broker.db",
        quote_provider: Optional[Callable[[str, str], Optional[Dict]]] = None,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.0005,
        slippage_bps: float = 10.0,
        fill_ratio: float = 1.0,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        super().__init__()
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._quote_provider = quote_provider
        self._initial_cash = float(initial_cash)
        self._commission_rate = float(commission_rate)
        self._min_commission = float(min_commission)
        self._stamp_duty_rate = float(stamp_duty_rate)
        self._slippage = float(slippage_bps) / 10_000.0
        self._fill_ratio = float(fill_ratio)
        self._now_fn = now_fn or datetime.now
        self._date: str = self._now_fn().strftime("%Y%m%d")
        self._init_db()

    # -- plumbing -----------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT cash FROM accounts WHERE id = 1").fetchone()
            if row is None:
                conn.execute("INSERT INTO accounts (id, cash) VALUES (1, ?)", (self._initial_cash,))

    def _fees(self, side: str, notional: float) -> float:
        commission = max(self._commission_rate * notional, self._min_commission)
        stamp = self._stamp_duty_rate * notional if side == "SELL" else 0.0
        return round(commission + stamp, 2)

    @staticmethod
    def _price_limit_band(symbol: str, preclose: float, is_st: bool) -> tuple:
        base = symbol.split(".")[-1] if "." in symbol else symbol
        # ST → 5%（与 engine/validation/data_interface 一致）；
        # 创业板 sz.3 / 科创板 sh.68 → 20%；其余 10%。
        if is_st:
            ratio = 0.05
        elif base.startswith("3") or base.startswith("68"):
            ratio = 0.20
        else:
            ratio = 0.10
        return round(preclose * (1 + ratio), 2), round(preclose * (1 - ratio), 2)

    def set_date(self, date_str: str) -> None:
        """Advance the paper clock (YYYYMMDD). T+1 availability is recomputed."""
        self._date = date_str.replace("-", "")
        with self._conn() as conn:
            conn.execute(
                "UPDATE positions SET available_qty = "
                "COALESCE((SELECT SUM(qty) FROM lots WHERE lots.symbol = positions.symbol"
                " AND lots.buy_date < ?), 0)",
                (self._date,),
            )

    @property
    def current_date(self) -> str:
        return self._date

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        self._init_db()
        self._connected = True
        self._last_error = ""

    def health(self) -> Dict:
        base = super().health()
        base["date"] = self._date
        return base

    def _require_quote(self, symbol: str) -> Dict:
        quote = self._quote_provider(symbol, self._date) if self._quote_provider else None
        if not quote or not quote.get("preclose"):
            raise ValueError(f"no quote for {symbol} on {self._date}")
        return quote

    # -- submit -------------------------------------------------------------
    def submit_order(self, order: Order) -> str:
        self._require_connected()
        if order.order_type != OrderType.LIMIT or order.limit_price is None:
            raise ValueError("PaperBroker requires a LIMIT order with limit_price")
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            nxt = conn.execute("SELECT next_id FROM seq WHERE id = 1").fetchone()[0]
            conn.execute("UPDATE seq SET next_id = ? WHERE id = 1", (nxt + 1,))
            broker_order_id = f"PAPER-{nxt:08d}"
            ts = self._now_fn().isoformat(timespec="seconds")
            filled, reason, trade_row, cash_delta = 0, "", None, 0.0
            try:
                quote = self._require_quote(order.symbol)
            except ValueError as exc:
                reason = str(exc)
                quote = {}
            else:
                preclose = float(quote["preclose"])
                is_st = bool(quote.get("is_st", 0))
                limit_up, limit_down = self._price_limit_band(order.symbol, preclose, is_st)
                side = order.side.value
                if side == "BUY":
                    if not (order.limit_price < limit_up):
                        reason = "limit_up_unbuyable"
                    elif float(quote.get("low") or 0) > 0 and \
                            order.limit_price < float(quote["low"]):
                        # 限价低于当日最低价：非可成交限价单，日频模拟中
                        # 保守地拒绝（而不是按限价立即成交）
                        reason = "not_marketable_below_low"
                    else:
                        qty = (order.qty // 100) * 100
                        if qty <= 0:
                            reason = "qty_below_lot_size"
                        else:
                            fill_price = round(order.limit_price * (1 + self._slippage), 2)
                            fees = self._fees("BUY", qty * fill_price)
                            cash = conn.execute("SELECT cash FROM accounts WHERE id = 1").fetchone()[0]
                            if qty * fill_price + fees > cash + 1e-6:
                                reason = "insufficient_cash"
                            else:
                                filled = qty if self._fill_ratio >= 1.0 else \
                                    (int(order.qty * self._fill_ratio) // 100) * 100
                                if filled > 0:
                                    fees = self._fees("BUY", filled * fill_price)
                                    cash_delta = -(filled * fill_price + fees)
                                    trade_row = (broker_order_id, order.symbol, "BUY", filled,
                                                 fill_price, fees, self._date, ts)
                else:
                    if not (order.limit_price > limit_down):
                        reason = "limit_down_unsellable"
                    elif float(quote.get("high") or 0) > 0 and \
                            order.limit_price > float(quote["high"]):
                        # 限价高于当日最高价：非可成交限价单，保守拒绝
                        reason = "not_marketable_above_high"
                    else:
                        avail = conn.execute(
                            "SELECT COALESCE(SUM(qty), 0) FROM lots WHERE symbol = ? AND buy_date < ?",
                            (order.symbol, self._date)).fetchone()[0]
                        if order.qty > avail:
                            reason = "insufficient_position"
                        else:
                            qty = order.qty if self._fill_ratio >= 1.0 else \
                                max(0, int(order.qty * self._fill_ratio) // 100 * 100)
                            if qty > 0:
                                filled = qty
                                fill_price = round(order.limit_price * (1 - self._slippage), 2)
                                fees = self._fees("SELL", qty * fill_price)
                                cash_delta = qty * fill_price - fees
                                trade_row = (broker_order_id, order.symbol, "SELL", qty,
                                             fill_price, fees, self._date, ts)
            if reason:
                status = OrderStatus.REJECTED.value
            elif filled >= order.qty and filled > 0:
                status = OrderStatus.FILLED.value
            elif filled > 0:
                status = OrderStatus.PARTIALLY_FILLED.value
            else:
                status = OrderStatus.SUBMITTED.value
            conn.execute(
                "INSERT INTO orders (broker_order_id, client_order_id, symbol, side, qty, price,"
                " filled_qty, status, reject_reason, trade_date, ts)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (broker_order_id, order.client_order_id, order.symbol, order.side.value, order.qty,
                 order.limit_price, filled, status, reason, self._date, ts))
            if trade_row is not None:
                conn.execute(
                    "INSERT INTO trades (broker_order_id, symbol, side, qty, price, fees,"
                    " trade_date, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", trade_row)
                conn.execute("UPDATE accounts SET cash = cash + ? WHERE id = 1", (cash_delta,))
                conn.execute(
                    "INSERT INTO cash_ledger (trade_date, broker_order_id, delta, reason, ts)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (self._date, broker_order_id, cash_delta, f"trade {order.side.value}", ts))
                symbol, tqty, tprice, tdate = trade_row[1], trade_row[3], trade_row[4], trade_row[6]
                if order.side == OrderSide.BUY:
                    row = conn.execute("SELECT qty, avg_cost FROM positions WHERE symbol = ?",
                                       (symbol,)).fetchone()
                    old_qty, old_cost = (row["qty"], row["avg_cost"]) if row else (0, 0.0)
                    new_qty = old_qty + tqty
                    new_cost = (old_qty * old_cost + tqty * tprice + trade_row[5]) / new_qty
                    conn.execute(
                        "INSERT INTO positions (symbol, qty, avg_cost, available_qty) VALUES (?, ?, ?, 0)"
                        " ON CONFLICT(symbol) DO UPDATE SET qty = ?, avg_cost = ?",
                        (symbol, new_qty, round(new_cost, 4), new_qty, round(new_cost, 4)))
                    conn.execute("INSERT INTO lots (symbol, qty, price, buy_date) VALUES (?, ?, ?, ?)",
                                 (symbol, tqty, tprice, tdate))
                else:
                    conn.execute("UPDATE positions SET qty = qty - ? WHERE symbol = ?", (tqty, symbol))
                    self._consume_lots(conn, symbol, tqty, self._date)
            conn.commit()
        return broker_order_id

    @staticmethod
    def _consume_lots(conn: sqlite3.Connection, symbol: str, qty: int, today: str) -> None:
        remaining = qty
        for lot in conn.execute(
                "SELECT id, qty FROM lots WHERE symbol = ? AND buy_date < ? ORDER BY id",
                (symbol, today)).fetchall():
            if remaining <= 0:
                break
            take = min(lot["qty"], remaining)
            if take >= lot["qty"]:
                conn.execute("DELETE FROM lots WHERE id = ?", (lot["id"],))
            else:
                conn.execute("UPDATE lots SET qty = qty - ? WHERE id = ?", (take, lot["id"]))
            remaining -= take

    # -- queries ------------------------------------------------------------
    def get_balance(self) -> BrokerBalance:
        self._require_connected()
        with self._conn() as conn:
            cash = conn.execute("SELECT cash FROM accounts WHERE id = 1").fetchone()[0]
            mv = 0.0
            for pos in conn.execute("SELECT symbol, qty, avg_cost FROM positions WHERE qty > 0"):
                try:
                    quote = self._require_quote(pos["symbol"])
                    px = float(quote.get("close") or quote["preclose"])
                except ValueError:
                    px = pos["avg_cost"]
                mv += pos["qty"] * px
        return BrokerBalance(total_asset=round(cash + mv, 2), available_cash=round(cash, 2),
                             frozen_cash=0.0, market_value=round(mv, 2), asof=self._date)

    def get_positions(self) -> List[BrokerPosition]:
        self._require_connected()
        out: List[BrokerPosition] = []
        with self._conn() as conn:
            for pos in conn.execute("SELECT * FROM positions WHERE qty > 0"):
                try:
                    quote = self._require_quote(pos["symbol"])
                    px = float(quote.get("close") or quote["preclose"])
                except ValueError:
                    px = pos["avg_cost"]
                avail = conn.execute(
                    "SELECT COALESCE(SUM(qty), 0) FROM lots WHERE symbol = ? AND buy_date < ?",
                    (pos["symbol"], self._date)).fetchone()[0]
                out.append(BrokerPosition(symbol=pos["symbol"], qty=pos["qty"], sellable_qty=int(avail),
                                          avg_cost=pos["avg_cost"], market_value=round(pos["qty"] * px, 2),
                                          asof=self._date))
        return out

    def get_orders(self, trade_date: Optional[str] = None) -> List[BrokerOrderRecord]:
        self._require_connected()
        sql = "SELECT * FROM orders"
        args: tuple = ()
        if trade_date:
            sql += " WHERE trade_date = ?"
            args = (trade_date.replace("-", ""),)
        sql += " ORDER BY broker_order_id"
        out = []
        with self._conn() as conn:
            for row in conn.execute(sql, args):
                out.append(BrokerOrderRecord(
                    broker_order_id=row["broker_order_id"], symbol=row["symbol"],
                    side=OrderSide(row["side"]), qty=row["qty"], filled_qty=row["filled_qty"],
                    price=row["price"], status=row["status"], submitted_at=row["ts"],
                    raw={"client_order_id": row["client_order_id"],
                         "reject_reason": row["reject_reason"] or "",
                         "trade_date": row["trade_date"]}))
        return out

    def get_trades(self, trade_date: Optional[str] = None) -> List[BrokerTradeRecord]:
        self._require_connected()
        sql = "SELECT * FROM trades"
        args: tuple = ()
        if trade_date:
            sql += " WHERE trade_date = ?"
            args = (trade_date.replace("-", ""),)
        sql += " ORDER BY trade_id"
        out = []
        with self._conn() as conn:
            for row in conn.execute(sql, args):
                out.append(BrokerTradeRecord(
                    broker_trade_id=f"T{row['trade_id']:08d}", broker_order_id=row["broker_order_id"],
                    symbol=row["symbol"], side=OrderSide(row["side"]), qty=row["qty"],
                    price=row["price"], traded_at=row["ts"], raw={"fees": row["fees"]}))
        return out

    def cancel_order(self, broker_order_id: str) -> bool:
        self._require_connected()
        with self._conn() as conn:
            row = conn.execute("SELECT status FROM orders WHERE broker_order_id = ?",
                               (broker_order_id,)).fetchone()
            if row is None or row["status"] in (OrderStatus.FILLED.value, OrderStatus.CANCELLED.value,
                                                OrderStatus.REJECTED.value):
                return False
            conn.execute("UPDATE orders SET status = ? WHERE broker_order_id = ?",
                         (OrderStatus.CANCELLED.value, broker_order_id))
        return True

    def query_order(self, broker_order_id: str) -> Optional[BrokerOrderRecord]:
        self._require_connected()
        for rec in self.get_orders():
            if rec.broker_order_id == broker_order_id:
                return rec
        return None
