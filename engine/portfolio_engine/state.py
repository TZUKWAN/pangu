"""Persisted portfolio/risk state: plans, risk decisions, daily PnL,
high-water mark and drawdown.  Every call opens its own SQLite connection
(thread-safe by construction, no shared handle)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..contracts import RiskDecision
from .constructor import PortfolioPlan


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_plans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_date TEXT NOT NULL,
    rule_hash     TEXT NOT NULL DEFAULT '',
    cash_weight   REAL NOT NULL DEFAULT 0,
    plan_json     TEXT NOT NULL,
    ts            TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS risk_decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_date TEXT,
    symbol        TEXT DEFAULT '',
    strategy_id   TEXT DEFAULT '',
    approved      INTEGER NOT NULL,
    reason        TEXT DEFAULT '',
    decision_json TEXT NOT NULL,
    ts            TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS daily_pnl (
    date       TEXT PRIMARY KEY,
    realized   REAL NOT NULL DEFAULT 0,
    unrealized REAL NOT NULL DEFAULT 0,
    total      REAL NOT NULL DEFAULT 0,
    ts         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS account_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    ts    TEXT NOT NULL
);
"""


class PortfolioState:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = str(db_path)
        parent = Path(self.db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    def _get_state(self, conn: sqlite3.Connection, key: str, default: str = "") -> str:
        row = conn.execute("SELECT value FROM account_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def _set_state(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO account_state (key, value, ts) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, ts = excluded.ts",
            (key, value, _now_iso()),
        )

    # ------------------------------------------------------------------ #
    def record_plan(self, plan: PortfolioPlan, decision_date: Optional[str] = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO portfolio_plans (decision_date, rule_hash, cash_weight, plan_json, ts) "
                "VALUES (?,?,?,?,?)",
                (decision_date or plan.decision_date, plan.rule_hash, plan.cash_weight,
                 json.dumps({
                     "decision_date": plan.decision_date,
                     "allocations": plan.allocations,
                     "cash_weight": plan.cash_weight,
                     "notes": plan.notes,
                     "no_trade_reason": plan.no_trade_reason,
                 }, ensure_ascii=False),
                 _now_iso()),
            )
            return int(cur.lastrowid)

    def record_risk_decision(
        self,
        decision: RiskDecision,
        decision_date: str = "",
        symbol: str = "",
        strategy_id: str = "",
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO risk_decisions "
                "(decision_date, symbol, strategy_id, approved, reason, decision_json, ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (decision_date, symbol, strategy_id, int(bool(decision.approved)),
                 decision.reason,
                 json.dumps({"approved": decision.approved, "reason": decision.reason,
                             "checks": decision.checks}, ensure_ascii=False),
                 _now_iso()),
            )
            return int(cur.lastrowid)

    # ------------------------------------------------------------------ #
    def update_daily_pnl(self, date: str, realized: float, unrealized: float) -> Dict[str, float]:
        total = float(realized) + float(unrealized)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO daily_pnl (date, realized, unrealized, total, ts) VALUES (?,?,?,?,?) "
                "ON CONFLICT(date) DO UPDATE SET realized = excluded.realized, "
                "unrealized = excluded.unrealized, total = excluded.total, ts = excluded.ts",
                (date, float(realized), float(unrealized), total, _now_iso()),
            )
        return {"realized": float(realized), "unrealized": float(unrealized), "total": total}

    def get_daily_pnl(self, date: str) -> Optional[Dict[str, float]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT realized, unrealized, total FROM daily_pnl WHERE date = ?", (date,)
            ).fetchone()
        if row is None:
            return None
        return {"realized": row[0], "unrealized": row[1], "total": row[2]}

    # ------------------------------------------------------------------ #
    def update_equity(self, equity: float) -> float:
        """Persist high-water mark; returns the current drawdown (0..1)."""
        equity = float(equity)
        with self._connect() as conn:
            prev_hw = float(self._get_state(conn, "high_water", "0") or 0)
            hw = max(prev_hw, equity)
            self._set_state(conn, "high_water", repr(hw))
            self._set_state(conn, "last_equity", repr(equity))
        dd = (hw - equity) / hw if hw > 0 else 0.0
        return max(0.0, dd)

    def high_water(self) -> float:
        with self._connect() as conn:
            return float(self._get_state(conn, "high_water", "0") or 0)

    def current_drawdown(self, equity: Optional[float] = None) -> float:
        """Drawdown of `equity` (or the last persisted equity) vs high water."""
        if equity is None:
            with self._connect() as conn:
                equity = float(self._get_state(conn, "last_equity", "0") or 0)
        hw = self.high_water()
        if hw <= 0:
            return 0.0
        return max(0.0, (hw - float(equity)) / hw)
