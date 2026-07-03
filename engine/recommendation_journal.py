"""Recommendation journal and forward-performance tracking.

This module records every daily candidate decision exactly as produced by the
pipeline, then evaluates later returns over fixed horizons. It is intentionally
separate from ``portfolio.py``: recommendations are a paper trail, not real
executed trades.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

from .data_loader import DataLoader, safe_float


HORIZONS = (1, 3, 5, 10)


@dataclass
class JournalSummary:
    total: int
    recommended: int
    evaluated: int
    win_rate: float
    avg_return: float
    avg_max_drawdown: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "recommended": self.recommended,
            "evaluated": self.evaluated,
            "win_rate": round(self.win_rate, 4),
            "avg_return": round(self.avg_return, 4),
            "avg_max_drawdown": round(self.avg_max_drawdown, 4),
        }


class RecommendationJournal:
    """SQLite-backed recommendation journal."""

    def __init__(
        self,
        db_path: str = "data/pangu.db",
        data_loader: Optional[DataLoader] = None,
    ) -> None:
        self.db_path = db_path
        self.dl = data_loader if data_loader is not None else DataLoader()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: Optional[sqlite3.Connection] = sqlite3.connect(":memory:") if db_path == ":memory:" else None
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            yield self._memory_conn
            return
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_journal (
                    run_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    board TEXT DEFAULT '',
                    xuanwu_status TEXT DEFAULT '',
                    is_recommended INTEGER NOT NULL DEFAULT 0,
                    recommend_score REAL DEFAULT 0,
                    grade TEXT DEFAULT '',
                    close_price REAL DEFAULT 0,
                    entry_price REAL DEFAULT 0,
                    stop_loss REAL DEFAULT 0,
                    take_profit REAL DEFAULT 0,
                    risk_reward REAL DEFAULT 0,
                    debate_verdict TEXT DEFAULT '',
                    debate_confidence REAL DEFAULT 0,
                    blockers_json TEXT DEFAULT '[]',
                    evidence_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_date, code)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_metrics (
                    run_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    eval_date TEXT NOT NULL,
                    close_return REAL DEFAULT 0,
                    high_return REAL DEFAULT 0,
                    low_return REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    win INTEGER NOT NULL DEFAULT 0,
                    evaluated_at TEXT NOT NULL,
                    PRIMARY KEY (run_date, code, horizon_days)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_status ON recommendation_journal(run_date, is_recommended, xuanwu_status)")
            conn.commit()

    def record_pipeline_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Upsert all candidates from a pipeline result."""
        run_date = str(data.get("date") or datetime.now().strftime("%Y%m%d"))
        now = datetime.now().isoformat(timespec="seconds")
        rows = []
        for c in data.get("candidates") or []:
            code = str(c.get("code") or "").strip().zfill(6)
            if not code or not code.isdigit():
                continue
            rec = c.get("recommend") or {}
            ee = c.get("entry_exit") or {}
            xw = c.get("xuanwu") or {}
            debate = c.get("debate") or {}
            buy_points = ee.get("buy_points") or []
            primary_buy = next((bp for bp in buy_points if bp.get("is_primary")), buy_points[0] if buy_points else {})
            stop_obj = ee.get("stop_loss") or {}
            targets = ee.get("take_profit") or []
            rows.append((
                run_date,
                code,
                str(c.get("name") or code),
                str(c.get("board") or ""),
                str(xw.get("status") or ""),
                1 if xw.get("status") == "xuanwu" else 0,
                safe_float(rec.get("recommend_score"), safe_float(c.get("recommend_score"), 0.0)),
                str(rec.get("grade") or c.get("grade") or ""),
                safe_float(c.get("close"), 0.0),
                safe_float(primary_buy.get("price"), 0.0) if isinstance(primary_buy, dict) else 0.0,
                safe_float(stop_obj.get("price"), 0.0) if isinstance(stop_obj, dict) else 0.0,
                safe_float(targets[0].get("price"), 0.0) if targets and isinstance(targets[0], dict) else 0.0,
                safe_float(ee.get("risk_reward_ratio"), safe_float(rec.get("risk_reward_ratio"), 0.0)),
                str(debate.get("verdict") or ""),
                safe_float(debate.get("confidence"), 0.0),
                json.dumps(xw.get("blockers") or [], ensure_ascii=False),
                json.dumps({
                    "xuanwu": xw,
                    "recommend": rec,
                    "reasons": c.get("reasons") or [],
                    "debate": debate,
                }, ensure_ascii=False),
                now,
            ))

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO recommendation_journal (
                    run_date, code, name, board, xuanwu_status, is_recommended,
                    recommend_score, grade, close_price, entry_price, stop_loss,
                    take_profit, risk_reward, debate_verdict, debate_confidence,
                    blockers_json, evidence_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_date, code) DO UPDATE SET
                    name=excluded.name,
                    board=excluded.board,
                    xuanwu_status=excluded.xuanwu_status,
                    is_recommended=excluded.is_recommended,
                    recommend_score=excluded.recommend_score,
                    grade=excluded.grade,
                    close_price=excluded.close_price,
                    entry_price=excluded.entry_price,
                    stop_loss=excluded.stop_loss,
                    take_profit=excluded.take_profit,
                    risk_reward=excluded.risk_reward,
                    debate_verdict=excluded.debate_verdict,
                    debate_confidence=excluded.debate_confidence,
                    blockers_json=excluded.blockers_json,
                    evidence_json=excluded.evidence_json
                """,
                rows,
            )
            conn.commit()
        return {"run_date": run_date, "recorded": len(rows), "recommended": sum(1 for r in rows if r[5])}

    def evaluate(self, as_of: Optional[str] = None, only_recommended: bool = False) -> dict[str, Any]:
        """Evaluate available journal rows over fixed horizons."""
        as_of = as_of or datetime.now().strftime("%Y%m%d")
        with self._connect() as conn:
            where = "WHERE is_recommended = 1" if only_recommended else ""
            records = conn.execute(
                f"""
                SELECT run_date, code, name, close_price
                FROM recommendation_journal
                {where}
                ORDER BY run_date DESC, recommend_score DESC
                """
            ).fetchall()

        inserted = 0
        skipped = 0
        for run_date, code, name, base_price in records:
            if str(run_date) >= str(as_of):
                skipped += 1
                continue
            metrics = self._evaluate_one(str(run_date), str(code), safe_float(base_price), str(as_of))
            if not metrics:
                skipped += 1
                continue
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO recommendation_metrics (
                        run_date, code, horizon_days, eval_date, close_return,
                        high_return, low_return, max_drawdown, win, evaluated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_date, code, horizon_days) DO UPDATE SET
                        eval_date=excluded.eval_date,
                        close_return=excluded.close_return,
                        high_return=excluded.high_return,
                        low_return=excluded.low_return,
                        max_drawdown=excluded.max_drawdown,
                        win=excluded.win,
                        evaluated_at=excluded.evaluated_at
                    """,
                    metrics,
                )
                conn.commit()
            inserted += len(metrics)
        return {"evaluated_metrics": inserted, "skipped": skipped, "as_of": as_of}

    def _evaluate_one(self, run_date: str, code: str, base_price: float, as_of: str) -> list[tuple[Any, ...]]:
        if base_price <= 0:
            return []
        try:
            k = self.dl.daily_kline(code, days=80, date=as_of)
        except Exception:
            return []
        if k is None or len(k) == 0:
            return []
        date_col = _find_col(k, ["日期", "date"])
        close_col = _find_col(k, ["收盘", "close", "收盘价"])
        high_col = _find_col(k, ["最高", "high", "最高价"])
        low_col = _find_col(k, ["最低", "low", "最低价"])
        if date_col is None or close_col is None:
            return []
        df = k.copy()
        df["_date"] = df[date_col].astype(str).str.replace("-", "", regex=False).str[:8]
        df = df[df["_date"] > run_date].sort_values("_date")
        if df.empty:
            return []
        rows = []
        now = datetime.now().isoformat(timespec="seconds")
        for horizon in HORIZONS:
            if len(df) < horizon:
                continue
            window = df.head(horizon)
            eval_row = window.iloc[-1]
            close_price = safe_float(eval_row.get(close_col), 0.0)
            high_price = safe_float(pd.to_numeric(window[high_col], errors="coerce").max(), close_price) if high_col else close_price
            low_price = safe_float(pd.to_numeric(window[low_col], errors="coerce").min(), close_price) if low_col else close_price
            close_ret = close_price / base_price - 1
            high_ret = high_price / base_price - 1
            low_ret = low_price / base_price - 1
            rows.append((
                run_date,
                code,
                horizon,
                str(eval_row.get("_date")),
                close_ret,
                high_ret,
                low_ret,
                min(0.0, low_ret),
                1 if close_ret > 0 else 0,
                now,
            ))
        return rows

    def summary(self, days: int = 30, only_recommended: bool = False) -> dict[str, Any]:
        with self._connect() as conn:
            cutoff_expr = "strftime('%Y%m%d', date('now', ?))"
            where_parts = [f"run_date >= {cutoff_expr}"]
            joined_where_parts = [f"j.run_date >= {cutoff_expr}"]
            params: list[Any] = [f"-{int(days)} days"]
            if only_recommended:
                where_parts.append("is_recommended = 1")
                joined_where_parts.append("j.is_recommended = 1")
            where = " AND ".join(where_parts)
            joined_where = " AND ".join(joined_where_parts)
            total, recommended = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(is_recommended),0) FROM recommendation_journal WHERE {where}",
                params,
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT m.horizon_days, m.close_return, m.max_drawdown, m.win
                FROM recommendation_metrics m
                JOIN recommendation_journal j ON j.run_date=m.run_date AND j.code=m.code
                WHERE {joined_where}
                """,
                params,
            ).fetchall()
            latest = conn.execute(
                f"""
                SELECT j.run_date, j.code, j.name, j.board, j.xuanwu_status,
                       j.is_recommended, j.recommend_score, j.grade,
                       j.debate_verdict, j.blockers_json,
                       m.horizon_days, m.close_return, m.max_drawdown, m.win
                FROM recommendation_journal j
                LEFT JOIN recommendation_metrics m ON j.run_date=m.run_date AND j.code=m.code AND m.horizon_days=1
                WHERE {joined_where}
                ORDER BY j.run_date DESC, j.recommend_score DESC
                LIMIT 80
                """,
                params,
            ).fetchall()
        by_horizon: dict[int, list[tuple[float, float, int]]] = {}
        for horizon, close_return, max_drawdown, win in rows:
            by_horizon.setdefault(int(horizon), []).append((safe_float(close_return), safe_float(max_drawdown), int(win or 0)))
        horizon_summary = {}
        for horizon, vals in sorted(by_horizon.items()):
            rets = [v[0] for v in vals]
            dds = [v[1] for v in vals]
            wins = [v[2] for v in vals]
            horizon_summary[horizon] = JournalSummary(
                total=int(total or 0),
                recommended=int(recommended or 0),
                evaluated=len(vals),
                win_rate=sum(wins) / len(wins) if wins else 0.0,
                avg_return=sum(rets) / len(rets) if rets else 0.0,
                avg_max_drawdown=sum(dds) / len(dds) if dds else 0.0,
            ).to_dict()
        return {
            "days": days,
            "only_recommended": only_recommended,
            "total": int(total or 0),
            "recommended": int(recommended or 0),
            "horizons": horizon_summary,
            "latest": [
                {
                    "run_date": r[0],
                    "code": r[1],
                    "name": r[2],
                    "board": r[3],
                    "xuanwu_status": r[4],
                    "is_recommended": bool(r[5]),
                    "recommend_score": round(safe_float(r[6]), 1),
                    "grade": r[7],
                    "debate_verdict": r[8],
                    "blockers": _loads_list(r[9]),
                    "horizon_1d": r[10],
                    "return_1d": round(safe_float(r[11]) * 100, 2) if r[11] is not None else None,
                    "drawdown_1d": round(safe_float(r[12]) * 100, 2) if r[12] is not None else None,
                    "win_1d": bool(r[13]) if r[13] is not None else None,
                }
                for r in latest
            ],
        }


def _find_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _loads_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []
