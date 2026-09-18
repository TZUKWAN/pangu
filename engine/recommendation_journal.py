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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

import pandas as pd

from .data_loader import DataLoader, safe_float
from .short_term_replay import ReplayOutcome, ShortTermReplayConfig, ShortTermReplayEngine


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
            existing_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(recommendation_journal)").fetchall()
            }
            migrations = {
                "decision_status": "TEXT DEFAULT ''",
                "entry_plan_json": "TEXT DEFAULT '{}'",
                "exit_plan_json": "TEXT DEFAULT '{}'",
                "causal_news_available": "INTEGER NOT NULL DEFAULT 0",
                "data_quality": "TEXT DEFAULT ''",
                "historical_mode": "TEXT DEFAULT ''",
            }
            for column, ddl in migrations.items():
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE recommendation_journal ADD COLUMN {column} {ddl}")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS short_term_replay_metrics (
                    run_date TEXT NOT NULL,
                    code TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT DEFAULT '',
                    entry_date TEXT DEFAULT '',
                    exit_date TEXT DEFAULT '',
                    shares INTEGER DEFAULT 0,
                    holding_days INTEGER DEFAULT 0,
                    first_target_taken INTEGER NOT NULL DEFAULT 0,
                    gross_return REAL,
                    net_return REAL,
                    net_pnl REAL,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    causal_context_complete INTEGER NOT NULL DEFAULT 0,
                    details_json TEXT DEFAULT '{}',
                    evaluated_at TEXT NOT NULL,
                    PRIMARY KEY (run_date, code)
                )
                """
            )
            conn.commit()

    def record_pipeline_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Upsert decisions while preserving the formal Gate recommendation boundary."""
        run_date = str(data.get("date") or datetime.now().strftime("%Y%m%d"))
        now = datetime.now().isoformat(timespec="seconds")
        formal_contract = "final_recommendations" in data
        final_codes = {
            str(item.get("code") or "").zfill(6)
            for item in (data.get("final_recommendations") or [])
            if item.get("code")
        }
        watch_codes = {
            str(item.get("code") or "").zfill(6)
            for item in (data.get("watchlist") or [])
            if item.get("code")
        }
        rejected_codes = {
            str(item.get("code") or "").zfill(6)
            for item in (data.get("rejected") or [])
            if item.get("code")
        }
        all_candidates: dict[str, dict[str, Any]] = {}
        for bucket in ("candidates", "watchlist", "rejected", "final_recommendations"):
            for item in data.get(bucket) or []:
                code = str(item.get("code") or "").strip().zfill(6)
                if not code or not code.isdigit():
                    continue
                merged = dict(all_candidates.get(code) or {})
                merged.update(item)
                all_candidates[code] = merged

        news = data.get("news") or {}
        news_archive = (news.get("source_state") or {}).get("archive") or {}
        causal_news_available = bool(
            str(news.get("date") or "").replace("-", "") == run_date
            and news_archive.get("status") == "ok"
            and news_archive.get("mode") in {"exact_date", "fresh_cache", "persisted_exact_date"}
        )
        evidence_map = data.get("candidate_evidence") or {}
        rows = []
        for code, c in sorted(all_candidates.items()):
            rec = c.get("recommend") or {}
            ee = c.get("entry_exit") or {}
            xw = c.get("xuanwu") or {}
            debate = c.get("debate") or {}
            entry_plan = c.get("entry_plan") or ee.get("entry_plan") or {}
            exit_plan = c.get("exit_plan") or ee.get("exit_plan") or {}
            candidate_evidence = c.get("candidate_evidence") or evidence_map.get(code) or {}
            buy_points = ee.get("buy_points") or []
            primary_buy = next((bp for bp in buy_points if bp.get("is_primary")), buy_points[0] if buy_points else {})
            stop_obj = ee.get("stop_loss") or {}
            targets = ee.get("take_profit") or []
            if code in final_codes:
                decision_status = "final"
            elif code in rejected_codes:
                decision_status = "rejected"
            elif code in watch_codes:
                decision_status = "watch"
            else:
                decision_status = str(c.get("gate_status") or "candidate")
            is_recommended = (
                code in final_codes if formal_contract else xw.get("status") == "xuanwu"
            )
            rows.append((
                run_date,
                code,
                str(c.get("name") or code),
                str(c.get("board") or ""),
                str(xw.get("status") or ""),
                1 if is_recommended else 0,
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
                    "strategy": candidate_evidence.get("strategy") or {},
                    "news_evidence": c.get("news_evidence") or candidate_evidence.get("news_evidence") or {},
                    "anti_chase": c.get("anti_chase") or candidate_evidence.get("anti_chase") or {},
                    "price_action": candidate_evidence.get("price_action") or {},
                    "decision": candidate_evidence.get("decision") or {},
                    "entry_temperature": safe_float((data.get("sentiment") or {}).get("temperature"), 0.0),
                }, ensure_ascii=False),
                decision_status,
                json.dumps(entry_plan, ensure_ascii=False),
                json.dumps(exit_plan, ensure_ascii=False),
                1 if causal_news_available else 0,
                str(data.get("data_quality") or ""),
                str(data.get("historical_mode") or ""),
                now,
            ))

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO recommendation_journal (
                    run_date, code, name, board, xuanwu_status, is_recommended,
                    recommend_score, grade, close_price, entry_price, stop_loss,
                    take_profit, risk_reward, debate_verdict, debate_confidence,
                    blockers_json, evidence_json, decision_status, entry_plan_json,
                    exit_plan_json, causal_news_available, data_quality,
                    historical_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    evidence_json=excluded.evidence_json,
                    decision_status=excluded.decision_status,
                    entry_plan_json=excluded.entry_plan_json,
                    exit_plan_json=excluded.exit_plan_json,
                    causal_news_available=excluded.causal_news_available,
                    data_quality=excluded.data_quality,
                    historical_mode=excluded.historical_mode
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

    def evaluate_short_term(
        self,
        as_of: Optional[str] = None,
        *,
        cfg: ShortTermReplayConfig | Mapping[str, Any] | None = None,
        only_recommended: bool = True,
        context_provider: Optional[
            Callable[[str, str, pd.DataFrame, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]
        ] = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Replay formal recommendations with executable entry/exit rules and costs.

        ``context_provider`` must return exact-date per-trading-day news, market
        phase, temperature and theme state.  When it is absent or incomplete,
        price outcomes are still computed, but the 85% acceptance contract is
        explicitly blocked by causal-context coverage.
        """
        as_of = str(as_of or datetime.now().strftime("%Y%m%d"))
        engine = ShortTermReplayEngine(cfg)
        where = "WHERE is_recommended = 1" if only_recommended else ""
        with self._connect() as conn:
            records = conn.execute(
                f"""
                SELECT run_date, code, entry_plan_json, exit_plan_json,
                       evidence_json, causal_news_available
                FROM recommendation_journal
                {where}
                ORDER BY run_date, code
                """
            ).fetchall()

        outcomes: list[ReplayOutcome] = []
        skipped_future = 0
        provider_errors: list[str] = []
        for run_date, code, entry_json, exit_json, evidence_json, causal_news in records:
            run_date = str(run_date)
            code = str(code)
            if run_date >= as_of:
                skipped_future += 1
                continue
            try:
                kline = self.dl.daily_kline(code, days=120, date=as_of)
            except Exception as exc:  # noqa: BLE001
                outcomes.append(ReplayOutcome(run_date, code, "invalid", f"K线加载失败: {exc}"))
                continue
            evidence = _loads_dict(evidence_json)
            contexts: Mapping[str, Mapping[str, Any]] = {}
            if context_provider is not None:
                try:
                    contexts = context_provider(run_date, code, kline, evidence) or {}
                except Exception as exc:  # noqa: BLE001
                    provider_errors.append(f"{run_date}/{code}: {exc}")
            recommendation = {
                "code": code,
                "entry_plan": _loads_dict(entry_json),
                "exit_plan": _loads_dict(exit_json),
                "entry_exit": {"exit_plan": _loads_dict(exit_json)},
                "news_evidence": evidence.get("news_evidence") or {},
            }
            outcome = engine.replay(
                recommendation,
                kline,
                signal_date=run_date,
                daily_context=contexts,
                causal_signal_evidence=bool(causal_news),
            )
            outcomes.append(outcome)

        if persist and outcomes:
            evaluated_at = datetime.now().isoformat(timespec="seconds")
            rows = []
            for outcome in outcomes:
                payload = outcome.to_dict()
                rows.append((
                    outcome.signal_date,
                    outcome.code,
                    outcome.status,
                    outcome.reason,
                    outcome.entry_date,
                    outcome.exit_date,
                    outcome.shares,
                    outcome.holding_days,
                    1 if outcome.first_target_taken else 0,
                    outcome.gross_return,
                    outcome.net_return,
                    outcome.net_pnl,
                    outcome.max_favorable_excursion,
                    outcome.max_adverse_excursion,
                    1 if outcome.causal_context_complete else 0,
                    json.dumps(payload, ensure_ascii=False),
                    evaluated_at,
                ))
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO short_term_replay_metrics (
                        run_date, code, status, reason, entry_date, exit_date,
                        shares, holding_days, first_target_taken, gross_return,
                        net_return, net_pnl, max_favorable_excursion,
                        max_adverse_excursion, causal_context_complete,
                        details_json, evaluated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_date, code) DO UPDATE SET
                        status=excluded.status,
                        reason=excluded.reason,
                        entry_date=excluded.entry_date,
                        exit_date=excluded.exit_date,
                        shares=excluded.shares,
                        holding_days=excluded.holding_days,
                        first_target_taken=excluded.first_target_taken,
                        gross_return=excluded.gross_return,
                        net_return=excluded.net_return,
                        net_pnl=excluded.net_pnl,
                        max_favorable_excursion=excluded.max_favorable_excursion,
                        max_adverse_excursion=excluded.max_adverse_excursion,
                        causal_context_complete=excluded.causal_context_complete,
                        details_json=excluded.details_json,
                        evaluated_at=excluded.evaluated_at
                    """,
                    rows,
                )
                conn.commit()

        acceptance = engine.acceptance(outcomes)
        return {
            "as_of": as_of,
            "only_recommended": only_recommended,
            "evaluated_records": len(outcomes),
            "skipped_future_or_same_day": skipped_future,
            "context_provider_errors": provider_errors,
            "acceptance": acceptance,
            "outcomes": [item.to_dict() for item in outcomes],
        }

    def short_term_summary(
        self,
        *,
        days: int = 3650,
        cfg: ShortTermReplayConfig | Mapping[str, Any] | None = None,
        only_recommended: bool = True,
        limit: int = 80,
    ) -> dict[str, Any]:
        cutoff = (datetime.now() - timedelta(days=max(1, int(days)))).strftime("%Y%m%d")
        recommended_clause = "AND j.is_recommended = 1" if only_recommended else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT m.details_json, j.name, j.decision_status
                FROM short_term_replay_metrics m
                JOIN recommendation_journal j
                  ON j.run_date=m.run_date AND j.code=m.code
                WHERE m.run_date >= ? {recommended_clause}
                ORDER BY m.run_date DESC, m.code
                """,
                (cutoff,),
            ).fetchall()
        outcomes: list[ReplayOutcome] = []
        latest: list[dict[str, Any]] = []
        for details_json, name, decision_status in rows:
            details = _loads_dict(details_json)
            outcome = ReplayOutcome(
                signal_date=str(details.get("signal_date") or ""),
                code=str(details.get("code") or ""),
                status=str(details.get("status") or "invalid"),
                reason=str(details.get("reason") or ""),
                entry_date=str(details.get("entry_date") or ""),
                exit_date=str(details.get("exit_date") or ""),
                shares=int(details.get("shares") or 0),
                holding_days=int(details.get("holding_days") or 0),
                first_target_taken=bool(details.get("first_target_taken")),
                gross_return=(safe_float(details.get("gross_return")) if details.get("gross_return") is not None else None),
                net_return=(safe_float(details.get("net_return")) if details.get("net_return") is not None else None),
                net_pnl=(safe_float(details.get("net_pnl")) if details.get("net_pnl") is not None else None),
                causal_context_complete=bool(details.get("causal_context_complete")),
            )
            outcomes.append(outcome)
            if len(latest) < max(1, int(limit)):
                latest.append({**details, "name": name, "decision_status": decision_status})
        acceptance = ShortTermReplayEngine(cfg).acceptance(outcomes)
        return {
            "days": days,
            "only_recommended": only_recommended,
            "acceptance": acceptance,
            "latest": latest,
        }

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


def _loads_dict(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}") if isinstance(value, str) else value
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        return {}
