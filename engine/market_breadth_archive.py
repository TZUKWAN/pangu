"""Build exact-date A-share market-breadth sentiment archives.

The archive uses BaoStock's historical daily security universe and unadjusted
daily bars.  It is deliberately separate from the recommendation logic: raw
cross-sectional facts are persisted first, then a transparent sentiment proxy
is derived without looking at any later return.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST"
_WEIGHTS = {
    "limit_up_count": 0.13,
    "limit_down_count": 0.09,
    "advance_decline": 0.08,
    "strong_gain_ratio": 0.09,
    "ma_bullish_ratio": 0.09,
    "above_ma20_ratio": 0.09,
    "volatility": 0.08,
}


@dataclass(frozen=True)
class BreadthBuildResult:
    start: str
    end: str
    trade_dates: int
    universe_codes: int
    completed_codes: int
    failed_codes: int
    exact_context_dates: int
    output_root: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class BaoStockBreadthArchive:
    def __init__(self, root: str | Path = "data/market_breadth") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "raw.sqlite3"

    def build(
        self,
        start: str,
        end: str,
        *,
        warmup_calendar_days: int = 45,
        overwrite: bool = False,
    ) -> BreadthBuildResult:
        start_date = datetime.strptime(start, "%Y%m%d").date()
        end_date = datetime.strptime(end, "%Y%m%d").date()
        if start_date > end_date:
            raise ValueError("start must be <= end")
        history_start = (start_date - timedelta(days=max(30, warmup_calendar_days))).strftime("%Y-%m-%d")
        history_end = end_date.strftime("%Y-%m-%d")

        try:
            import baostock as bs
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("baostock is required for historical breadth archival") from exc

        login = bs.login()
        if str(login.error_code) != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
        connection = sqlite3.connect(self.db_path)
        try:
            ohlc_migrated = self._init_db(connection)
            if ohlc_migrated:
                # Existing close-only rows must be refreshed before this DB can
                # drive exact entry/exit replay. Universe rows remain reusable.
                connection.execute("DELETE FROM breadth_progress")
                connection.commit()
            if overwrite:
                connection.execute("DELETE FROM breadth_raw")
                connection.execute("DELETE FROM breadth_progress")
                connection.execute("DELETE FROM breadth_universe")
                connection.commit()

            trade_dates = self._trade_dates(bs, start_date, end_date)
            universe = self._cached_universe(connection, trade_dates)
            missing_dates = [date for date in trade_dates if date not in universe]
            if missing_dates:
                fetched_universe = self._load_universe(
                    bs, missing_dates, connection=connection
                )
                universe.update(fetched_universe)
            all_codes = sorted({code for rows in universe.values() for code in rows})
            completed = {
                row[0] for row in connection.execute(
                    """SELECT code FROM breadth_progress
                    WHERE status='ok' AND covered_start<=? AND covered_end>=?""",
                    (history_start.replace("-", ""), history_end.replace("-", "")),
                ).fetchall()
            }
            pending = [code for code in all_codes if code not in completed]
            for index, code in enumerate(pending, start=1):
                status, error = self._fetch_code(
                    bs, connection, code, history_start, history_end
                )
                if index == 1 or index % 100 == 0 or index == len(pending):
                    print(
                        f"breadth history {index}/{len(pending)} code={code} "
                        f"status={status}",
                        flush=True,
                    )
                if error and index % 100 == 0:
                    print(f"latest error: {error}", flush=True)

            records = self._aggregate_from_db(connection, universe, start, end)
            exact_dates = 0
            for date, record in records.items():
                if record["data_quality"]["market_context_exact"]:
                    exact_dates += 1
                self._save_date(date, record)
            failures = int(connection.execute(
                "SELECT COUNT(*) FROM breadth_progress WHERE status!='ok'"
            ).fetchone()[0])
            completed_count = int(connection.execute(
                "SELECT COUNT(*) FROM breadth_progress WHERE status='ok'"
            ).fetchone()[0])
            result = BreadthBuildResult(
                start=start,
                end=end,
                trade_dates=len(trade_dates),
                universe_codes=len(all_codes),
                completed_codes=completed_count,
                failed_codes=failures,
                exact_context_dates=exact_dates,
                output_root=str(self.root),
            )
            self._save_manifest(result, records)
            return result
        finally:
            connection.close()
            bs.logout()

    @staticmethod
    def _result_rows(result: Any) -> list[list[str]]:
        """Read BaoStock ResultData without its pandas-2-incompatible get_data."""
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        return rows

    def _trade_dates(self, bs: Any, start: Any, end: Any) -> list[str]:
        result = bs.query_trade_dates(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        if str(result.error_code) != "0":
            raise RuntimeError(f"trade dates failed: {result.error_code} {result.error_msg}")
        rows = self._result_rows(result)
        fields = list(result.fields)
        date_i = fields.index("calendar_date")
        trading_i = fields.index("is_trading_day")
        return [row[date_i].replace("-", "") for row in rows if row[trading_i] == "1"]

    def _load_universe(
        self,
        bs: Any,
        dates: Iterable[str],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, dict[str, str]]:
        universe: dict[str, dict[str, str]] = {}
        date_list = list(dates)
        for index, date in enumerate(date_list, start=1):
            result = bs.query_all_stock(day=datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d"))
            if str(result.error_code) != "0":
                raise RuntimeError(f"universe {date} failed: {result.error_code} {result.error_msg}")
            fields = list(result.fields)
            rows = self._result_rows(result)
            code_i = fields.index("code")
            status_i = fields.index("tradeStatus")
            name_i = fields.index("code_name")
            daily = {
                row[code_i]: row[name_i]
                for row in rows
                if row[status_i] == "1" and self._is_a_share(row[code_i])
            }
            if len(daily) < 1000:
                raise RuntimeError(
                    f"universe {date} incomplete: only {len(daily)} active A-share rows"
                )
            universe[date] = daily
            if connection is not None:
                self._persist_universe(connection, {date: daily})
            if index == 1 or index % 10 == 0 or index == len(date_list):
                print(
                    f"breadth universe {index}/{len(date_list)} date={date} rows={len(daily)}",
                    flush=True,
                )
        return universe

    @staticmethod
    def _is_a_share(code: str) -> bool:
        if code.startswith("sh."):
            return code[3:].startswith("6") and len(code[3:]) == 6
        if code.startswith("sz."):
            return code[3:].startswith(("000", "001", "002", "003", "300", "301"))
        return False

    @staticmethod
    def _init_db(connection: sqlite3.Connection) -> bool:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS breadth_raw (
                date TEXT NOT NULL,
                code TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                preclose REAL,
                volume REAL,
                amount REAL,
                turnover REAL,
                pct_change REAL,
                is_st INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(date, code)
            );
            CREATE TABLE IF NOT EXISTS breadth_progress (
                code TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                error TEXT,
                updated_at TEXT NOT NULL,
                covered_start TEXT NOT NULL DEFAULT '',
                covered_end TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS breadth_universe (
                date TEXT NOT NULL,
                code TEXT NOT NULL,
                name TEXT,
                PRIMARY KEY(date, code)
            );
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(breadth_raw)").fetchall()
        }
        migrated = False
        for name in ("open", "high", "low", "preclose", "turnover"):
            if name not in columns:
                connection.execute(f"ALTER TABLE breadth_raw ADD COLUMN {name} REAL")
                migrated = True
        progress_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(breadth_progress)").fetchall()
        }
        for name in ("covered_start", "covered_end"):
            if name not in progress_columns:
                connection.execute(
                    f"ALTER TABLE breadth_progress ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                )
        connection.execute(
            """UPDATE breadth_progress SET
            covered_start=COALESCE(NULLIF(covered_start,''),(
                SELECT min(date) FROM breadth_raw r WHERE r.code=breadth_progress.code
            ),''),
            covered_end=COALESCE(NULLIF(covered_end,''),(
                SELECT max(date) FROM breadth_raw r WHERE r.code=breadth_progress.code
            ),'')
            WHERE covered_start='' OR covered_end=''"""
        )
        connection.commit()
        return migrated

    @staticmethod
    def _persist_universe(
        connection: sqlite3.Connection,
        universe: Mapping[str, Mapping[str, str]],
    ) -> None:
        rows = [
            (date, code, name)
            for date, stocks in universe.items()
            for code, name in stocks.items()
        ]
        connection.executemany(
            "INSERT OR REPLACE INTO breadth_universe(date,code,name) VALUES(?,?,?)",
            rows,
        )
        connection.commit()

    @staticmethod
    def _cached_universe(
        connection: sqlite3.Connection,
        dates: Iterable[str],
    ) -> dict[str, dict[str, str]]:
        requested = set(dates)
        if not requested:
            return {}
        rows = connection.execute(
            "SELECT date,code,name FROM breadth_universe"
        ).fetchall()
        output: dict[str, dict[str, str]] = {}
        for date, code, name in rows:
            if date in requested:
                output.setdefault(date, {})[code] = name or code
        # A valid A-share trading day has thousands of rows. Treat tiny/partial
        # sets as incomplete so an interrupted universe fetch is repaired.
        return {date: stocks for date, stocks in output.items() if len(stocks) >= 1000}

    def _fetch_code(
        self,
        bs: Any,
        connection: sqlite3.Connection,
        code: str,
        start: str,
        end: str,
    ) -> tuple[str, str]:
        error = ""
        for attempt in range(1, 4):
            try:
                result = bs.query_history_k_data_plus(
                    code,
                    _FIELDS,
                    start_date=start,
                    end_date=end,
                    frequency="d",
                    adjustflag="3",
                )
                if str(result.error_code) != "0":
                    raise RuntimeError(f"{result.error_code} {result.error_msg}")
                fields = list(result.fields)
                rows = self._result_rows(result)
                parsed = self._parse_history(fields, rows)
                if not parsed:
                    raise RuntimeError("empty history")
                with connection:
                    connection.executemany(
                        """INSERT OR REPLACE INTO breadth_raw
                        (date,code,open,high,low,close,preclose,volume,amount,turnover,pct_change,is_st)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        parsed,
                    )
                    connection.execute(
                        """INSERT OR REPLACE INTO breadth_progress
                        (code,status,error,updated_at,covered_start,covered_end)
                        VALUES(?,?,?,?,?,?)""",
                        (
                            code, "ok", "", datetime.now(timezone.utc).isoformat(),
                            start.replace("-", ""), end.replace("-", ""),
                        ),
                    )
                return "ok", ""
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                if attempt < 3:
                    time.sleep(0.2 * attempt)
        with connection:
            connection.execute(
                """INSERT OR REPLACE INTO breadth_progress
                (code,status,error,updated_at,covered_start,covered_end)
                VALUES(?,?,?,?,?,?)""",
                (
                    code, "failed", error, datetime.now(timezone.utc).isoformat(),
                    start.replace("-", ""), end.replace("-", ""),
                ),
            )
        return "failed", error

    @staticmethod
    def _parse_history(fields: list[str], rows: list[list[str]]) -> list[tuple[Any, ...]]:
        indices = {name: fields.index(name) for name in fields}
        output: list[tuple[Any, ...]] = []
        for row in rows:
            if row[indices["tradestatus"]] != "1":
                continue
            try:
                output.append((
                    row[indices["date"]].replace("-", ""),
                    row[indices["code"]],
                    float(row[indices["open"]]),
                    float(row[indices["high"]]),
                    float(row[indices["low"]]),
                    float(row[indices["close"]]),
                    float(row[indices["preclose"]] or 0),
                    float(row[indices["volume"]] or 0),
                    float(row[indices["amount"]] or 0),
                    float(row[indices["turn"]] or 0),
                    float(row[indices["pctChg"]] or 0),
                    int(float(row[indices["isST"]] or 0)),
                ))
            except (TypeError, ValueError):
                continue
        return output

    def _aggregate_from_db(
        self,
        connection: sqlite3.Connection,
        universe: Mapping[str, Mapping[str, str]],
        start: str,
        end: str,
    ) -> dict[str, dict[str, Any]]:
        frame = pd.read_sql_query(
            "SELECT * FROM breadth_raw WHERE date<=? ORDER BY code,date",
            connection,
            params=(end,),
        )
        return self.aggregate(frame, universe, start=start, end=end)

    @classmethod
    def aggregate(
        cls,
        frame: pd.DataFrame,
        universe: Mapping[str, Mapping[str, str]],
        *,
        start: str,
        end: str,
    ) -> dict[str, dict[str, Any]]:
        if frame.empty:
            return {}
        frame = frame.copy().sort_values(["code", "date"])
        frame["synthetic"] = frame.groupby("code")["pct_change"].transform(
            lambda values: (1.0 + pd.to_numeric(values, errors="coerce").fillna(0) / 100.0).cumprod()
        )
        grouped = frame.groupby("code", group_keys=False)["synthetic"]
        frame["ma5"] = grouped.transform(lambda values: values.rolling(5, min_periods=5).mean())
        frame["ma10"] = grouped.transform(lambda values: values.rolling(10, min_periods=10).mean())
        frame["ma20"] = grouped.transform(lambda values: values.rolling(20, min_periods=20).mean())
        frame["above_ma20"] = frame["synthetic"] > frame["ma20"]
        frame["ma_bullish"] = (frame["ma5"] > frame["ma10"]) & (frame["ma10"] > frame["ma20"])
        output: dict[str, dict[str, Any]] = {}
        for date in sorted(d for d in universe if start <= d <= end):
            expected = set(universe[date])
            daily = frame[(frame["date"] == date) & frame["code"].isin(expected)].copy()
            count = len(daily)
            coverage = count / len(expected) if expected else 0.0
            pct = pd.to_numeric(daily["pct_change"], errors="coerce").dropna()
            advance = int((pct > 0).sum())
            decline = int((pct < 0).sum())
            flat = int((pct == 0).sum())
            limit_mask = daily.apply(cls._limit_threshold, axis=1) if count else pd.Series(dtype=float)
            limit_up = int((daily["pct_change"] >= limit_mask).sum()) if count else 0
            limit_down = int((daily["pct_change"] <= -limit_mask).sum()) if count else 0
            valid_ma = daily[daily["ma20"].notna()]
            above_ma20 = float(valid_ma["above_ma20"].mean()) if len(valid_ma) else None
            ma_bullish = float(valid_ma["ma_bullish"].mean()) if len(valid_ma) else None
            ad_total = advance + decline
            ad_ratio = advance / ad_total if ad_total else 0.5
            strong_ratio = float((pct > 5).mean()) if len(pct) else 0.0
            volatility = float(pct.std(ddof=1)) if len(pct) > 1 else 0.0
            components = {
                "limit_up_count": {"raw": limit_up, "score": cls._anchor(limit_up, 15, 40, 80, 120)},
                "limit_down_count": {"raw": limit_down, "score": cls._clamp(100 - limit_down / 60 * 100)},
                "advance_decline": {"raw": f"{advance}/{decline}", "score": cls._clamp(ad_ratio * 100)},
                "strong_gain_ratio": {"raw": round(strong_ratio, 6), "score": cls._anchor(strong_ratio * 100, 2, 5, 15, 25)},
                "ma_bullish_ratio": {"raw": None if ma_bullish is None else round(ma_bullish, 6), "score": None if ma_bullish is None else cls._anchor(ma_bullish * 100, 10, 30, 60, 80)},
                "above_ma20_ratio": {"raw": None if above_ma20 is None else round(above_ma20, 6), "score": None if above_ma20 is None else cls._anchor(above_ma20 * 100, 15, 35, 65, 85)},
                "volatility": {"raw": round(volatility, 6), "score": cls._clamp(100 - max(0.0, volatility - 3.0) / 3.0 * 100)},
            }
            available = {
                key: value for key, value in components.items() if value["score"] is not None
            }
            weight_total = sum(_WEIGHTS[key] for key in available)
            temperature = sum(
                float(value["score"]) * _WEIGHTS[key] for key, value in available.items()
            ) / weight_total if weight_total else 0.0
            exact = bool(expected and coverage >= 0.98 and len(pct) / len(expected) >= 0.98)
            output[date] = {
                "date": date,
                "temperature": round(temperature, 2),
                "posture": "cold" if temperature < 40 else ("overheated" if temperature > 85 else "normal"),
                "breadth": {
                    "expected_trading_stocks": len(expected),
                    "observed_stocks": count,
                    "advance": advance,
                    "decline": decline,
                    "flat": flat,
                    "advance_ratio": round(ad_ratio, 6),
                    "median_pct_change": round(float(pct.median()), 6) if len(pct) else None,
                    "strong_gain_count": int((pct > 5).sum()),
                    "strong_loss_count": int((pct < -5).sum()),
                    "limit_up_count_approx": limit_up,
                    "limit_down_count_approx": limit_down,
                    "total_amount": round(float(daily["amount"].sum()), 2),
                },
                "components": components,
                "data_quality": {
                    "market_context_exact": exact,
                    "exact_trade_date_universe": True,
                    "coverage": round(coverage, 6),
                    "universe_source": "baostock.query_all_stock(date)",
                    "kline_source": "baostock.query_history_k_data_plus(unadjusted)",
                    "limit_counts_approximate": True,
                    "causal": True,
                    "warnings": [
                        "涨跌停家数由历史涨跌幅和当日ST/板块阈值近似，非交易所涨跌停池",
                        "情绪温度仅使用可历史重建的横截面分项，未填造连板/炸板/题材轮动",
                    ],
                },
            }
        return output

    @staticmethod
    def _limit_threshold(row: pd.Series) -> float:
        if int(row.get("is_st") or 0) == 1:
            return 4.8
        code = str(row.get("code") or "")
        if code.startswith(("sh.688", "sh.689", "sz.300", "sz.301")):
            return 19.5
        return 9.5

    @staticmethod
    def _clamp(value: float) -> float:
        return round(max(0.0, min(100.0, float(value))), 2)

    @classmethod
    def _anchor(cls, value: float, cold: float, normal: float, hot: float, extreme: float) -> float:
        value = float(value)
        if value <= cold:
            return cls._clamp(25.0 * value / cold) if cold else 0.0
        if value <= normal:
            return cls._clamp(25.0 + 25.0 * (value - cold) / (normal - cold))
        if value <= hot:
            return cls._clamp(50.0 + 30.0 * (value - normal) / (hot - normal))
        if value <= extreme:
            return cls._clamp(80.0 + 20.0 * (value - hot) / (extreme - hot))
        return 100.0

    def _save_date(self, date: str, record: Mapping[str, Any]) -> None:
        target = self.root / f"{date}.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(record), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)

    def _save_manifest(
        self,
        result: BreadthBuildResult,
        records: Mapping[str, Mapping[str, Any]],
    ) -> None:
        target = self.root / "manifest.json"
        body = result.to_dict()
        body["generated_at"] = datetime.now(timezone.utc).isoformat()
        body["dates"] = sorted(records)
        body["research_limitations"] = [
            "该档案是市场情绪的因果横截面代理，不替代新闻与题材语义证据",
            "涨跌停数量为价格阈值近似，不能冒充精确涨停池或连板数据",
        ]
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)

    def audit(self, start: str, end: str, *, output: str | Path | None = None) -> dict[str, Any]:
        """Run reusable quality gates for the persisted universe and OHLCV bars."""
        with sqlite3.connect(self.db_path) as connection:
            scalar = lambda sql, params=(): connection.execute(sql, params).fetchone()[0]
            row_count = int(scalar("SELECT COUNT(*) FROM breadth_raw"))
            distinct_keys = int(scalar(
                "SELECT COUNT(*) FROM (SELECT date,code FROM breadth_raw GROUP BY date,code)"
            ))
            scope_params = (start, end)
            expected_scope = int(scalar(
                "SELECT COUNT(*) FROM breadth_universe WHERE date BETWEEN ? AND ?",
                scope_params,
            ))
            observed_scope = int(scalar(
                "SELECT COUNT(*) FROM breadth_raw WHERE date BETWEEN ? AND ?",
                scope_params,
            ))
            missing_join = int(scalar(
                """SELECT COUNT(*) FROM breadth_universe u
                LEFT JOIN breadth_raw r ON r.date=u.date AND r.code=u.code
                WHERE u.date BETWEEN ? AND ? AND r.code IS NULL""",
                scope_params,
            ))
            orphan_scope = int(scalar(
                """SELECT COUNT(*) FROM breadth_raw r
                LEFT JOIN breadth_universe u ON u.date=r.date AND u.code=r.code
                WHERE r.date BETWEEN ? AND ? AND u.code IS NULL""",
                scope_params,
            ))
            null_ohlc = int(scalar(
                "SELECT COUNT(*) FROM breadth_raw WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL"
            ))
            nonpositive_prices = int(scalar(
                "SELECT COUNT(*) FROM breadth_raw WHERE open<=0 OR high<=0 OR low<=0 OR close<=0"
            ))
            invalid_high_low = int(scalar(
                """SELECT COUNT(*) FROM breadth_raw
                WHERE high < max(open,close,low) OR low > min(open,close,high)"""
            ))
            invalid_volume_amount = int(scalar(
                "SELECT COUNT(*) FROM breadth_raw WHERE volume<0 OR amount<0"
            ))
            extreme_first_observation = int(scalar(
                """SELECT COUNT(*) FROM breadth_raw r
                WHERE abs(r.pct_change)>1000 AND NOT EXISTS (
                    SELECT 1 FROM breadth_raw prior
                    WHERE prior.code=r.code AND prior.date<r.date
                )"""
            ))
            extreme_pct = int(scalar(
                """SELECT COUNT(*) FROM breadth_raw r
                WHERE abs(r.pct_change)>1000 AND EXISTS (
                    SELECT 1 FROM breadth_raw prior
                    WHERE prior.code=r.code AND prior.date<r.date
                )"""
            ))
            pct_mismatch = int(scalar(
                """SELECT COUNT(*) FROM breadth_raw
                WHERE preclose>0 AND abs(((close/preclose)-1)*100-pct_change)>0.15"""
            ))
            completed = int(scalar(
                "SELECT COUNT(*) FROM breadth_progress WHERE status='ok'"
            ))
            failed = int(scalar(
                "SELECT COUNT(*) FROM breadth_progress WHERE status!='ok'"
            ))
            min_date, max_date = connection.execute(
                "SELECT min(date),max(date) FROM breadth_raw"
            ).fetchone()
            coverage_rows = connection.execute(
                """SELECT u.date,COUNT(*) expected,COUNT(r.code) observed
                FROM breadth_universe u
                LEFT JOIN breadth_raw r ON r.date=u.date AND r.code=u.code
                WHERE u.date BETWEEN ? AND ? GROUP BY u.date ORDER BY u.date""",
                scope_params,
            ).fetchall()
        daily_coverage = [
            {
                "date": date,
                "expected": expected,
                "observed": observed,
                "coverage": round(observed / expected, 6) if expected else 0.0,
            }
            for date, expected, observed in coverage_rows
        ]
        min_coverage = min((row["coverage"] for row in daily_coverage), default=0.0)
        checks = {
            "primary_key_unique": row_count == distinct_keys,
            "ohlc_complete": null_ohlc == 0,
            "prices_positive": nonpositive_prices == 0,
            "ohlc_relationship_valid": invalid_high_low == 0,
            "volume_amount_nonnegative": invalid_volume_amount == 0,
            "pct_change_domain_valid": extreme_pct == 0,
            "pct_change_reconciles_preclose": pct_mismatch == 0,
            "scope_join_complete": missing_join == 0 and orphan_scope == 0,
            "daily_coverage_100pct": min_coverage == 1.0,
            "all_codes_completed": failed == 0 and completed > 0,
            "date_boundary_valid": bool(min_date and max_date and max_date <= end),
        }
        blockers = [name for name, passed in checks.items() if not passed]
        report = {
            "dataset": str(self.db_path),
            "grain": "one unadjusted daily OHLCV row per exact trading date and A-share code",
            "scope": {"start": start, "end": end},
            "status": "ready" if not blockers else "needs_revision",
            "checks": checks,
            "blockers": blockers,
            "metrics": {
                "row_count": row_count,
                "distinct_date_code_keys": distinct_keys,
                "expected_scope_rows": expected_scope,
                "observed_scope_rows": observed_scope,
                "missing_universe_bars": missing_join,
                "orphan_scope_bars": orphan_scope,
                "null_ohlc_rows": null_ohlc,
                "nonpositive_price_rows": nonpositive_prices,
                "invalid_high_low_rows": invalid_high_low,
                "negative_volume_or_amount_rows": invalid_volume_amount,
                "extreme_pct_change_rows": extreme_pct,
                "extreme_first_observation_rows": extreme_first_observation,
                "pct_preclose_mismatch_rows": pct_mismatch,
                "completed_codes": completed,
                "failed_codes": failed,
                "min_date": min_date,
                "max_date": max_date,
                "trade_dates": len(daily_coverage),
                "minimum_daily_coverage": min_coverage,
            },
            "daily_coverage": daily_coverage,
            "limitations": [
                "BaoStock is a third-party public source rather than an exchange system of record",
                "breadth limit-up/down counts remain threshold approximations, not exact limit-pool records",
                "this audit validates market data quality, not the 85% recommendation claim",
            ],
            "warnings": ([
                f"{extreme_first_observation} first-observation rows exceed 1000%; verified/listing-day review required"
            ] if extreme_first_observation else []),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        target = Path(output) if output else self.root / f"audit_{start}_{end}.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)
        return report


class ArchivedBreadthKlineLoader:
    """Read exact BaoStock OHLCV bars from the breadth archive SQLite file."""

    def __init__(self, db_path: str | Path = "data/market_breadth/raw.sqlite3") -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)
        with sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(breadth_raw)").fetchall()
            }
        required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
        missing = required - columns
        if missing:
            raise RuntimeError(f"breadth archive lacks OHLCV columns: {sorted(missing)}")

    def daily_kline(
        self,
        code: str,
        days: int = 120,
        date: str | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        normalized = str(code).zfill(6)
        provider_code = f"sh.{normalized}" if normalized.startswith("6") else f"sz.{normalized}"
        end = str(date or "99999999")
        query = """
            SELECT date,open,high,low,close,volume,amount,turnover,pct_change
            FROM breadth_raw
            WHERE code=? AND date<=?
            ORDER BY date DESC LIMIT ?
        """
        with sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True) as connection:
            rows = connection.execute(query, (provider_code, end, max(1, int(days)))).fetchall()
        if not rows:
            return pd.DataFrame()
        rows.reverse()
        return pd.DataFrame(rows, columns=[
            "日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率", "涨跌幅",
        ])


def main() -> int:
    parser = argparse.ArgumentParser(description="回填精确交易日A股市场广度/情绪档案")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--root", default="data/market_breadth")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    for value in (args.start, args.end):
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as exc:
            parser.error(str(exc))
    archive = BaoStockBreadthArchive(args.root)
    if args.audit_only:
        report = archive.audit(args.start, args.end, output=args.audit_output)
        print(json.dumps({
            "status": report["status"],
            "blockers": report["blockers"],
            "metrics": report["metrics"],
        }, ensure_ascii=False), flush=True)
        return 0 if report["status"] == "ready" else 1
    result = archive.build(args.start, args.end, overwrite=args.overwrite)
    print(json.dumps(result.to_dict(), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
