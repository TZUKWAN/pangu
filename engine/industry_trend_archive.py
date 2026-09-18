"""Exact-date industry membership and causal short-term industry trend archive."""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class BaoStockIndustryTrendArchive:
    def __init__(
        self,
        root: str | Path = "data/industry_trend",
        *,
        breadth_db: str | Path = "data/market_breadth/raw.sqlite3",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.breadth_db = Path(breadth_db)
        if not self.breadth_db.exists():
            raise FileNotFoundError(self.breadth_db)

    def build(self, start: str, end: str, *, overwrite: bool = False) -> dict[str, Any]:
        try:
            import baostock as bs
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("baostock is required") from exc
        connection = sqlite3.connect(self.breadth_db)
        self._init_db(connection)
        if overwrite:
            connection.execute("DELETE FROM industry_membership WHERE date BETWEEN ? AND ?", (start, end))
            connection.execute("DELETE FROM industry_progress WHERE date BETWEEN ? AND ?", (start, end))
            connection.commit()
        dates = [
            row[0] for row in connection.execute(
                "SELECT DISTINCT date FROM breadth_universe WHERE date BETWEEN ? AND ? ORDER BY date",
                (start, end),
            ).fetchall()
        ]
        complete = {
            row[0] for row in connection.execute(
                "SELECT date FROM industry_progress WHERE status='ok' AND date BETWEEN ? AND ?",
                (start, end),
            ).fetchall()
        }
        login = bs.login()
        if str(login.error_code) != "0":
            connection.close()
            raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
        try:
            for index, date in enumerate((item for item in dates if item not in complete), start=1):
                self._fetch_date(bs, connection, date)
                if index == 1 or index % 10 == 0:
                    print(f"industry membership {index}/{len(dates) - len(complete)} date={date}", flush=True)
            reports = self._aggregate(connection, start, end)
            for date, payload in reports.items():
                self._save(date, payload)
            failed = int(connection.execute(
                "SELECT COUNT(*) FROM industry_progress WHERE status!='ok' AND date BETWEEN ? AND ?",
                (start, end),
            ).fetchone()[0])
            exact = sum(
                1 for payload in reports.values()
                if payload["data_quality"]["industry_context_exact"]
            )
            manifest = {
                "start": start,
                "end": end,
                "trade_dates": len(dates),
                "generated_dates": len(reports),
                "exact_context_dates": exact,
                "failed_dates": failed,
                "method": (
                    "exact-date CSRC industry membership; score is same-day cross-sectional percentile of "
                    "causal 5-day member return, advance ratio, and above-MA20 breadth"
                ),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_json(self.root / "manifest.json", manifest)
            return manifest
        finally:
            connection.close()
            bs.logout()

    @staticmethod
    def _init_db(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS industry_membership (
                date TEXT NOT NULL,
                code TEXT NOT NULL,
                industry TEXT NOT NULL,
                classification TEXT,
                update_date TEXT,
                PRIMARY KEY(date,code)
            );
            CREATE TABLE IF NOT EXISTS industry_progress (
                date TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                member_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.commit()

    @staticmethod
    def _result_rows(result: Any) -> list[list[str]]:
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        return rows

    def _fetch_date(self, bs: Any, connection: sqlite3.Connection, date: str) -> None:
        result = bs.query_stock_industry(date=datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d"))
        if str(result.error_code) != "0":
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO industry_progress VALUES(?,?,?,?,?)",
                    (date, "failed", 0, f"{result.error_code} {result.error_msg}", datetime.now(timezone.utc).isoformat()),
                )
            return
        fields = list(result.fields)
        indices = {name: fields.index(name) for name in fields}
        universe = {
            row[0] for row in connection.execute(
                "SELECT code FROM breadth_universe WHERE date=?", (date,)
            ).fetchall()
        }
        rows = []
        future_updates = 0
        for row in self._result_rows(result):
            code = row[indices["code"]]
            industry = row[indices["industry"]].strip()
            update_date = row[indices["updateDate"]].replace("-", "")
            if code not in universe or not industry:
                continue
            if update_date and update_date > date:
                future_updates += 1
                continue
            rows.append((
                date,
                code,
                industry,
                row[indices["industryClassification"]],
                update_date,
            ))
        status = "ok" if rows and future_updates == 0 else "failed"
        error = "" if status == "ok" else f"rows={len(rows)} future_updates={future_updates}"
        with connection:
            connection.execute("DELETE FROM industry_membership WHERE date=?", (date,))
            connection.executemany(
                """INSERT INTO industry_membership
                (date,code,industry,classification,update_date) VALUES(?,?,?,?,?)""",
                rows,
            )
            connection.execute(
                "INSERT OR REPLACE INTO industry_progress VALUES(?,?,?,?,?)",
                (date, status, len(rows), error, datetime.now(timezone.utc).isoformat()),
            )

    def _aggregate(
        self,
        connection: sqlite3.Connection,
        start: str,
        end: str,
    ) -> dict[str, dict[str, Any]]:
        bars = pd.read_sql_query(
            """SELECT date,code,pct_change,open,close FROM breadth_raw
            WHERE date<=? ORDER BY code,date""",
            connection,
            params=(end,),
        )
        membership = pd.read_sql_query(
            """SELECT date,code,industry FROM industry_membership
            WHERE date BETWEEN ? AND ?""",
            connection,
            params=(start, end),
        )
        if bars.empty or membership.empty:
            return {}
        bars["growth"] = 1.0 + pd.to_numeric(bars["pct_change"], errors="coerce").fillna(0) / 100.0
        grouped = bars.groupby("code", group_keys=False)
        bars["ret5"] = grouped["growth"].transform(
            lambda values: values.rolling(5, min_periods=5).apply(np.prod, raw=True) - 1.0
        ) * 100.0
        bars["synthetic"] = grouped["growth"].transform(lambda values: values.cumprod())
        bars["ma20"] = bars.groupby("code")["synthetic"].transform(
            lambda values: values.rolling(20, min_periods=20).mean()
        )
        bars["above_ma20"] = bars["synthetic"] > bars["ma20"]
        scope_bars = bars[(bars["date"] >= start) & (bars["date"] <= end)]
        merged = membership.merge(scope_bars, on=["date", "code"], how="left", validate="one_to_one")
        merged["advance"] = merged["pct_change"] > 0
        records = merged.groupby(["date", "industry"], as_index=False).agg(
            member_count=("code", "count"),
            observed_count=("pct_change", "count"),
            median_ret5=("ret5", "median"),
            median_day_return=("pct_change", "median"),
            advance_ratio=("advance", "mean"),
            above_ma20_ratio=("above_ma20", "mean"),
        )
        for column in ("median_ret5", "advance_ratio", "above_ma20_ratio"):
            records[f"{column}_pctile"] = records.groupby("date")[column].rank(
                pct=True, method="average"
            ) * 100.0
        records["trend_score"] = (
            records["median_ret5_pctile"] * 0.50
            + records["advance_ratio_pctile"] * 0.25
            + records["above_ma20_ratio_pctile"] * 0.25
        )
        records["rank"] = records.groupby("date")["trend_score"].rank(
            ascending=False, method="min"
        ).astype(int)
        output: dict[str, dict[str, Any]] = {}
        universe_counts = dict(connection.execute(
            """SELECT date,COUNT(*) FROM breadth_universe
            WHERE date BETWEEN ? AND ? GROUP BY date""",
            (start, end),
        ).fetchall())
        member_counts = dict(connection.execute(
            """SELECT date,COUNT(*) FROM industry_membership
            WHERE date BETWEEN ? AND ? GROUP BY date""",
            (start, end),
        ).fetchall())
        progress = dict(connection.execute(
            """SELECT date,status FROM industry_progress
            WHERE date BETWEEN ? AND ?""",
            (start, end),
        ).fetchall())
        for date, daily in records.groupby("date"):
            expected = int(universe_counts.get(date, 0))
            mapped = int(member_counts.get(date, 0))
            coverage = mapped / expected if expected else 0.0
            industries: dict[str, Any] = {}
            total_industries = len(daily)
            for row in daily.itertuples(index=False):
                valid = bool(row.member_count >= 5 and row.observed_count / row.member_count >= 0.95)
                strong = bool(valid and row.trend_score >= 70.0 and row.median_ret5 > 0)
                weak = bool(valid and (row.trend_score < 35.0 or row.median_ret5 < -2.0))
                industries[row.industry] = {
                    "member_count": int(row.member_count),
                    "observed_count": int(row.observed_count),
                    "median_5d_return": round(float(row.median_ret5), 6),
                    "median_day_return": round(float(row.median_day_return), 6),
                    "advance_ratio": round(float(row.advance_ratio), 6),
                    "above_ma20_ratio": round(float(row.above_ma20_ratio), 6),
                    "trend_score": round(float(row.trend_score), 2),
                    "rank": int(row.rank),
                    "total_industries": total_industries,
                    "trend_status": "strong" if strong else ("weak" if weak else "neutral"),
                    "status_known": valid,
                }
            exact = bool(progress.get(date) == "ok" and coverage >= 0.95)
            output[str(date)] = {
                "date": str(date),
                "industries": industries,
                "data_quality": {
                    "industry_context_exact": exact,
                    "exact_date_membership": True,
                    "membership_coverage": round(coverage, 6),
                    "mapped_stocks": mapped,
                    "expected_trading_stocks": expected,
                    "causal": True,
                    "source": "baostock.query_stock_industry(date)",
                },
            }
        return output

    def _save(self, date: str, payload: Mapping[str, Any]) -> None:
        self._write_json(self.root / f"{date}.json", payload)

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)


class IndustryTrendReader:
    def __init__(
        self,
        root: str | Path = "data/industry_trend",
        *,
        breadth_db: str | Path = "data/market_breadth/raw.sqlite3",
    ) -> None:
        self.root = Path(root)
        self.breadth_db = Path(breadth_db)
        self._payload_cache: dict[str, dict[str, Any]] = {}
        self._membership_cache: dict[tuple[str, str], str] = {}

    def context(self, code: str, date: str, *, allow_previous: bool) -> dict[str, Any]:
        source_date = self._resolve_date(date, allow_previous=allow_previous)
        if not source_date:
            return self._empty()
        payload = self._payload(source_date)
        quality = payload.get("data_quality") or {}
        provider_code = f"sh.{code}" if str(code).startswith("6") else f"sz.{code}"
        industry = self._industry(provider_code, source_date)
        metrics = (payload.get("industries") or {}).get(industry) or {}
        known = bool(
            quality.get("industry_context_exact")
            and industry
            and metrics.get("status_known")
        )
        return {
            "theme_status_known": known,
            "theme_context_date": source_date,
            "theme": industry,
            "industry": industry,
            "industry_trend": metrics,
            "theme_invalidated": bool(known and metrics.get("trend_status") == "weak"),
            "theme_trend_strong": bool(known and metrics.get("trend_status") == "strong"),
        }

    def _resolve_date(self, date: str, *, allow_previous: bool) -> str:
        exact = self.root / f"{date}.json"
        if exact.exists():
            return date
        if not allow_previous:
            return ""
        available = sorted(path.stem for path in self.root.glob("20*.json"))
        position = bisect.bisect_right(available, date) - 1
        if position < 0:
            return ""
        candidate = available[position]
        gap = (
            datetime.strptime(date, "%Y%m%d").date()
            - datetime.strptime(candidate, "%Y%m%d").date()
        ).days
        return candidate if gap <= 3 else ""

    def _payload(self, date: str) -> dict[str, Any]:
        if date not in self._payload_cache:
            try:
                self._payload_cache[date] = json.loads(
                    (self.root / f"{date}.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError):
                self._payload_cache[date] = {}
        return self._payload_cache[date]

    def _industry(self, provider_code: str, date: str) -> str:
        key = (date, provider_code)
        if key not in self._membership_cache:
            try:
                with sqlite3.connect(
                    f"file:{self.breadth_db.as_posix()}?mode=ro", uri=True
                ) as connection:
                    row = connection.execute(
                        "SELECT industry FROM industry_membership WHERE date=? AND code=?",
                        (date, provider_code),
                    ).fetchone()
                self._membership_cache[key] = str(row[0]) if row else ""
            except (OSError, sqlite3.Error):
                self._membership_cache[key] = ""
        return self._membership_cache[key]

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "theme_status_known": False,
            "theme_context_date": "",
            "theme": "",
            "industry": "",
            "industry_trend": {},
            "theme_invalidated": False,
            "theme_trend_strong": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="回填精确历史行业成员和短期行业趋势")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--root", default="data/industry_trend")
    parser.add_argument("--breadth-db", default="data/market_breadth/raw.sqlite3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for value in (args.start, args.end):
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as exc:
            parser.error(str(exc))
    report = BaoStockIndustryTrendArchive(
        args.root, breadth_db=args.breadth_db
    ).build(args.start, args.end, overwrite=args.overwrite)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
