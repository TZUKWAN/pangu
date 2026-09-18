"""Historical market-breadth archive tests."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from engine.market_breadth_archive import ArchivedBreadthKlineLoader, BaoStockBreadthArchive


def _frame() -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    dates = pd.bdate_range("2026-06-01", periods=25).strftime("%Y%m%d").tolist()
    rows = []
    universe = {}
    for index, date in enumerate(dates):
        universe[date] = {"sh.600001": "甲", "sz.300001": "乙"}
        rows.extend([
            {
                "date": date, "code": "sh.600001", "close": 10 + index * 0.1,
                "volume": 1000, "amount": 10000, "pct_change": 1.0, "is_st": 0,
            },
            {
                "date": date, "code": "sz.300001", "close": 20 - index * 0.05,
                "volume": 900, "amount": 9000, "pct_change": -0.25, "is_st": 0,
            },
        ])
    return pd.DataFrame(rows), universe


def test_a_share_filter_excludes_indices_b_shares_and_beijing() -> None:
    assert BaoStockBreadthArchive._is_a_share("sh.600000")
    assert BaoStockBreadthArchive._is_a_share("sh.688001")
    assert BaoStockBreadthArchive._is_a_share("sz.000001")
    assert BaoStockBreadthArchive._is_a_share("sz.300001")
    assert not BaoStockBreadthArchive._is_a_share("sh.000001")
    assert not BaoStockBreadthArchive._is_a_share("sh.900901")
    assert not BaoStockBreadthArchive._is_a_share("sz.200001")
    assert not BaoStockBreadthArchive._is_a_share("bj.430001")


def test_result_rows_avoids_baostock_broken_get_data() -> None:
    class Result:
        def __init__(self) -> None:
            self.rows = [["a"], ["b"]]
            self.index = -1

        def next(self) -> bool:
            self.index += 1
            return self.index < len(self.rows)

        def get_row_data(self) -> list[str]:
            return self.rows[self.index]

    assert BaoStockBreadthArchive._result_rows(Result()) == [["a"], ["b"]]


def test_aggregate_is_exact_date_and_has_no_future_leakage() -> None:
    frame, universe = _frame()
    target = sorted(universe)[20]
    baseline = BaoStockBreadthArchive.aggregate(
        frame, universe, start=target, end=target
    )[target]
    changed = frame.copy()
    changed.loc[changed["date"] > target, "pct_change"] = 99.0
    after = BaoStockBreadthArchive.aggregate(
        changed, universe, start=target, end=target
    )[target]
    assert baseline == after
    assert baseline["data_quality"]["market_context_exact"] is True
    assert baseline["breadth"]["advance"] == 1
    assert baseline["breadth"]["decline"] == 1
    assert baseline["components"]["above_ma20_ratio"]["raw"] == 0.5


def test_limit_threshold_respects_st_and_twenty_percent_boards() -> None:
    assert BaoStockBreadthArchive._limit_threshold(pd.Series({"code": "sh.600001", "is_st": 1})) == 4.8
    assert BaoStockBreadthArchive._limit_threshold(pd.Series({"code": "sh.688001", "is_st": 0})) == 19.5
    assert BaoStockBreadthArchive._limit_threshold(pd.Series({"code": "sz.300001", "is_st": 0})) == 19.5
    assert BaoStockBreadthArchive._limit_threshold(pd.Series({"code": "sz.000001", "is_st": 0})) == 9.5


def test_cached_universe_reuses_only_complete_dates() -> None:
    connection = sqlite3.connect(":memory:")
    BaoStockBreadthArchive._init_db(connection)
    complete = {f"sh.{600000 + index}": f"股票{index}" for index in range(1000)}
    partial = {"sh.600001": "一只"}
    BaoStockBreadthArchive._persist_universe(
        connection, {"20260701": complete, "20260702": partial}
    )
    cached = BaoStockBreadthArchive._cached_universe(
        connection, ["20260701", "20260702", "20260703"]
    )
    assert set(cached) == {"20260701"}
    assert len(cached["20260701"]) == 1000


def test_load_universe_persists_each_completed_date_before_later_failure() -> None:
    class Result:
        error_code = "0"
        error_msg = ""
        fields = ["code", "tradeStatus", "code_name"]

        def __init__(self, rows: list[list[str]]) -> None:
            self.rows = rows
            self.index = -1

        def next(self) -> bool:
            self.index += 1
            return self.index < len(self.rows)

        def get_row_data(self) -> list[str]:
            return self.rows[self.index]

    rows = [
        [f"sh.{600000 + index}", "1", f"stock-{index}"]
        for index in range(1000)
    ]

    class BS:
        def query_all_stock(self, *, day: str) -> Result:
            if day == "2026-07-02":
                raise RuntimeError("provider interrupted")
            return Result(rows)

    connection = sqlite3.connect(":memory:")
    BaoStockBreadthArchive._init_db(connection)
    archive = BaoStockBreadthArchive()

    with pytest.raises(RuntimeError, match="provider interrupted"):
        archive._load_universe(
            BS(), ["20260701", "20260702"], connection=connection
        )

    assert connection.execute(
        "SELECT COUNT(*) FROM breadth_universe WHERE date='20260701'"
    ).fetchone()[0] == 1000
    assert connection.execute(
        "SELECT COUNT(*) FROM breadth_universe WHERE date='20260702'"
    ).fetchone()[0] == 0


def test_audit_allows_first_observation_ipo_extreme_and_loader_reads_ohlcv(tmp_path) -> None:
    root = tmp_path / "breadth"
    archive = BaoStockBreadthArchive(root)
    connection = sqlite3.connect(archive.db_path)
    BaoStockBreadthArchive._init_db(connection)
    connection.execute(
        "INSERT INTO breadth_universe(date,code,name) VALUES(?,?,?)",
        ("20260701", "sh.688001", "新股"),
    )
    connection.execute(
        """INSERT INTO breadth_raw
        (date,code,open,high,low,close,preclose,volume,amount,turnover,pct_change,is_st)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("20260701", "sh.688001", 100, 120, 90, 110, 9.1667, 1000, 100000, 1.2, 1100, 0),
    )
    connection.execute(
        "INSERT INTO breadth_progress(code,status,error,updated_at) VALUES(?,?,?,?)",
        ("sh.688001", "ok", "", "2026-07-01T00:00:00+00:00"),
    )
    connection.commit()
    connection.close()

    report = archive.audit("20260701", "20260701")
    assert report["status"] == "ready"
    assert report["metrics"]["extreme_first_observation_rows"] == 1
    loader = ArchivedBreadthKlineLoader(archive.db_path)
    frame = loader.daily_kline("688001", days=10, date="20260701")
    assert frame.iloc[0]["开盘"] == 100
    assert frame.iloc[0]["最高"] == 120
    assert frame.iloc[0]["最低"] == 90
    assert frame.iloc[0]["收盘"] == 110


def test_progress_schema_migration_backfills_covered_range() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE breadth_raw (
            date TEXT NOT NULL, code TEXT NOT NULL, close REAL, volume REAL,
            amount REAL, pct_change REAL, is_st INTEGER, PRIMARY KEY(date,code)
        );
        CREATE TABLE breadth_progress (
            code TEXT PRIMARY KEY, status TEXT NOT NULL, error TEXT, updated_at TEXT NOT NULL
        );
        CREATE TABLE breadth_universe (
            date TEXT NOT NULL, code TEXT NOT NULL, name TEXT, PRIMARY KEY(date,code)
        );
        INSERT INTO breadth_raw VALUES('20260410','sh.600000',10,100,1000,1,0);
        INSERT INTO breadth_raw VALUES('20260724','sh.600000',11,100,1000,1,0);
        INSERT INTO breadth_progress VALUES('sh.600000','ok','','2026-07-24T00:00:00Z');
        """
    )
    BaoStockBreadthArchive._init_db(connection)
    row = connection.execute(
        "SELECT covered_start,covered_end FROM breadth_progress WHERE code='sh.600000'"
    ).fetchone()
    assert row == ("20260410", "20260724")
