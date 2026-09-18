"""回放基础设施测试：PIT-safe 数据面 + 档案 RPS 直算。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from engine.replay_loader import ReplayDataLoader, _limit_price, _normalize_code, _price_limit_pct


@pytest.fixture(scope="module")
def archive_db(tmp_path_factory) -> Path:
    """构造小型回放档案：3 只股票 × 若干交易日。"""
    path = tmp_path_factory.mktemp("breadth") / "raw.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE breadth_raw (
               date TEXT, code TEXT, close REAL, volume REAL, amount REAL,
               pct_change REAL, is_st INTEGER, open REAL, high REAL, low REAL,
               preclose REAL, turnover REAL,
               PRIMARY KEY (date, code))"""
    )
    conn.execute("CREATE TABLE breadth_universe (date TEXT, code TEXT, name TEXT)")
    conn.execute(
        "CREATE TABLE industry_membership (date TEXT, code TEXT, industry TEXT,"
        " classification TEXT, update_date TEXT)"
    )

    dates = ["20260601", "20260602", "20260603", "20260604", "20260605"]
    # 000001 平稳股；000002 连续涨停（10% 板）；000003 ST 股 5% 板
    rows = []
    close_000001 = 10.0
    close_000002 = 20.0
    close_000003 = 5.0
    for d in dates:
        rows.append((d, "sh.000001", close_000001, 1_000, 10_000, 0.0, 0,
                     close_000001, close_000001, close_000001, close_000001, 1.0))
        pre_000002 = close_000002
        close_000002 = round(pre_000002 * 1.1, 2)
        rows.append((d, "sz.000002", close_000002, 2_000, 40_000,
                     round((close_000002 / pre_000002 - 1) * 100, 4), 0,
                     close_000002, close_000002, pre_000002, pre_000002, 2.0))
        pre_000003 = close_000003
        close_000003 = round(pre_000003 * 1.05, 2)
        rows.append((d, "sz.000003", close_000003, 3_000, 15_000,
                     round((close_000003 / pre_000003 - 1) * 100, 4), 1,
                     close_000003, close_000003, pre_000003, pre_000003, 3.0))
    conn.executemany("INSERT INTO breadth_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    for d in dates:
        conn.execute("INSERT INTO breadth_universe VALUES (?,?,?)", (d, "000001", "平安测试"))
        conn.execute("INSERT INTO breadth_universe VALUES (?,?,?)", (d, "000002", "连板测试"))
        conn.execute("INSERT INTO breadth_universe VALUES (?,?,?)", (d, "000003", "ST测试"))
        conn.execute(
            "INSERT INTO industry_membership VALUES (?,?,?,?,?)",
            (d, "000002", "C39测试行业", "SW", d),
        )
    conn.commit()
    conn.close()
    return path


def test_normalize_code():
    assert _normalize_code("sh.600000") == "600000"
    assert _normalize_code("sz.000001") == "000001"
    assert _normalize_code("600000") == "600000"
    assert _normalize_code(1) == "000001"


def test_price_limit_rules():
    assert _price_limit_pct("600519", False) == 0.10
    assert _price_limit_pct("300750", False) == 0.20
    assert _price_limit_pct("688981", False) == 0.20
    assert _price_limit_pct("830799", False) == 0.30
    assert _price_limit_pct("600519", True) == 0.05
    assert _limit_price(10.00, 0.10) == 11.00


def test_loader_spot_and_pit_kline(archive_db):
    dl = ReplayDataLoader(db_path=archive_db, preload=True)
    dl.set_date("20260603")
    spot = dl.all_spot()
    assert len(spot) == 3
    row = spot[spot["代码"] == "000002"].iloc[0]
    assert row["名称"] == "连板测试"

    k = dl.daily_kline("000002", days=10, date="20260603")
    # PIT 截断：只包含 <= 20260603
    assert list(k["date"]) == ["20260601", "20260602", "20260603"]


def test_limit_up_pool_synthesis(archive_db):
    dl = ReplayDataLoader(db_path=archive_db)
    lu = dl.limit_up_pool("20260605")
    codes = set(lu["代码"])
    # 000002 五连板、000003 ST 五连板都应识别为涨停
    assert "000002" in codes and "000003" in codes
    consec = dict(zip(lu["代码"], lu["连板数"]))
    assert consec["000002"] == 5
    assert consec["000003"] == 5
    # 平稳股不在涨停池
    assert "000001" not in codes
    # 行业成员来自档案
    assert (lu[lu["代码"] == "000002"]["所属行业"] == "C39测试行业").all()


def test_limit_down_pool_empty_on_rally(archive_db):
    dl = ReplayDataLoader(db_path=archive_db)
    assert dl.limit_down_pool("20260605").empty


def test_trading_days_and_prev(archive_db):
    dl = ReplayDataLoader(db_path=archive_db)
    assert dl.trading_days("20260602", "20260604") == ["20260602", "20260603", "20260604"]
    assert dl.prev_trading_day("20260603") == "20260602"


def test_announcement_events_hook(tmp_path):
    dl = ReplayDataLoader(db_path=str(tmp_path / "none.sqlite3")) if False else None
    # 公告档案钩子：目录不存在时返回空
    loader = ReplayDataLoader.__new__(ReplayDataLoader)
    loader.announcement_dir = tmp_path
    loader._trade_dates = ["20260601"]
    assert loader.announcement_events("000002", "20260601") == []
