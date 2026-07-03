"""RPS 预计算模块测试。"""

import pandas as pd
import pytest

from engine import rps as rps_mod


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_rps.db")


class StubDL:
    """内存 DataLoader，返回预设 K 线。"""

    def __init__(self, kline_fn):
        self._fn = kline_fn

    def all_spot(self):
        return pd.DataFrame({"代码": ["000001", "000002", "000003", "000004"]})

    def daily_kline(self, code, days=25, adjust="qfq"):
        return self._fn(code)


def make_kline(prices):
    """从价格序列造 K 线。"""
    return pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=len(prices)).astype(str),
        "收盘": prices,
        "成交量": [10000] * len(prices),
    })


def test_ensure_table_creates(tmp_db):
    rps_mod.ensure_table(tmp_db)
    import sqlite3
    with sqlite3.connect(tmp_db) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "rps" in tables


def test_compute_and_load(tmp_db):
    """计算 RPS 后能查表，且百分位分布正确。"""
    # 4 只票，涨幅依次递增
    dl = StubDL(lambda c: {
        "000001": make_kline([10] * 21 + [12]),     # +20%
        "000002": make_kline([10] * 21 + [11]),     # +10%
        "000003": make_kline([10] * 21 + [10.5]),   # +5%
        "000004": make_kline([10] * 21 + [10]),     # 0%
    }[c])
    res = rps_mod.compute_all_rps(dl, date="20240101", db_path=tmp_db, workers=2)
    assert res["ok"] == 4
    m = rps_mod.load_rps_map("20240101", tmp_db)
    assert len(m) == 4
    # 涨幅最大的 000001 应该 RPS 最高（=100），最小的 000004 最低
    assert m["000001"] == 100.0
    assert m["000004"] < m["000001"]
    assert rps_mod.is_available("20240101", tmp_db)


def test_load_empty_when_no_table(tmp_db):
    """表不存在时返回空 dict。"""
    assert rps_mod.load_rps_map("20240101", tmp_db) == {}
    assert not rps_mod.is_available("20240101", tmp_db)


def test_rps_overwrite_same_date(tmp_db):
    """同日重算应覆盖旧数据。"""
    dl = StubDL(lambda c: make_kline([10] * 21 + [11]))
    rps_mod.compute_all_rps(dl, date="20240101", db_path=tmp_db, workers=2)
    # 再算一次，数据应被覆盖而非重复
    rps_mod.compute_all_rps(dl, date="20240101", db_path=tmp_db, workers=2)
    import sqlite3
    with sqlite3.connect(tmp_db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM rps WHERE date='20240101'").fetchone()[0]
    assert n == 4  # 仍是 4 只，没有重复
