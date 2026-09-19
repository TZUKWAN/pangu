"""PIT 数据底座测试：全部使用 tmp_path 合成 sqlite 夹具，不触真实档案/网络。

夹具：10 只股票 × 30 个交易日（2026-01-05 起），覆盖：
- ST 翻转（000003 第 15 天起变 ST，名称同步变化）
- 停牌缺口（000004 第 10-14 天无 bar 但仍在宇宙）
- 除权日（600001 第 15 天 close 减半而 pct_change 不匹配）
- 晚上市（600005 第 20 天起才有 bar/成员）
- 行业数据仅 2026-01-28 起（第 17 天起）可用
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from engine.data.corporate_actions import detect_ex_right_candidates, persist_corporate_actions
from engine.data.pit_store import BAR_COLUMNS, PITStore

N_DAYS = 30
DATES = []
_d = date(2026, 1, 5)
while len(DATES) < N_DAYS:
    if _d.weekday() < 5:
        DATES.append(_d.strftime("%Y%m%d"))
    _d += timedelta(days=1)

BASE_CODES = ["000001", "000002", "000003", "000004", "000005",
              "600001", "600002", "600003", "600004"]  # 600005 第 20 天才上市
ST_FLIP_DAY = 15
SUSPEND_DAYS = set(range(10, 15))          # 000004 停牌
EX_RIGHT_DAY = 15                          # 600001 除权
LATE_LIST_DAY = 20                         # 600005 上市
INDUSTRY_FROM = 17                         # DATES[17] == '20260128'


def build_archive(db_path: Path) -> int:
    """构造合成档案，返回插入的日线行数（供行数断言）。"""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE breadth_raw (
               date TEXT NOT NULL, code TEXT NOT NULL,
               open REAL, high REAL, low REAL, close REAL, preclose REAL,
               volume REAL, amount REAL, pct_change REAL, turnover REAL,
               is_st INTEGER NOT NULL DEFAULT 0,
               PRIMARY KEY (date, code))"""
    )
    conn.execute("CREATE TABLE breadth_universe (date TEXT, code TEXT, name TEXT)")
    conn.execute(
        "CREATE TABLE industry_membership (date TEXT, code TEXT, industry TEXT,"
        " classification TEXT, update_date TEXT)"
    )
    conn.execute(
        "CREATE TABLE index_daily (date TEXT, code TEXT, open REAL, high REAL,"
        " low REAL, close REAL, preclose REAL, volume REAL, amount REAL, pct_change REAL,"
        " PRIMARY KEY (date, code))"
    )

    bar_rows = []
    uni_rows = []
    for i, d in enumerate(DATES):
        for n, code in enumerate(BASE_CODES):
            if code == "000004" and i in SUSPEND_DAYS:
                pass  # 停牌：无 bar，但宇宙成员保留
            else:
                prev = 10.0 + n
                r = [-1.0, 0.5, 1.0][i % 3]
                close = round(prev * (1 + r / 100), 2) if i else prev
                if i == 0:
                    close = prev
                    pct = 0.0
                elif code == "600001" and i == EX_RIGHT_DAY:
                    close = round(prev * 0.5, 2)   # 10送10：价格腰斩
                    pct = 0.5                      # 调整后收益与价格隐含收益不一致
                else:
                    pct = round((close / prev - 1) * 100, 4)
                bar_rows.append((d, code, close, close, close, close, prev,
                                 1000.0 + n, 10000.0 + n, pct, 1.0 + n / 10,
                                 1 if code == "000003" and i >= ST_FLIP_DAY else 0))
            if code == "000003":
                name = "样本三" if i < ST_FLIP_DAY else "*ST样本三"
            else:
                name = f"样本{n + 1}"
            uni_rows.append((d, code, name))
        if i >= LATE_LIST_DAY:
            bar_rows.append((d, "600005", 20.0, 20.0, 20.0, 20.0, 20.0,
                             9999.0, 99990.0, 0.0, 9.9, 0))
            uni_rows.append((d, "600005", "晚上市"))
        if i >= INDUSTRY_FROM:
            for code in BASE_CODES:
                if code == "000005":
                    continue  # 有成员无行业 → 该票 degraded
                conn.execute(
                    "INSERT INTO industry_membership VALUES (?,?,?,?,?)",
                    (d, code, f"行业{code[-1]}", "SW", d),
                )
    conn.executemany("INSERT INTO breadth_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", bar_rows)
    conn.executemany("INSERT INTO breadth_universe VALUES (?,?,?)", uni_rows)
    for i, d in enumerate(DATES):
        close = 4000.0 + i
        conn.execute(
            "INSERT INTO index_daily VALUES (?,?,?,?,?,?,?,?,?,?)",
            (d, "000300", close, close, close, close, close - 1, 1e9, 1e10,
             round((close / (close - 1) - 1) * 100, 4)),
        )
    conn.commit()
    conn.close()
    return len(bar_rows)


@pytest.fixture()
def store(tmp_path) -> PITStore:
    db = tmp_path / "raw.sqlite3"
    build_archive(db)
    return PITStore(
        db_path=db,
        announcement_dir=tmp_path / "announcement_archive",
        news_dir=tmp_path / "wscn_news_archive",
        pit_dir=tmp_path / "pit",
    )


# ---------------------------------------------------------------------- #
# 日期归一化
# ---------------------------------------------------------------------- #
def test_date_normalization_and_trading_days(store):
    days = store.trading_days("2026-01-05", "20260109")
    assert days == ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    assert store.next_trading_day("2026-01-09") == "2026-01-12"
    assert store.prev_trading_day("2026-02-02") == "2026-01-30"
    assert store.next_trading_day("20260213") is None
    assert store.prev_trading_day("2025-12-31") is None


# ---------------------------------------------------------------------- #
# 日线面板 + asof 守卫
# ---------------------------------------------------------------------- #
def test_daily_panel_strict_asof_iso_index_and_symbols(store):
    panel = store.daily_panel("2026-01-05", "2026-01-20")
    assert panel.index.names == ["date", "code"]
    idx_dates = list(panel.index.get_level_values("date").unique())
    assert all("-" in d for d in idx_dates)                       # ISO 输出
    assert max(idx_dates) <= "2026-01-20"                          # 严格 asof
    assert list(panel.columns) == BAR_COLUMNS
    single = store.daily_panel("2026-01-05", "2026-01-20", symbols=["000001"])
    assert set(single.index.get_level_values("code")) == {"000001"}
    # 停牌缺口：000004 在 2026-01-05..2026-01-30（20 个交易日）里缺 5 天
    k4 = store.daily_panel("2026-01-05", "2026-01-30", symbols=["000004"])
    assert len(k4) == 20 - len(SUSPEND_DAYS)


def test_guard_asof_raises_on_future_rows(store):
    future = pd.DataFrame({"date": ["2026-01-05", "2026-01-21"], "close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="PIT guard"):
        store._guard_asof(future, "2026-01-20")
    with pytest.raises(ValueError):
        store._guard_asof(future, "20260120")                      # 紧凑格式同样拦截
    ok = pd.DataFrame({"date": ["2026-01-05", "20260120"], "close": [1.0, 2.0]})
    store._guard_asof(ok, "2026-01-20")                            # 不抛
    # MultiIndex date 级同样受保护
    mi = future.set_index(["date", "close"]).sort_index()
    with pytest.raises(ValueError):
        store._guard_asof(mi, "2026-01-20")


# ---------------------------------------------------------------------- #
# PIT 宇宙
# ---------------------------------------------------------------------- #
def test_universe_reflects_historical_st_state(store):
    early = store.universe("2026-01-09")
    late = store.universe("2026-02-02")
    assert not bool(early.loc["000003", "is_st"])
    assert early.loc["000003", "name"] == "样本三"
    assert bool(late.loc["000003", "is_st"])
    assert late.loc["000003", "name"] == "*ST样本三"
    # 其它票始终非 ST
    assert not bool(late.loc["000001", "is_st"])


def test_universe_detects_suspension(store):
    suspended_day = store.universe(DATES[12])
    assert bool(suspended_day.loc["000004", "suspended"])
    assert not bool(suspended_day.loc["000004", "tradable"])
    normal_day = store.universe(DATES[8])
    assert not bool(normal_day.loc["000004", "suspended"])
    assert bool(normal_day.loc["000004", "tradable"])


def test_universe_industry_availability_degrades_before_coverage(store):
    before = store.universe("2026-01-20")
    assert pd.isna(before.loc["000001", "industry"])
    assert (before["availability"] == "degraded").all()
    after = store.universe("2026-02-02")
    assert after.loc["000001", "industry"] == "行业1"
    assert after.loc["000001", "availability"] == "ok"
    # 000005 在行业覆盖期内也缺失 → 单票 degraded
    assert after.loc["000005", "availability"] == "degraded"
    assert pd.isna(after.loc["000005", "industry"])


def test_universe_listed_days_pit_and_late_listing(store):
    day0 = store.universe(DATES[0])
    assert int(day0.loc["000001", "listed_days"]) == 1
    assert "600005" not in store.universe(DATES[10]).index     # 未上市：不在成员表
    late = store.universe(DATES[25])
    assert int(late.loc["600005", "listed_days"]) == 6          # 第 20 天上市，含当日
    assert int(late.loc["000001", "listed_days"]) == 26


def test_index_daily_iso_and_asof(store):
    df = store.index_daily("000300", "2026-01-05", "2026-01-15")
    assert list(df.index) == [
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
        "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15",
    ]
    assert (df["close"].diff().dropna() == 1.0).all()
    short = store.index_daily("000300", "2026-01-05", "20260109")
    assert len(short) == 5


# ---------------------------------------------------------------------- #
# 除权候选（诊断用途）
# ---------------------------------------------------------------------- #
def test_corporate_actions_detected_and_persisted(tmp_path):
    db = tmp_path / "raw.sqlite3"
    build_archive(db)
    hits = detect_ex_right_candidates(db)
    assert len(hits) == 1
    row = hits.iloc[0]
    assert row["date"] == DATES[EX_RIGHT_DAY]
    assert row["code"] == "600001"
    assert row["kind"] == "ex_right"
    assert 0 < float(row["ratio_implied"]) < 3

    out = tmp_path / "pit" / "corporate_actions.sqlite"
    persist_corporate_actions(db, out_path=out)
    assert out.exists()
    conn = sqlite3.connect(out)
    rows = conn.execute("SELECT date, code, kind FROM corporate_actions").fetchall()
    conn.close()
    assert rows == [(DATES[EX_RIGHT_DAY], "600001", "ex_right")]


# ---------------------------------------------------------------------- #
# 公告 / 新闻 availability
# ---------------------------------------------------------------------- #
def test_announcements_available_next_trading_day(tmp_path, store):
    ann_dir = store.announcement_dir
    ann_dir.mkdir(parents=True)
    (ann_dir / f"{DATES[17]}.json").write_text(
        '{"events": [{"code": "000001", "name": "样本1", "title": "回购公告",'
        ' "published_at": "2026-01-28T00:00:00+08:00"}]}',
        encoding="utf-8",
    )
    df = store.announcements((DATES[17], DATES[19]))
    assert len(df) == 1
    row = df.iloc[0]
    assert row["code"] == "000001"
    assert row["published_at"] == "2026-01-28"
    # 日期精度未知盘中/盘后 → 保守：下一交易日才可用于决策
    assert row["available_at_decision"] == "2026-01-29"
    assert store.announcements((DATES[17], DATES[19]), codes=["600001"]).empty


def test_news_cutoff_at_decision_time(tmp_path, store):
    news_dir = store.news_dir
    news_dir.mkdir(parents=True)
    d = DATES[18]  # 2026-01-29
    # naive datetime → 本地时区 epoch，与实现侧 datetime.fromtimestamp 自洽
    early = int(datetime(2026, 1, 29, 10, 0, 0).timestamp())
    late = int(datetime(2026, 1, 29, 16, 30, 0).timestamp())
    (news_dir / f"{d}.json").write_text(
        '{"items": ['
        '{"id": "1", "display_timestamp": %d, "title": "盘中快讯", "stock_codes": ["000001"]},'
        '{"id": "2", "display_timestamp": %d, "title": "盘后快讯", "stock_codes": ["000001"]}'
        "]}" % (early, late),
        encoding="utf-8",
    )
    df = store.news((d, d))
    avail = dict(zip(df["id"], df["available_at_decision"]))
    assert avail["1"] == "2026-01-29"        # <= 15:05 决策时刻 → 当日可用
    assert avail["2"] == "2026-01-30"        # > 15:05 → 顺延下一交易日


def test_missing_event_dirs_return_empty_frames(tmp_path):
    db = tmp_path / "raw.sqlite3"
    build_archive(db)
    bare = PITStore(
        db_path=db,
        announcement_dir=tmp_path / "no_ann",
        news_dir=tmp_path / "no_news",
        pit_dir=tmp_path / "pit",
    )
    assert bare.announcements((DATES[0], DATES[5])).empty
    assert bare.news((DATES[0], DATES[5])).empty


# ---------------------------------------------------------------------- #
# 快照 manifest
# ---------------------------------------------------------------------- #
def test_snapshot_manifest_written_with_counts_and_hashes(tmp_path):
    db = tmp_path / "raw.sqlite3"
    n_bars = build_archive(db)
    st = PITStore(
        db_path=db,
        announcement_dir=tmp_path / "no_ann",
        news_dir=tmp_path / "no_news",
        pit_dir=tmp_path / "pit",
    )
    out = tmp_path / "manifests"
    payload = st.snapshot_manifest(DATES[-1], out_dir=out)
    path = out / f"{DATES[-1]}.json"
    assert path.exists()
    assert payload["row_counts"]["breadth_raw"] == n_bars
    # 停牌日没有 bar，但仍是宇宙成员（每天 9 只基础成员 + 600005 上市后 10 天）
    assert payload["row_counts"]["breadth_universe"] == n_bars + len(SUSPEND_DAYS)
    assert payload["row_counts"]["index_daily"] == N_DAYS
    git = payload["git_commit"]
    assert git == "unknown" or (len(git) == 40 and all(c in "0123456789abcdef" for c in git))
    settings_sha = payload["settings_sha256"]
    if settings_sha is not None:
        assert settings_sha == hashlib.sha256(st.settings_path.read_bytes()).hexdigest()
