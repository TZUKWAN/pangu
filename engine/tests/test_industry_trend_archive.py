"""Exact-date industry trend archive tests."""

from __future__ import annotations

import json
import sqlite3

import pandas as pd

from engine.industry_trend_archive import BaoStockIndustryTrendArchive, IndustryTrendReader
from engine.market_breadth_archive import BaoStockBreadthArchive


def _build_db(path) -> tuple[list[str], list[str], list[str]]:
    connection = sqlite3.connect(path)
    BaoStockBreadthArchive._init_db(connection)
    BaoStockIndustryTrendArchive._init_db(connection)
    dates = pd.bdate_range("2026-06-01", periods=25).strftime("%Y%m%d").tolist()
    strong_codes = [f"sh.{600100 + index}" for index in range(6)]
    weak_codes = [f"sz.{300100 + index}" for index in range(6)]
    for date in dates:
        for code in strong_codes + weak_codes:
            pct = 1.0 if code in strong_codes else -1.0
            connection.execute(
                "INSERT INTO breadth_universe(date,code,name) VALUES(?,?,?)",
                (date, code, code),
            )
            connection.execute(
                """INSERT INTO breadth_raw
                (date,code,open,high,low,close,preclose,volume,amount,turnover,pct_change,is_st)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (date, code, 10, 10.2, 9.8, 10.1, 10, 1000, 10000, 1, pct, 0),
            )
            connection.execute(
                """INSERT INTO industry_membership
                (date,code,industry,classification,update_date) VALUES(?,?,?,?,?)""",
                (date, code, "强行业" if code in strong_codes else "弱行业", "证监会", date),
            )
        connection.execute(
            "INSERT INTO industry_progress VALUES(?,?,?,?,?)",
            (date, "ok", 12, "", "2026-07-01T00:00:00+00:00"),
        )
    connection.commit()
    connection.close()
    return dates, strong_codes, weak_codes


def test_industry_trend_is_causal_and_classifies_strong_vs_weak(tmp_path) -> None:
    db = tmp_path / "bars.sqlite3"
    dates, _, _ = _build_db(db)
    archive = BaoStockIndustryTrendArchive(tmp_path / "trend", breadth_db=db)
    target = dates[20]
    connection = sqlite3.connect(db)
    baseline = archive._aggregate(connection, target, target)[target]
    connection.execute(
        "UPDATE breadth_raw SET pct_change=99 WHERE date>?", (target,)
    )
    connection.commit()
    after = archive._aggregate(connection, target, target)[target]
    connection.close()
    assert baseline == after
    assert baseline["data_quality"]["industry_context_exact"] is True
    assert baseline["industries"]["强行业"]["trend_status"] == "strong"
    assert baseline["industries"]["弱行业"]["trend_status"] == "weak"


def test_reader_uses_previous_trading_day_for_weekend_signal(tmp_path) -> None:
    db = tmp_path / "bars.sqlite3"
    _, strong_codes, _ = _build_db(db)
    root = tmp_path / "trend"
    root.mkdir()
    (root / "20260703.json").write_text(json.dumps({
        "date": "20260703",
        "industries": {
            "强行业": {
                "status_known": True,
                "trend_status": "strong",
                "trend_score": 88,
            }
        },
        "data_quality": {"industry_context_exact": True},
    }), encoding="utf-8")
    connection = sqlite3.connect(db)
    connection.execute(
        "INSERT OR REPLACE INTO industry_membership VALUES(?,?,?,?,?)",
        ("20260703", strong_codes[0], "强行业", "证监会", "20260703"),
    )
    connection.commit()
    connection.close()
    reader = IndustryTrendReader(root, breadth_db=db)
    context = reader.context(strong_codes[0][3:], "20260704", allow_previous=True)
    assert context["theme_status_known"] is True
    assert context["theme_context_date"] == "20260703"
    assert context["theme_trend_strong"] is True
