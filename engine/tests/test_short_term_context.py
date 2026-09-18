"""Exact-date context archive tests."""

from __future__ import annotations

import pandas as pd

from engine.short_term_context import ShortTermContextArchive


def _report(date: str = "20260701") -> dict:
    evidence = {
        "strategy": {"theme": "算力"},
        "news_evidence": {"sentiment_label": "bullish", "risk_events": []},
    }
    candidate = {
        "code": "000001", "name": "测试", "board": "算力",
        "candidate_evidence": evidence,
    }
    return {
        "date": date,
        "data_quality": "ok",
        "sentiment": {"temperature": 68},
        "market_modules": {"market_phase": {"market_phase": "主升期"}},
        "boards": [{"name": "算力", "pct": 2.1, "score": 80}],
        "news": {
            "date": date,
            "hot_themes": [["算力", 3]],
            "source_state": {
                "archive": {"status": "ok", "mode": "persisted_exact_date", "date": date},
            },
        },
        "candidates": [candidate],
        "final_recommendations": [candidate],
        "watchlist": [],
        "rejected": [],
        "candidate_evidence": {"000001": evidence},
    }


def test_context_archive_roundtrip_and_exact_date_replay(tmp_path) -> None:
    archive = ShortTermContextArchive(tmp_path)
    saved = archive.save_pipeline_result(_report("20260702"))
    assert saved["complete"] is True
    loaded = archive.load("20260702")
    assert loaded["market_phase"] == "主升期"
    assert loaded["stock_context"]["000001"]["news_scanned"] is True

    kline = pd.DataFrame([
        {"日期": "20260701", "收盘": 10},
        {"日期": "20260702", "收盘": 10.5},
        {"日期": "20260703", "收盘": 10.6},
    ])
    contexts = archive.build_replay_context(
        "20260701", "000001", kline, {"entry_temperature": 70},
    )
    assert "20260702" in contexts
    assert "20260703" not in contexts
    assert contexts["20260702"]["news_evidence"]["sentiment_label"] == "bullish"
    assert contexts["20260702"]["theme_invalidated"] is False


def test_missing_exact_news_marks_global_context_incomplete(tmp_path) -> None:
    report = _report()
    report["news"]["source_state"]["archive"] = {"status": "unavailable", "mode": "exact_date_required"}
    archive = ShortTermContextArchive(tmp_path)
    saved = archive.save_pipeline_result(report)
    assert saved["complete"] is False


def test_unscanned_stock_does_not_get_fake_neutral_news(tmp_path) -> None:
    archive = ShortTermContextArchive(tmp_path)
    archive.save_pipeline_result(_report("20260702"))
    kline = pd.DataFrame([{"日期": "20260702", "收盘": 10.5}])
    contexts = archive.build_replay_context("20260701", "000002", kline)
    assert "20260702" in contexts
    assert "news_evidence" not in contexts["20260702"]
    assert "theme_invalidated" not in contexts["20260702"]
