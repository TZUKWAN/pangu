"""Announcement research replay selection tests."""

from __future__ import annotations

import json

import pandas as pd

from engine.announcement_detail import AnnouncementDetail
from engine.announcement_replay import AnnouncementReplayResearch, AnnouncementSignal


def _frame(future_jump: float = 0.0) -> pd.DataFrame:
    rows = []
    for i in range(35):
        close = 10 + i * 0.05
        rows.append({
            "日期": f"202606{i + 1:02d}",
            "开盘": close - 0.02,
            "最高": close + 0.08,
            "最低": close - 0.08,
            "收盘": close,
            "成交量": 100000,
        })
    rows[-1]["收盘"] += future_jump
    return pd.DataFrame(rows)


def test_technical_snapshot_uses_only_rows_on_or_before_signal() -> None:
    frame = _frame()
    before, reason = AnnouncementReplayResearch._technical_snapshot(frame, "20260625")
    changed = frame.copy()
    changed.loc[changed["日期"] > "20260625", "收盘"] = 999.0
    after, changed_reason = AnnouncementReplayResearch._technical_snapshot(changed, "20260625")
    assert reason == changed_reason
    assert before == after


def test_technical_snapshot_rejects_chasing() -> None:
    frame = _frame()
    mask = frame["日期"].isin(["20260623", "20260624", "20260625"])
    frame.loc[mask, "收盘"] = [11.5, 12.2, 13.0]
    _, reason = AnnouncementReplayResearch._technical_snapshot(frame, "20260625")
    assert reason == "five_day_chasing"


def test_replay_blocks_title_positive_when_pdf_substance_is_not_tradable(tmp_path) -> None:
    class PurchaseContractAnalyzer:
        def analyze(self, event: dict) -> AnnouncementDetail:
            return AnnouncementDetail(
                announcement_id=event["announcement_id"],
                code=event["code"],
                event_type=event["event_type"],
                title=event["title"],
                text_chars=500,
                catalyst_strength=20.0,
                tradable_catalyst=False,
                contract_direction="purchase",
                reasons=["采购合同不是新增销售收入"],
            )

    class StrongIndustryReader:
        def context(self, code: str, date: str, *, allow_previous: bool) -> dict:
            return {
                "theme_status_known": True,
                "theme_context_date": date,
                "theme": "测试行业",
                "industry": "测试行业",
                "industry_trend": {"trend_status": "strong", "trend_score": 80},
                "theme_invalidated": False,
                "theme_trend_strong": True,
            }

    signal = AnnouncementSignal(
        signal_date="20260625",
        code="000001",
        name="测试股份",
        event_type="major_contract",
        title="重大合同公告",
        announcement_id="123",
        adjunct_url="https://static.cninfo.com.cn/finalpage/2026-06-25/123.PDF",
    )
    research = AnnouncementReplayResearch(
        object(),
        announcement_root=tmp_path / "archive",
        detail_analyzer=PurchaseContractAnalyzer(),
        market_breadth_root=tmp_path / "breadth",
        industry_reader=StrongIndustryReader(),
    )
    breadth_root = tmp_path / "breadth"
    breadth_root.mkdir()
    (breadth_root / "20260625.json").write_text(json.dumps({
        "date": "20260625",
        "temperature": 60,
        "posture": "normal",
        "breadth": {"advance_ratio": 0.6},
        "data_quality": {"market_context_exact": True},
    }), encoding="utf-8")
    research._load_signals = lambda *_: [signal]
    research._load_klines = lambda *_: ({signal.code: _frame()}, [])
    research._technical_snapshot = lambda *_: ({"close": 10.0}, "")

    report = research.run("20260625", "20260625", as_of="20260701")

    assert report["technical_confirmed_signals"] == 1
    assert report["sentiment_confirmed_signals"] == 1
    assert report["industry_trend_confirmed_signals"] == 1
    assert report["substance_confirmed_signals"] == 0
    assert report["acceptance"]["executed_count"] == 0
    assert report["rejection_reasons"]["purchase_contract_not_revenue"] == 1
    assert report["announcement_details"][0]["contract_direction"] == "purchase"


def test_weekend_signal_uses_last_exact_trading_day_context(tmp_path) -> None:
    root = tmp_path / "breadth"
    root.mkdir()
    (root / "20260710.json").write_text(json.dumps({
        "date": "20260710",
        "temperature": 55.5,
        "posture": "normal",
        "breadth": {},
        "data_quality": {"market_context_exact": True},
    }), encoding="utf-8")
    research = AnnouncementReplayResearch(
        object(), announcement_root=tmp_path / "archive", market_breadth_root=root
    )
    context = research._market_context("20260711", allow_previous=True)
    assert context["market_context_exact"] is True
    assert context["market_context_date"] == "20260710"
    assert context["current_temperature"] == 55.5
    assert research._market_context("20260711", allow_previous=False)["market_context_exact"] is False
