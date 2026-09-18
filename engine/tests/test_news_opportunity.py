"""News-first discovery and causal archive tests."""

from __future__ import annotations

import pandas as pd

from engine.news_fetcher import NewsFetcher, NewsFlash, NewsResult, StockNews
from engine.news_opportunity import NewsOpportunityScanner, normalize_a_share_code


def _spot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "代码": ["300911", "603685", "000001"],
            "名称": ["亿田智能", "晨丰科技", "平安银行"],
        }
    )


def test_normalize_a_share_code() -> None:
    assert normalize_a_share_code("sz300911") == "300911"
    assert normalize_a_share_code("603685.SH") == "603685"
    assert normalize_a_share_code("hk00700") == ""
    assert normalize_a_share_code("noise") == ""


def test_purchase_contract_is_not_treated_as_revenue_catalyst() -> None:
    news = NewsResult(date="20260726")
    news.flashes = [
        NewsFlash(
            time="15:34",
            content="亿田智能子公司签订重大算力服务器采购合同，订单金额20亿元",
            important=True,
            subjects=["A股公告速递", "算力"],
            stocks=[{"code": "sz300911", "name": "亿田智能"}],
            source="cls",
        )
    ]
    result = NewsOpportunityScanner().scan(news, _spot())
    assert result.linked_flashes == 0
    assert result.opportunities == []
    assert result.risk_alerts == []


def test_negative_precedence_prevents_false_buyback_signal() -> None:
    news = NewsResult(date="20260726")
    news.flashes = [
        NewsFlash(
            time="18:00",
            content="公司终止回购计划，控股股东同时拟减持股份",
            stocks=[{"code": "603685", "name": "晨丰科技"}],
            source="cls",
        )
    ]
    result = NewsOpportunityScanner().scan(news, _spot())
    assert result.opportunities == []
    assert len(result.risk_alerts) == 1
    assert result.risk_alerts[0].polarity == "negative"
    assert result.risk_alerts[0].event_type in {"shareholder_reduction", "event_terminated"}


def test_unlinked_macro_news_does_not_create_stock_candidate() -> None:
    news = NewsResult(date="20260726")
    news.flashes = [
        NewsFlash(
            time="12:00",
            content="海外市场发布人工智能产业支持政策",
            subjects=["人工智能"],
            source="wscn",
        )
    ]
    result = NewsOpportunityScanner().scan(news, _spot())
    assert result.opportunities == []
    assert result.risk_alerts == []
    assert result.linked_flashes == 0


def test_negated_restructuring_and_conditional_risk_are_not_high_confidence_buy() -> None:
    """Regression from the real 2026-07-26 亿田智能 announcement."""
    news = NewsResult(date="20260726")
    news.flashes = [
        NewsFlash(
            time="15:34",
            content=(
                "亿田智能子公司拟不超20亿元采购服务器，为客户提供算力服务。"
                "本次交易不构成重大资产重组，尚需提交股东会审议；"
                "算力业务尚处拓展初期，存在市场风险。"
            ),
            subjects=["A股公告速递", "算力"],
            stocks=[{"code": "sz300911", "name": "亿田智能"}],
            source="cls",
        )
    ]
    result = NewsOpportunityScanner().scan(news, _spot())
    assert result.opportunities == []
    assert result.risk_alerts == []


def test_exact_date_archive_roundtrip_is_lossless(tmp_path) -> None:
    fetcher = NewsFetcher(
        dl=None,
        cfg={"news": {"archive_dir": str(tmp_path), "cache_ttl_minutes": 30}},
    )
    news = NewsResult(date="20260105")
    news.flashes = [
        NewsFlash(time=f"10:{i:02d}", content=f"第{i}条重大合同 300911", source="test")
        for i in range(25)
    ]
    news.hot_themes = [("算力", 3)]
    fetcher._save_archive(news)

    loaded = fetcher.fetch_today(date="20260105")
    assert len(loaded.flashes) == 25
    assert loaded.flashes[-1].content.startswith("第24条")
    assert loaded.hot_themes == [("算力", 3)]
    assert loaded.source_state["archive"]["mode"] == "exact_date"


def test_missing_historical_archive_fails_closed_without_live_news(tmp_path) -> None:
    fetcher = NewsFetcher(dl=None, cfg={"news": {"archive_dir": str(tmp_path)}})
    result = fetcher.fetch_today(date="19990101")
    assert result.flashes == []
    assert result.source_state["archive"]["status"] == "unavailable"
    assert any("禁止用当前新闻替代历史证据" in warning for warning in result.warnings)


def test_enriched_stock_news_is_persisted_for_pipeline_reuse(tmp_path, monkeypatch) -> None:
    fetcher = NewsFetcher(dl=None, cfg={"news": {"archive_dir": str(tmp_path)}})
    result = NewsResult(date="20260726")
    result.flashes = [NewsFlash(time="10:00", content="市场快讯", source="test")]
    monkeypatch.setattr(
        fetcher,
        "_fetch_stock_news",
        lambda code: [StockNews(code=code, title=f"{code} 签订重大合同", source="test")],
    )

    fetcher.enrich_stock_news(result, [{"code": "000001", "name": "测试股"}])

    archive_path = tmp_path / "20260726.json"
    assert archive_path.exists()
    loaded = NewsResult.from_dict(__import__("json").loads(archive_path.read_text(encoding="utf-8")))
    assert loaded.stock_news["000001"][0].title == "000001 签订重大合同"
