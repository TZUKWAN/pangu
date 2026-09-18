"""WSCN news beneficiary-linking tests."""

from __future__ import annotations

import pandas as pd

from engine.news_fetcher import NewsResult
from engine.news_opportunity import NewsOpportunityScanner
from engine.wscn_news_replay import WscnNewsReplayResearch


def _research() -> WscnNewsReplayResearch:
    return WscnNewsReplayResearch.__new__(WscnNewsReplayResearch)


def test_contract_beneficiary_is_not_customer_named_first() -> None:
    spot = pd.DataFrame({
        "代码": ["600941", "603019"],
        "名称": ["中国移动", "中科曙光"],
    })
    content = (
        "近日，中国移动发布分布式存储采购项目中标候选人公示。"
        "中科曙光成功入围并获得第一份额，中标金额约1.8亿元。"
    )
    stocks = _research()._direct_stocks(content, spot)
    assert stocks == [{"code": "603019", "name": "中科曙光"}]


def test_colon_prefixed_company_is_directly_linked() -> None:
    spot = pd.DataFrame({"代码": ["600971"], "名称": ["恒源煤电"]})
    stocks = _research()._direct_stocks(
        "恒源煤电：预计上半年净利润将实现扭亏为盈。", spot
    )
    assert stocks == [{"code": "600971", "name": "恒源煤电"}]


def test_plain_company_mention_without_beneficiary_action_is_not_direct() -> None:
    spot = pd.DataFrame({"代码": ["600941"], "名称": ["中国移动"]})
    stocks = _research()._direct_stocks(
        "某公司连续多年成为中国移动核心供应商。", spot
    )
    assert stocks == []


def test_digest_is_split_so_company_evidence_does_not_use_market_lead_paragraph() -> None:
    research = _research()
    research._name_index_cache = {}
    research.scanner = NewsOpportunityScanner({
        "short_term_agent": {
            "news_discovery_min_score": 0,
            "news_discovery_max_candidates": 20,
        }
    })
    spot = pd.DataFrame({
        "代码": ["601138", "603986"],
        "名称": ["工业富联", "兆易创新"],
    })
    item = {
        "id": "digest-1",
        "published_at": "2026-07-10T07:35:08+08:00",
        "important": True,
        "content": (
            "芯片股再撑美股，美光收涨4.5%，闪迪涨7.6%。\n\n"
            "工业富联上半年净利预增最高翻一倍，AI服务器营收同比增超230%。"
            "兆易创新上半年净利预增10.99倍。"
        ),
        "themes": [],
    }
    news = NewsResult(date="20260710")
    news.flashes = research._flashes(item, spot)
    result = research.scanner.scan(news, spot)

    assert len(news.flashes) == 3
    assert {item.code for item in result.opportunities} == {"601138", "603986"}
    by_code = {item.code: item for item in result.opportunities}
    assert "美光收涨4.5%" not in by_code["601138"].evidence[0]
    assert "工业富联" in by_code["601138"].evidence[0]
    assert "兆易创新" not in by_code["601138"].evidence[0]
    assert "兆易创新" in by_code["603986"].evidence[0]
    assert "工业富联" not in by_code["603986"].evidence[0]
