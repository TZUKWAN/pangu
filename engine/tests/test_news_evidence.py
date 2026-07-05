"""新闻证据层单元测试。"""

from __future__ import annotations

import pytest

from engine.news_evidence import (
    CandidateEvidence,
    NewsEvidenceCollector,
    NewsQueryContext,
    ThemeEvidence,
)


@pytest.fixture
def collector() -> NewsEvidenceCollector:
    return NewsEvidenceCollector()


def test_theme_evidence_bullish(collector: NewsEvidenceCollector) -> None:
    flashes = [
        {"content": "算力板块大涨，龙头涨停", "source": "cls", "subjects": ["算力"]},
        {"content": "算力板块持续走强，资金流入", "source": "wscn"},
    ]
    ctx = NewsQueryContext(date="20260703", hot_themes=[("算力", 2)])
    result = collector.collect(ctx, flashes, {})
    ev = result["theme_evidence"]["算力"]
    assert ev["sentiment_label"] == "bullish"
    assert ev["support_count"] >= 2
    assert any("大涨" in s for s in ev["bullish_snippets"])


def test_theme_evidence_bearish_with_risk(collector: NewsEvidenceCollector) -> None:
    flashes = [
        {"content": "光伏板块不及预期，龙头业绩变脸", "source": "cls"},
    ]
    ctx = NewsQueryContext(date="20260703", hot_themes=[("光伏", 1)])
    result = collector.collect(ctx, flashes, {})
    ev = result["theme_evidence"]["光伏"]
    assert ev["sentiment_label"] == "bearish"
    assert ev["risk_events"]


def test_candidate_evidence_inherits_theme(collector: NewsEvidenceCollector) -> None:
    """场景13：泛题材利好、无个股直接关联时不加分（保持中性）。"""
    flashes = [
        {"content": "机器人赛道利好，政策大力支持", "source": "cls", "subjects": ["机器人"]},
    ]
    candidates = [{"code": "000001", "name": "测试机器人", "board": "机器人"}]
    ctx = NewsQueryContext(
        date="20260703",
        hot_themes=[("机器人", 1)],
        candidates=candidates,
        candidate_themes={"000001": {"机器人"}},
    )
    result = collector.collect(ctx, flashes, {})
    ev = result["candidate_evidence"]["000001"]
    # 仅题材继承、无个股直接关联：保持中性，不加个股多空分
    assert ev["sentiment_label"] == "neutral"
    assert ev["individual_backed"] is False
    assert ev["direct_count"] == 0
    assert "机器人" in ev["themes"]
    # 继承的题材片段仍可见于上下文，但不改变定性
    assert ev["inherited_count"] >= 1


def test_candidate_evidence_individual_backed_gets_bullish(collector: NewsEvidenceCollector) -> None:
    """对照场景13：个股直接被快讯提及 + 题材利好 → 可加分。"""
    flashes = [
        {"content": "000001 机器人赛道利好，政策大力支持", "source": "cls", "subjects": ["机器人"]},
    ]
    candidates = [{"code": "000001", "name": "测试机器人", "board": "机器人"}]
    ctx = NewsQueryContext(
        date="20260703",
        hot_themes=[("机器人", 1)],
        candidates=candidates,
        candidate_themes={"000001": {"机器人"}},
    )
    result = collector.collect(ctx, flashes, {})
    ev = result["candidate_evidence"]["000001"]
    assert ev["sentiment_label"] == "bullish"
    assert ev["individual_backed"] is True
    assert ev["direct_count"] >= 1


def test_top_level_aggregations_present(collector: NewsEvidenceCollector) -> None:
    """顶层多空/中性/催化聚合字段应存在。"""
    flashes = [
        {"content": "算力板块大涨", "source": "cls", "subjects": ["算力"]},
    ]
    ctx = NewsQueryContext(date="20260703", hot_themes=[("算力", 1)])
    result = collector.collect(ctx, flashes, {})
    for key in ("bullish_news", "bearish_news", "neutral_news", "catalysts"):
        assert key in result
    assert isinstance(result["bullish_news"], list)
    assert isinstance(result["neutral_news"], list)
    assert isinstance(result["catalysts"], list)


def test_candidate_evidence_direct_stock_mention(collector: NewsEvidenceCollector) -> None:
    flashes = [
        {"content": "000002 发布减持公告，股价承压", "source": "sina"},
    ]
    candidates = [{"code": "000002", "name": "万科A"}]
    ctx = NewsQueryContext(date="20260703", candidates=candidates)
    result = collector.collect(ctx, flashes, {})
    ev = result["candidate_evidence"]["000002"]
    assert ev["sentiment_label"] == "bearish"
    assert ev["risk_events"]


def test_market_narrative_sorted(collector: NewsEvidenceCollector) -> None:
    flashes = [
        {"content": "芯片板块大涨", "source": "cls", "subjects": ["芯片"]},
        {"content": "白酒板块走弱", "source": "cls", "subjects": ["白酒"]},
    ]
    ctx = NewsQueryContext(date="20260703", hot_themes=[("芯片", 1), ("白酒", 1)])
    result = collector.collect(ctx, flashes, {})
    narratives = result["market_narrative"]
    assert narratives
    assert narratives[0]["theme"] == "芯片"


def test_empty_input_returns_safe_defaults(collector: NewsEvidenceCollector) -> None:
    ctx = NewsQueryContext(date="20260703")
    result = collector.collect(ctx, [], {})
    assert result["theme_evidence"] == {}
    assert result["candidate_evidence"] == {}
    assert result["market_narrative"] == []


def test_label_score_math() -> None:
    ev = ThemeEvidence(theme="t", bullish_snippets=["a", "b"], bearish_snippets=["c"], risk_events=[])
    label, score = NewsEvidenceCollector()._label_score(ev)
    assert label == "bullish"
    assert score > 50

    ev2 = ThemeEvidence(theme="t", bullish_snippets=[], bearish_snippets=[], risk_events=["r"])
    label2, score2 = NewsEvidenceCollector()._label_score(ev2)
    assert label2 == "bearish"
    assert score2 < 50


def test_candidate_evidence_verdict() -> None:
    ev = CandidateEvidence(
        code="000001", name="x",
        bullish_snippets=["a"], bearish_snippets=[], risk_events=[],
    )
    ev.sentiment_label, _ = NewsEvidenceCollector()._label_score(ev)
    ev.support_count = 1
    ev.verdict_reason = NewsEvidenceCollector()._verdict_reason(ev)
    assert "偏弱" in ev.verdict_reason
