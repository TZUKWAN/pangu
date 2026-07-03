"""新闻情绪打分纯逻辑测试（mock Markdown，不依赖网络）。"""

import pytest

from engine.news_sentiment import NewsSentimentScorer, ThemeSentiment


SAMPLE_FINNEWS = """# FinNews 金融热点简报 · 2026-06-27

## 今日看点
1. **半导体设备国产化加速**：晶圆厂扩产带动订单增长，利好上游设备与材料。影响：国产替代主线仍有机会。
2. **新能源车企销量超预期**：比亚迪 002594 6 月销量同比大增，带动电池产业链走强。影响：关注电池龙头。

## A股异动
### 领涨板块
| 排名 | 板块 | 涨跌幅 |
|------|------|--------|
| 1 | 工业气体 | +5.8% |
| 2 | 玻璃基板 | +4.2% |

### 异动解读
今日市场热点集中在半导体与新能源方向，资金呈现净流入态势，工业气体板块 603698 受订单利好驱动大涨。

## 概念科普
### 玻璃基板
**解释**：用于显示面板的核心材料。
**为什么重要**：消费电子复苏带动需求回暖，行业龙头 000725 京东方A 受益明显。

## 资金动向
| 方向 | 标的 | 数据 |
|------|------|------|
| 北向资金 | 净流入 | 约45亿元 |
| 比亚迪 | 涨跌幅 | +3.2% |

## 今日总结
情绪温和，关注半导体与新能源的持续性，回避高位回调风险。
"""


def test_find_latest_report(tmp_path):
    """应能找到最新匹配的 Markdown 文件。"""
    old = tmp_path / "2026-06-26-finnews.md"
    new = tmp_path / "2026-06-27-finnews.md"
    old.write_text("old", encoding="utf-8")
    new.write_text(SAMPLE_FINNEWS, encoding="utf-8")
    scorer = NewsSentimentScorer(report_dir=str(tmp_path))
    latest = scorer._find_latest_report()
    assert latest is not None
    assert latest.name == "2026-06-27-finnews.md"


def test_extract_themes_basic():
    scorer = NewsSentimentScorer()
    themes = scorer._extract_themes(SAMPLE_FINNEWS)
    names = {t.name for t in themes}
    assert "今日看点" in names or any("今日看点" in n for n in names)
    assert any("工业气体" in t.name for t in themes)
    assert any("玻璃基板" in t.name for t in themes)


def test_theme_sentiment_positive():
    scorer = NewsSentimentScorer()
    t = scorer._score_theme("半导体", "订单增长，超预期，强势上涨")
    assert t.label == "positive"
    assert t.score > 60


def test_theme_sentiment_negative():
    scorer = NewsSentimentScorer()
    t = scorer._score_theme("高位股", "回调风险，净流出，谨慎回避")
    assert t.label == "negative"
    assert t.score < 40


def test_extract_codes():
    scorer = NewsSentimentScorer()
    codes = scorer._extract_codes(SAMPLE_FINNEWS)
    assert "002594" in codes
    assert "603698" in codes
    assert "000725" in codes


def test_parse_report_returns_code_scores():
    tmp = pytest.importorskip("pathlib").Path
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = tmp(d) / "2026-06-27-finnews.md"
        p.write_text(SAMPLE_FINNEWS, encoding="utf-8")
        scorer = NewsSentimentScorer(report_dir=d)
        result = scorer.score()
        assert "002594" in result
        assert "603698" in result
        assert "000725" in result
        for code in ["002594", "603698", "000725"]:
            assert "sentiment_score" in result[code]
            assert "themes" in result[code]
            assert "risks" in result[code]


def test_empty_report_dir_returns_empty():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        scorer = NewsSentimentScorer(report_dir=d)
        assert scorer.score() == {}


def test_code_region_mapping():
    """代码应被正确归属到对应题材区域。"""
    scorer = NewsSentimentScorer()
    themes = scorer._extract_themes(SAMPLE_FINNEWS)
    code_themes, _ = scorer._map_codes_to_themes(SAMPLE_FINNEWS, themes)
    # 比亚迪在今日看点区域被提及
    assert "002594" in code_themes
    # 京东方A在玻璃基板概念科普区域被提及
    assert "000725" in code_themes
    assert any("玻璃基板" in t for t in code_themes.get("000725", set()))



def test_stock_news_positive_sentiment():
    """个股新闻标题含明显正向词时应给出 positive 分数和 explain。"""
    scorer = NewsSentimentScorer()
    stock_news = {
        "000001": [
            {"title": "平安银行业绩超预期，净利润大涨 20%", "content": "", "source": "财联社", "time": "10:00"},
        ]
    }
    result = scorer.score_from_text("# 新闻\n## 市场\n情绪温和", stock_news=stock_news)
    r = result["000001"]
    assert r["sentiment_label"] == "positive"
    assert r["sentiment_score"] > 60
    assert "财联社" in r["explain"]
    assert "正向词" in r["explain"]
    assert "银行" in r["themes"]


def test_stock_news_negative_sentiment():
    """个股新闻标题含明显负向词时应给出 negative 分数。"""
    scorer = NewsSentimentScorer()
    stock_news = {
        "000002": [
            {"title": "万科A遭遇减持，股价暴跌，风险加大", "content": "", "source": "证券时报", "time": "11:00"},
        ]
    }
    result = scorer.score_from_text("# 新闻\n## 市场\n情绪温和", stock_news=stock_news)
    r = result["000002"]
    assert r["sentiment_label"] == "negative"
    assert r["sentiment_score"] < 40
    assert "负向词" in r["explain"]


def test_stock_news_neutral_sentiment_has_explain():
    """无情感词的个股新闻应给出 neutral 分数和说明来源的 explain。"""
    scorer = NewsSentimentScorer()
    stock_news = {
        "000003": [
            {"title": "某公司召开年度股东大会", "content": "会议审议日常议案", "source": "公司公告", "time": "09:30"},
        ]
    }
    result = scorer.score_from_text("# 新闻\n## 市场\n情绪温和", stock_news=stock_news)
    r = result["000003"]
    assert r["sentiment_label"] == "neutral"
    assert 45 <= r["sentiment_score"] <= 55
    assert "未命中任何情感词" in r["explain"]


def test_no_stock_news_returns_empty():
    """无个股新闻且快讯无相关代码时应返回空映射。"""
    scorer = NewsSentimentScorer()
    result = scorer.score_from_text("# 新闻\n## 市场\n情绪温和，无明确方向")
    assert result == {}


def test_stock_news_themes_are_industry_not_sentiment_words():
    """个股新闻提取的题材应为行业/概念词，而非情感词。"""
    scorer = NewsSentimentScorer()
    stock_news = {
        "000004": [
            {"title": "新能源储能订单大涨，光伏装机超预期", "content": "", "source": "财联社"},
        ]
    }
    result = scorer.score_from_text("# 新闻\n## 市场\n情绪温和", stock_news=stock_news)
    themes = result["000004"]["themes"]
    assert "新能源" in themes or "光伏" in themes or "储能" in themes
    assert "上涨" not in themes
    assert "超预期" not in themes
