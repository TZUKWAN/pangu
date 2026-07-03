"""新闻/板块轮动/推荐权重 三模块逻辑测试。

不依赖网络，用 mock 数据验证纯逻辑正确性。
"""

import pandas as pd
import pytest

# ---------------------------------------------------------------------- #
# news_sentiment.score_from_text（实时新闻文本解析）
# ---------------------------------------------------------------------- #
def test_score_from_text_parses_codes():
    """score_from_text 能从文本提取股票代码和题材情绪。"""
    from engine.news_sentiment import NewsSentimentScorer
    scorer = NewsSentimentScorer()
    text = """# 今日财经快讯
## 今日看点
- 14:32 【重要】AI算力板块领涨，000001 平安银行涨停，算力需求强劲增长
- 13:15 半导体板块走强，600519 贵州茅台突破创新高，资金大幅流入

## 领涨板块
AI算力 半导体
"""
    result = scorer.score_from_text(text, source="test")
    # 应该能识别到 000001 / 600519
    assert "000001" in result or "600519" in result
    # 被识别的代码应有情绪分
    for code, info in result.items():
        assert "sentiment_score" in info
        assert 0 <= info["sentiment_score"] <= 100


def test_score_from_text_empty():
    """空文本返回空 dict，不报错。"""
    from engine.news_sentiment import NewsSentimentScorer
    scorer = NewsSentimentScorer()
    assert scorer.score_from_text("", source="empty") == {}


# ---------------------------------------------------------------------- #
# NewsFetcher.to_markdown / 题材提取（纯逻辑，mock 数据）
# ---------------------------------------------------------------------- #
def test_news_result_to_markdown():
    """NewsResult.to_markdown 生成可被 score_from_text 解析的格式。"""
    from engine.news_fetcher import NewsResult, NewsFlash
    result = NewsResult(date="20260630")
    result.flashes = [
        NewsFlash(time="1432", content="AI算力板块上涨，000001涨停", important=True),
        NewsFlash(time="1315", content="半导体走强"),
    ]
    result.hot_themes = [("AI算力", 3), ("半导体", 2)]
    md = result.to_markdown()
    assert "今日财经快讯" in md
    assert "000001" in md
    assert "AI算力" in md
    # 生成的 md 能被 score_from_text 解析回来
    from engine.news_sentiment import NewsSentimentScorer
    scorer = NewsSentimentScorer()
    parsed = scorer.score_from_text(md, source="roundtrip")
    assert "000001" in parsed


def test_news_fetcher_extract_themes():
    """题材热度提取（从电报文本统计关键词频次）。"""
    from engine.news_fetcher import NewsFetcher, NewsResult, NewsFlash
    fetcher = NewsFetcher(dl=None)  # dl 仅用于取数，提取逻辑不需要
    result = NewsResult(date="20260630")
    result.flashes = [
        NewsFlash(time="1", content="AI算力板块大涨，算力需求爆发"),
        NewsFlash(time="2", content="半导体AI芯片紧缺"),
        NewsFlash(time="3", content="新能源车销量创新高，光伏也走强"),
    ]
    themes = fetcher._extract_hot_themes(result)
    # 应该识别出 AI/算力/半导体/新能源/光伏 等
    theme_names = [t[0] for t in themes]
    assert "AI" in theme_names or "算力" in theme_names
    assert "半导体" in theme_names
    # 频次降序
    if len(themes) >= 2:
        assert themes[0][1] >= themes[1][1]


# ---------------------------------------------------------------------- #
# SectorRotation 板块轮动得分
# ---------------------------------------------------------------------- #
def test_sector_rotation_rank_pct():
    """_rank_pct 把净流入额转成百分位排名。"""
    from engine.sector_rotation import SectorRotationAnalyzer
    value_map = {"AI": 1000, "半导体": 500, "光伏": -300, "银行": 200}
    ranks = SectorRotationAnalyzer._rank_pct(value_map)
    # AI 净流入最大，排名应最高（接近100）
    assert ranks["AI"] > ranks["银行"]
    assert ranks["半导体"] > ranks["光伏"]
    assert all(0 <= v <= 100 for v in ranks.values())


def test_sector_rotation_score_of_fuzzy_match():
    """score_of 支持模糊匹配板块名。"""
    from engine.sector_rotation import SectorRotationResult
    r = SectorRotationResult()
    r.scores = {"AI算力概念": 85.0, "半导体": 60.0}
    # 精确匹配
    assert r.score_of("半导体") == 60.0
    # 包含匹配
    assert r.score_of("AI算力") == 85.0
    # 未命中返回中性
    assert r.score_of("未知板块") == 50.0
    assert r.score_of("") == 50.0


# ---------------------------------------------------------------------- #
# Recommender 新权重（6 维正式加权，momentum 作为 breakdown 维度不参与主权重）
# ---------------------------------------------------------------------- #
def test_recommender_weights_normalized():
    """recommender 主权重和为 1.0，news_sentiment 仅作 bonus，lhb 已移除。"""
    from engine.recommender import Recommender
    r = Recommender()
    assert "news_sentiment" in r.weights
    assert "lhb" not in r.weights
    # momentum 作为 breakdown/雷达维度展示，不占用主权重
    # 权重和归一化
    assert abs(sum(r.weights.values()) - 1.0) < 0.01


def test_recommender_news_affects_score():
    """有题材催化的票推荐度更高（news_sentiment 生效）。"""
    from engine.recommender import Recommender
    r = Recommender()
    base_cand = {
        "code": "000001", "name": "测试", "board": "AI", "close": 10.0,
        "rps": 90, "fund_inflow_days": 3, "turnover_rate": 8,
        "reasons": ["均线多头", "突破", "放量"],
        "entry_exit": {"risk_reward_ratio": 2.0,
                       "buy_points": [{"price": 9.8, "is_primary": True, "type": "突破位"}],
                       "stop_loss": {"price": 9.5}, "take_profit": [{"price": 10.5}]},
    }
    # 无新闻情绪
    recs_no_news = r.rank([dict(base_cand)], news_sentiment={})
    # 有正面新闻情绪（AI 题材催化）
    recs_with_news = r.rank([dict(base_cand)],
                            news_sentiment={"000001": {"sentiment_score": 90, "themes": ["AI"]}})
    score_no = recs_no_news[0].recommend_score
    score_with = recs_with_news[0].recommend_score
    # 有题材催化的分数应更高
    assert score_with > score_no
    # breakdown 含 news_sentiment 维度
    assert "news_sentiment" in recs_with_news[0].score_breakdown


# ---------------------------------------------------------------------- #
# tdx_source 腾讯前缀转换（纯逻辑）
# ---------------------------------------------------------------------- #
def test_tencent_prefix():
    """6位代码 → 腾讯市场前缀。"""
    from engine.tdx_source import _tencent_prefix
    assert _tencent_prefix("000001") == "sz000001"
    assert _tencent_prefix("600519") == "sh600519"
    assert _tencent_prefix("980001") == "sh980001"
    assert _tencent_prefix("830001") == "bj830001"


# ---------------------------------------------------------------------- #
# 情绪温度计双模式（东财不可用时近似封板指标）
# ---------------------------------------------------------------------- #
def test_approximate_limit_pools_hot_market():
    """亢奋市（涨停多）近似封板指标应体现亢奋。"""
    import numpy as np
    from engine.market_structure import MarketStructureAnalyzer
    analyzer = MarketStructureAnalyzer.__new__(MarketStructureAnalyzer)
    # 涨停 80 只，跌停 20 只
    pcts = pd.Series(np.concatenate([
        np.random.normal(2, 4, 5000), np.full(80, 9.9), np.full(20, -9.9)
    ]))
    approx = analyzer._approximate_limit_pools(pcts)
    assert approx["limit_up_count"]["raw"] >= 70   # 涨停家数多
    assert approx["limit_down_count"]["raw"] < 40  # 跌停少（随机分布波动留余量）
    assert approx["limit_up_count"]["score"] > approx["limit_down_count"]["score"]


def test_approximate_limit_pools_cold_market():
    """冰点市（跌停多）近似封板指标应体现恐慌。"""
    import numpy as np
    from engine.market_structure import MarketStructureAnalyzer
    analyzer = MarketStructureAnalyzer.__new__(MarketStructureAnalyzer)
    # 跌停 80 只，涨停 20 只
    pcts = pd.Series(np.concatenate([
        np.random.normal(-2, 4, 5000), np.full(20, 9.9), np.full(80, -9.9)
    ]))
    approx = analyzer._approximate_limit_pools(pcts)
    assert approx["limit_down_count"]["raw"] >= 70
    assert approx["limit_up_count"]["raw"] < 40  # 随机分布波动留余量
    assert approx["limit_down_count"]["score"] < approx["limit_up_count"]["score"]


def test_approximate_limit_pools_empty():
    """空涨跌幅分布返回中性值，不报错。"""
    from engine.market_structure import MarketStructureAnalyzer
    analyzer = MarketStructureAnalyzer.__new__(MarketStructureAnalyzer)
    approx = analyzer._approximate_limit_pools(pd.Series(dtype=float))
    for key in ["limit_up_count", "consecutive_height", "broke_rate",
                "limit_down_count", "seal_rate", "strong_pool_count"]:
        assert key in approx
        assert approx[key]["score"] == 50.0  # 中性
