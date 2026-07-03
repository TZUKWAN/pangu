"""趋势扫描器纯逻辑测试（mock 数据）。"""

import pandas as pd
import pytest

from engine.trend_scanner import (
    _ma,
    _zscore,
    _find_col,
    StockCandidate,
    TrendResult,
)


def test_ma_basic():
    k = pd.DataFrame({"收盘": [10, 11, 12, 13, 14]})
    assert _ma(k, 5) == pytest.approx(12.0)
    assert _ma(k, 3) == pytest.approx(13.0)


def test_ma_insufficient_data():
    k = pd.DataFrame({"收盘": [10, 11]})
    assert _ma(k, 5) is None


def test_ma_alternate_column():
    # 列名不是「收盘」时，退而用第 5 列（index 4）
    k = pd.DataFrame({
        "a": [1] * 5, "b": [2] * 5, "c": [3] * 5, "d": [4] * 5,
        "close": [10, 11, 12, 13, 14],
    })
    assert _ma(k, 5) == pytest.approx(12.0)


def test_zscore_constant():
    s = pd.Series([5, 5, 5])
    z = _zscore(s)
    assert (z == 0).all()


def test_zscore_varies():
    s = pd.Series([1, 2, 3])
    z = _zscore(s)
    assert z.iloc[0] < 0 < z.iloc[2]   # 小的为负，大的为正


def test_find_col_fuzzy():
    df = pd.DataFrame({"涨跌幅.1": [1], "名称": ["x"]})
    assert _find_col(df, ["涨跌幅"]) == "涨跌幅.1"
    assert _find_col(df, ["名称"]) == "名称"
    assert _find_col(df, ["不存在"]) is None


def test_find_col_prefers_exact():
    """精确匹配优先于子串：避免 '涨跌幅' 被误命中 '涨跌幅.1'。"""
    df = pd.DataFrame({"涨跌幅.1": [1], "涨跌幅": [2], "名称": ["x"]})
    # 两列都存在时，应精确命中 "涨跌幅" 而非 "涨跌幅.1"
    assert _find_col(df, ["涨跌幅"]) == "涨跌幅"
    assert _find_col(df, ["不存在"]) is None


def test_stock_candidate_to_dict():
    c = StockCandidate(
        code="000001", name="平安银行", board="银行", close=12.5,
        pct_change=3.2, turnover_rate=1.5, circ_mv_yi=200, rps=85,
        reasons=["均线多头", "放量"], fund_inflow_days=3, score=60,
    )
    d = c.to_dict()
    assert d["code"] == "000001"
    assert d["rps"] == 85.0
    assert "均线多头" in d["reasons"]


def test_trend_result_to_dict():
    r = TrendResult(boards=[{"name": "AI"}], candidates=[], scanned_count=10)
    d = r.to_dict()
    assert d["scanned_count"] == 10
    assert d["boards"][0]["name"] == "AI"


class FakeBoardDataLoader:
    """仅提供 concept_boards 的 DataLoader，用于测试板块排名。"""

    def __init__(
        self,
        concept: pd.DataFrame,
        constituents: dict[str, pd.DataFrame] | None = None,
        limit_up: pd.DataFrame | None = None,
        strong: pd.DataFrame | None = None,
    ) -> None:
        self._concept = concept
        self._constituents = constituents or {}
        self._limit_up = limit_up if limit_up is not None else pd.DataFrame()
        self._strong = strong if strong is not None else pd.DataFrame()

    def concept_boards(self):
        return self._concept

    def sector_fund_flow_rank(self, indicator="今日"):
        return pd.DataFrame()

    def concept_constituents(self, board_symbol: str, board_name: str | None = None):
        return self._constituents.get(str(board_symbol), pd.DataFrame())

    def limit_up_pool(self, date=None):
        return self._limit_up

    def strong_pool(self, date=None):
        return self._strong


def test_rank_boards_with_low_persistence_penalty(tmp_path):
    """板块轮动持续性低时，板块得分应被扣分。"""
    from engine.trend_scanner import TrendScanner
    from engine.market_structure import HistoryKeeper

    concept = pd.DataFrame({
        "板块名称": ["AI", "机器人", "CPO", "光伏", "芯片", "传媒", "游戏"],
        "板块代码": ["BK0001"] * 7,
        "涨跌幅": [5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5],
    })
    dl = FakeBoardDataLoader(concept)
    scanner = TrendScanner(dl, {
        "board": {"top_n": 5},
        "sector_rotation": {"persistence_threshold": 0.9, "persistence_penalty": 10.0},
        "history_dir": str(tmp_path),
    })
    # 构造历史：今日领涨 AI/机器人/CPO/光伏/芯片，与昨日完全不同
    scanner.history.record_sector_leaders("20240105", ["新能源", "汽车", "医药", "白酒", "银行"])
    scanner.history.record_sector_leaders("20240104", ["新能源", "汽车", "医药", "白酒", "银行"])
    scanner.history.record_sector_leaders("20240103", ["新能源", "汽车", "医药", "白酒", "银行"])

    boards = scanner._rank_boards(pd.DataFrame(), date="20240106")
    assert len(boards) == 5
    # 因持续性极低，第一名得分应被扣掉 penalty
    assert boards[0]["score"] < _zscore(concept["涨跌幅"]).iloc[0] * 0.6


def test_sector_persistence_score_no_history(tmp_path):
    """无历史数据时返回 None，不触发过滤。"""
    from engine.trend_scanner import TrendScanner

    concept = pd.DataFrame({
        "板块名称": ["AI", "机器人"],
        "板块代码": ["BK0001", "BK0002"],
        "涨跌幅": [5.0, 4.0],
    })
    dl = FakeBoardDataLoader(concept)
    scanner = TrendScanner(dl, {
        "history_dir": str(tmp_path),
    })
    score = scanner._sector_persistence_score(date="20240102")
    assert score is None


def test_enrich_spot_with_tencent_fills_market_values(monkeypatch, tmp_path):
    """全市场降级扫描前，应能用腾讯行情补齐同花顺源缺失的市值字段。"""
    from engine import tdx_source
    from engine.trend_scanner import TrendScanner

    def fake_tencent_quote(codes):
        assert codes == ["000001"]
        return {
            "000001": {
                "float_mcap_yi": 123.4,
                "mcap_yi": 456.7,
                "pe_ttm": 8.9,
                "pb": 1.2,
            }
        }

    monkeypatch.setattr(tdx_source, "tencent_quote", fake_tencent_quote)
    scanner = TrendScanner(
        FakeBoardDataLoader(pd.DataFrame()),
        {"history_dir": str(tmp_path)},
    )
    spot = pd.DataFrame({
        "_code": ["000001"],
        "代码": ["000001"],
        "名称": ["平安银行"],
        "流通市值": [0],
    })

    out = scanner._enrich_spot_with_tencent(spot, "_code")

    assert out.loc[0, "流通市值"] == pytest.approx(123.4 * 1e8)
    assert out.loc[0, "总市值"] == pytest.approx(456.7 * 1e8)
    assert out.loc[0, "市盈率-动态"] == pytest.approx(8.9)
    assert out.loc[0, "市净率"] == pytest.approx(1.2)


def test_scan_broad_pool_fills_target(tmp_path):
    """板块/严格候选不足时，宽松观察池应补充到目标数量。"""
    from engine.trend_scanner import TrendScanner

    spot = pd.DataFrame({
        "代码": [f"{i:06d}" for i in range(1, 201)],
        "名称": [f"票{i}" for i in range(1, 201)],
        "最新价": [10.0] * 200,
        "涨跌幅": [3.0] * 200,
        "换手率": [5.0] * 200,
        "流通市值": [5_000_000_000.0] * 200,
    })
    scanner = TrendScanner(
        FakeBoardDataLoader(pd.DataFrame()),
        {
            "history_dir": str(tmp_path),
            "stock": {"broad_pool_target": 80, "broad_pct_min": 0.0},
        },
    )
    seen: set[str] = {"000001"}
    extra = scanner._scan_broad_pool(spot, seen)
    assert len(extra) >= 79
    assert all(c.score == 0 for c in extra)
    assert "观察池" in extra[0].reasons[0]


def test_rank_boards_builds_code_to_board_mapping(monkeypatch, tmp_path):
    """同花顺人气榜 concept_tag 应回填到候选股板块映射。"""
    from engine.p0_factors import P0FactorCollector
    from engine.trend_scanner import TrendScanner

    def fake_hot_list(self):
        return [
            {"code": "000001", "name": "平安银行", "concepts": ["AI算力", "金融科技"]},
            {"code": "000002", "name": "万科A", "concepts": ["AI算力"]},
            {"code": "000003", "name": "测试股", "concepts": ["机器人"]},
            {"code": "000004", "name": "样本股", "concepts": ["机器人"]},
        ]

    monkeypatch.setattr(P0FactorCollector, "_ths_hot_list", fake_hot_list)
    spot = pd.DataFrame({
        "代码": ["000001", "000002", "000003", "000004"],
        "名称": ["平安银行", "万科A", "测试股", "样本股"],
        "涨跌幅": [8.0, 7.0, 4.0, 3.0],
        "成交额": [1e9, 8e8, 5e8, 4e8],
        "主力净流入-净额": [1e7, 1e7, 1e6, 1e6],
    })
    scanner = TrendScanner(
        FakeBoardDataLoader(pd.DataFrame()),
        {"board": {"top_n": 2}, "history_dir": str(tmp_path)},
    )

    boards = scanner._rank_boards(spot, date="20260703")

    assert boards[0]["name"] == "AI算力"
    assert scanner._board_for_code("000001", "全市场强势") == "AI算力"
    assert scanner._board_for_code("000003", "全市场强势") == "机器人"


def test_top_board_constituents_extend_code_mapping(monkeypatch, tmp_path):
    """Top 概念板块成分股应补全不在人气榜里的候选映射。"""
    from engine.p0_factors import P0FactorCollector
    from engine.trend_scanner import TrendScanner

    def fake_hot_list(self):
        return [
            {"code": "000001", "name": "平安银行", "concepts": ["AI算力"]},
            {"code": "000002", "name": "万科A", "concepts": ["AI算力"]},
        ]

    monkeypatch.setattr(P0FactorCollector, "_ths_hot_list", fake_hot_list)
    spot = pd.DataFrame({
        "代码": ["000001", "000002"],
        "名称": ["平安银行", "万科A"],
        "涨跌幅": [8.0, 7.0],
        "成交额": [1e9, 8e8],
        "主力净流入-净额": [1e7, 1e7],
    })
    concept = pd.DataFrame({
        "板块名称": ["AI算力"],
        "板块代码": ["309001"],
        "涨跌幅": [4.0],
    })
    constituents = {
        "309001": pd.DataFrame({"代码": ["000099"], "名称": ["非热榜成分股"]})
    }
    scanner = TrendScanner(
        FakeBoardDataLoader(concept, constituents),
        {"board": {"top_n": 1}, "history_dir": str(tmp_path)},
    )

    scanner._rank_boards(spot, date="20260703")

    assert scanner._board_for_code("000099", "全市场强势") == "AI算力"


def test_traditional_boards_backfill_mapping_when_hot_list_empty(monkeypatch, tmp_path):
    """人气榜缺概念标签时，应回退到同花顺概念板块+成分股映射。"""
    from engine.p0_factors import P0FactorCollector
    from engine.trend_scanner import TrendScanner

    monkeypatch.setattr(P0FactorCollector, "_ths_hot_list", lambda self: [])
    spot = pd.DataFrame({
        "代码": ["000001", "000002"],
        "名称": ["平安银行", "万科A"],
        "涨跌幅": [8.0, 7.0],
        "成交额": [1e9, 8e8],
    })
    concept = pd.DataFrame({
        "板块名称": ["AI算力"],
        "板块代码": ["309001"],
        "涨跌幅": [4.0],
    })
    constituents = {
        "309001": pd.DataFrame({"代码": ["000001", "000099"], "名称": ["平安银行", "非热榜成分股"]})
    }
    scanner = TrendScanner(
        FakeBoardDataLoader(concept, constituents),
        {"board": {"top_n": 1}, "history_dir": str(tmp_path)},
    )

    boards = scanner._rank_boards(spot, date="20260703")

    assert boards[0]["name"] == "AI算力"
    assert scanner._board_for_code("000001", "全市场强势") == "AI算力"
    assert scanner._board_for_code("000099", "全市场强势") == "AI算力"


def test_ths_limit_and_strong_pools_extend_board_mapping(monkeypatch, tmp_path):
    """同花顺涨停/强势池题材应作为非东财弱映射补充。"""
    from engine.p0_factors import P0FactorCollector
    from engine.trend_scanner import TrendScanner

    monkeypatch.setattr(P0FactorCollector, "_ths_hot_list", lambda self: [])
    spot = pd.DataFrame({
        "代码": ["000001", "000002"],
        "名称": ["测试A", "测试B"],
        "涨跌幅": [8.0, 7.0],
        "成交额": [1e9, 8e8],
    })
    strong = pd.DataFrame({
        "代码": ["000002"],
        "名称": ["测试B"],
        "所属行业": ["机器人+高端装备"],
    })
    scanner = TrendScanner(
        FakeBoardDataLoader(pd.DataFrame(), strong=strong),
        {"board": {"top_n": 1}, "history_dir": str(tmp_path)},
    )

    scanner._rank_boards(spot, date="20260703")

    assert scanner._board_for_code("000002", "全市场强势") == "机器人"


def test_name_keyword_board_fallback_is_weak_industry_label(tmp_path):
    from engine.trend_scanner import TrendScanner

    scanner = TrendScanner(FakeBoardDataLoader(pd.DataFrame()), {"history_dir": str(tmp_path)})

    assert scanner._infer_board_from_name("祥源新材") == "行业:化工"
    assert scanner._infer_board_from_name("津投城开") == "行业:房地产"


def test_stock_candidate_has_momentum_fields():
    """StockCandidate 支持动量字段（替代已移除的龙虎榜）。"""
    c = StockCandidate(
        code="000001", name="平安银行", board="银行", close=12.5,
        pct_change=3.2, turnover_rate=1.5, circ_mv_yi=200, rps=85,
        reasons=["均线多头"], fund_inflow_days=3, score=60,
    )
    d = c.to_dict()
    assert "lhb_score" not in d
    assert "lhb_summary" not in d
