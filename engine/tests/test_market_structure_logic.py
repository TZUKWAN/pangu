"""market_structure 模块纯逻辑测试（mock 数据，不依赖网络）。"""

import pandas as pd
import pytest

from engine.market_structure import HistoryKeeper, MarketStructureAnalyzer


class FakeDataLoader:
    """仅返回空 DataFrame 的 DataLoader，用于测试不依赖网络的逻辑。"""

    def all_spot(self):
        return pd.DataFrame()

    def limit_up_pool(self, date=None):
        return pd.DataFrame()

    def broke_pool(self, date=None):
        return pd.DataFrame()

    def limit_down_pool(self, date=None):
        return pd.DataFrame()

    def strong_pool(self, date=None):
        return pd.DataFrame()

    def concept_boards(self):
        return pd.DataFrame()

    def sector_fund_flow_rank(self, indicator="今日"):
        return pd.DataFrame()


def test_history_keeper_prev_leaders_window(tmp_path):
    """HistoryKeeper 能正确返回昨日/近 3 日领涨板块。"""
    keeper = HistoryKeeper(tmp_path)
    keeper.record_sector_leaders("20240105", ["AI", "机器人", "CPO"])
    keeper.record_sector_leaders("20240104", ["AI", "机器人", "光伏"])
    keeper.record_sector_leaders("20240103", ["AI", "芯片", "传媒"])
    keeper.record_sector_leaders("20240102", ["新能源", "汽车", "AI"])

    window = keeper.prev_leaders_window("20240106", window=3)
    assert window["yesterday"] == ["AI", "机器人", "CPO"]
    # 近 3 日合并
    assert set(window["last_n"]) == {"AI", "机器人", "CPO", "光伏", "芯片", "传媒"}


def test_sector_rotation_persistence_calculation(tmp_path):
    """轮动持续性指标正确计算 1 日/3 日重合度。"""
    keeper = HistoryKeeper(tmp_path)
    # 今日是 20240105，昨日 20240104
    keeper.record_sector_leaders("20240104", ["AI", "机器人", "CPO", "传媒", "游戏"])
    keeper.record_sector_leaders("20240103", ["AI", "机器人", "新能源", "汽车", "芯片"])
    keeper.record_sector_leaders("20240102", ["医药", "白酒", "银行", "地产", "钢铁"])

    dl = FakeDataLoader()
    analyzer = MarketStructureAnalyzer(
        dl, {"history_dir": str(tmp_path), "sector_rotation_lookback": 2}
    )
    concept = pd.DataFrame({
        "板块名称": ["AI", "机器人", "CPO", "光伏", "芯片", "传媒", "游戏"],
        "涨跌幅": [5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5],
    })
    res = analyzer._sector_rotation_persistence(concept, "20240105")

    # 今日前 5：AI, 机器人, CPO, 光伏, 芯片
    # 昨日前 5：AI, 机器人, CPO, 传媒, 游戏 → Jaccard 重合 3/7
    assert res["overlap_1d"] == pytest.approx(3 / 7)
    # 近两日（lookback=2）合并去重共 8 个：AI, 机器人, CPO, 传媒, 游戏, 新能源, 汽车, 芯片
    # 与今日前 5 交集 4 个，并集 9 个 → Jaccard 重合 4/9
    assert res["overlap_3d"] == pytest.approx(4 / 9)
    assert res["persistence_score"] > 0


def test_sector_rotation_persistence_no_history(tmp_path):
    """无历史数据时返回中性值并带 warning。"""
    keeper = HistoryKeeper(tmp_path)
    dl = FakeDataLoader()
    analyzer = MarketStructureAnalyzer(dl, {"history_dir": str(tmp_path)})
    concept = pd.DataFrame({
        "板块名称": ["AI", "机器人", "CPO", "光伏", "芯片"],
        "涨跌幅": [5.0, 4.0, 3.5, 3.0, 2.5],
    })
    res = analyzer._sector_rotation_persistence(concept, "20240102")
    assert res["overlap_1d"] == 0.0
    assert res["overlap_3d"] == 0.0
    assert "warning" in res
