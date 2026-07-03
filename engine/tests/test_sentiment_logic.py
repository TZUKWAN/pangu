"""情绪温度计纯逻辑测试（mock 数据，不依赖网络）。

验证打分函数、姿态判定、权重加权的正确性。
"""

import pandas as pd
import pytest

from engine.sentiment_meter import (
    SentimentMeter,
    _anchor_score,
    _clamp,
    _max_consecutive,
)


class FakeDataLoader:
    """内存版 DataLoader，返回预设 DataFrame，避免网络。"""

    def __init__(self, zt, broke, dt, spot):
        self._zt = zt
        self._broke = broke
        self._dt = dt
        self._spot = spot

    def limit_up_pool(self, date=None):
        return self._zt

    def broke_pool(self, date=None):
        return self._broke

    def limit_down_pool(self, date=None):
        return self._dt

    def all_spot(self):
        return self._spot


def make_meter(zt, broke, dt, spot):
    dl = FakeDataLoader(zt, broke, dt, spot)
    return SentimentMeter(dl, {})


# ---------------------------------------------------------------------- #
def test_clamp():
    assert _clamp(50) == 50
    assert _clamp(-10) == 0
    assert _clamp(150) == 100


def test_anchor_score_monotonic():
    """锚点打分应单调递增。"""
    assert _anchor_score(0, 15, 40, 80, 120) == 0       # 冰点
    assert _anchor_score(15, 15, 40, 80, 120) == 0
    mid = _anchor_score(40, 15, 40, 80, 120)
    assert 45 <= mid <= 55                                # 正常锚点≈50
    hot = _anchor_score(80, 15, 40, 80, 120)
    assert hot == 85
    assert _anchor_score(200, 15, 40, 80, 120) == 100    # 极热封顶


def test_max_consecutive_numeric():
    df = pd.DataFrame({"连板数": [1, 3, 5, 2]})
    assert _max_consecutive(df) == 5


def test_max_consecutive_string_stat():
    # akshare「涨停统计」形如 "6天5板"：第二个数字才是连板数
    df = pd.DataFrame({"涨停统计": ["3天2板", "6天5板", "10天8板"]})
    # 应取连板数（第二个数字）的最大 → 8，而非天数（第一个数字）10
    assert _max_consecutive(df) == 8


def test_sentiment_cold_posture():
    """涨停极少 + 跌停多 → 冰点。"""
    zt = pd.DataFrame({"连板数": [1, 1, 1]})          # 仅 3 家涨停
    broke = pd.DataFrame({"代码": ["1", "2"]})        # 2 家炸板
    dt = pd.DataFrame({"代码": list(range(80))})      # 80 家跌停（恐慌）
    # 涨少跌多
    spot = pd.DataFrame({"涨跌幅": [-5] * 3000 + [2] * 1000})
    m = make_meter(zt, broke, dt, spot)
    bd = m.measure("20250101")
    assert bd.temperature < 40
    assert bd.posture == "冰点"
    assert bd.limit_down_count == 80


def test_sentiment_hot_posture():
    """涨停很多 + 连板高 + 无炸板 + 无跌停 → 亢奋。"""
    zt = pd.DataFrame({"连板数": [8] * 120})          # 120 家涨停，最高 8 板
    broke = pd.DataFrame({"代码": []})                 # 0 炸板
    dt = pd.DataFrame({"代码": []})                    # 0 跌停
    spot = pd.DataFrame({"涨跌幅": [3] * 4000 + [-1] * 500})  # 涨远多于跌
    m = make_meter(zt, broke, dt, spot)
    bd = m.measure("20250101")
    assert bd.temperature > 85
    assert bd.posture == "亢奋"
    assert bd.limit_up_count == 120
    assert bd.consecutive_height == 8


def test_sentiment_normal_posture():
    """温和市况 → 正常区间。"""
    zt = pd.DataFrame({"连板数": [3] * 45})           # 45 家涨停
    broke = pd.DataFrame({"代码": list(range(10))})   # 10 家炸板
    dt = pd.DataFrame({"代码": list(range(8))})       # 8 家跌停
    spot = pd.DataFrame({"涨跌幅": [1] * 2500 + [-1] * 2200})
    m = make_meter(zt, broke, dt, spot)
    bd = m.measure("20250101")
    assert 40 <= bd.temperature <= 85
    assert bd.posture == "正常"


def test_sentiment_to_dict_has_components():
    zt = pd.DataFrame({"连板数": [1, 2, 3]})
    broke = pd.DataFrame({"代码": [1]})
    dt = pd.DataFrame({"代码": [1, 2]})
    spot = pd.DataFrame({"涨跌幅": [1, -1, 2, -2]})
    m = make_meter(zt, broke, dt, spot)
    d = m.measure("20250101").to_dict()
    assert "temperature" in d
    assert "components" in d
    assert "posture" in d
    assert d["components"]["limit_up_count"] == 3
