"""P2-002 因子库测试：全量 compute 冒烟 + 涨停结构因子的手工面板精确验证。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research import SyntheticResearchData
from engine.research.factors import FactorUnavailableError
from engine.research.factors.base import Factor
from engine.research.factors.registry import build_default_registry
from engine.research.factors.structure import limit_ratio

# 合成数据上自然涨停极罕见，这两个因子允许常数截面（行为正确）；
# 它们的精确行为用下面的手工 StubData 验证。
ALLOW_CONSTANT = {"consec_limit_up_days", "near_limit_rate_5d"}


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture(scope="module")
def data():
    return SyntheticResearchData(seed=7, n_symbols=40, n_days=300)


def _tradable_universe(data, asof):
    u = data.universe(asof)
    return u.index[u.tradable & ~u.suspended]


# ---------------------------------------------------------------------------
# limit_ratio 规则
# ---------------------------------------------------------------------------

def test_limit_ratio_rules():
    assert limit_ratio("000001.SZ") == 0.10
    assert limit_ratio("600519.SH") == 0.10
    assert limit_ratio("300750.SZ") == 0.20      # sz.3 创业板
    assert limit_ratio("301236.SZ") == 0.20
    assert limit_ratio("688981.SH") == 0.20      # sh.68 科创板
    assert limit_ratio("000001.SZ", is_st=True) == 0.05
    assert limit_ratio("688981.SH", is_st=True) == 0.05  # ST 优先


# ---------------------------------------------------------------------------
# 手工 StubData：涨停行为精确验证
# ---------------------------------------------------------------------------

class StubData:
    """最小 ResearchData 实现，价格手工构造、涨停可控。"""

    def __init__(self):
        dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-02", periods=12)]
        self.dates = dates
        rows = []

        def add(code, t, close, preclose, is_st=False):
            rows.append({
                "date": dates[t], "code": code,
                "open": preclose, "high": max(close, preclose),
                "low": min(close, preclose), "close": close, "preclose": preclose,
                "volume": 1000.0, "amount": 1000.0 * close,
                "pct_change": (close / preclose - 1) * 100.0,
                "turnover": 1.0, "is_st": is_st,
            })

        # 000001.SZ 主板 10%：第 5/6/7 天三连板（10 -> 11 -> 12.10 -> 13.31）
        px = {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0,
              5: 11.0, 6: 12.10, 7: 13.31, 8: 13.31, 9: 13.31, 10: 13.31, 11: 13.31}
        prev = 10.0
        for t in range(12):
            add("000001.SZ", t, px[t], prev)
            prev = px[t]

        # 000002.SZ ST 5%：第 5 天触板（10 -> 10.50）
        prev = 10.0
        for t in range(12):
            close = 10.50 if t == 5 else 10.0 if t < 5 else 10.0
            add("000002.SZ", t, close, prev, is_st=True)
            prev = close

        df = pd.DataFrame(rows)
        self._panel = df.set_index(["date", "code"]).sort_index()
        self._panel.index.names = ["date", "code"]

    def daily_panel(self, start, end, symbols=None):
        idx = self._panel.index
        m = (idx.get_level_values("date") >= start) & (idx.get_level_values("date") <= end)
        out = self._panel[m]
        if symbols is not None:
            out = out[out.index.get_level_values("code").isin(set(symbols))]
        return out.copy()

    def universe(self, date):
        codes = self._panel.index.get_level_values("code").unique()
        return pd.DataFrame({
            "name": codes, "is_st": [c == "000002.SZ" for c in codes],
            "suspended": [False] * len(codes), "listed_days": [500] * len(codes),
            "tradable": [True] * len(codes),
        }, index=pd.Index(codes, name="code"))

    def index_daily(self, code, start, end):
        raise NotImplementedError

    def trading_days(self, start, end):
        return [d for d in self.dates if start <= d <= end]


def test_limit_up_count_and_consec_on_stub():
    from engine.research.factors.structure import (
        ConsecLimitUpDaysFactor,
        DaysSinceLimitUpFactor,
        LimitUpCountFactor,
    )
    stub = StubData()
    uni = stub.universe(stub.dates[9]).index

    count = LimitUpCountFactor().compute(stub.dates[9], uni, stub)
    assert count["000001.SZ"] == 3.0           # 三连板
    assert count["000002.SZ"] == 1.0           # ST 5% 涨停同样被识别

    consec7 = ConsecLimitUpDaysFactor().compute(stub.dates[7], uni, stub)
    assert consec7["000001.SZ"] == 3.0         # 截至第 7 天连板数 = 3
    consec9 = ConsecLimitUpDaysFactor().compute(stub.dates[9], uni, stub)
    assert consec9["000001.SZ"] == 0.0         # 第 8/9 天未涨停

    since9 = DaysSinceLimitUpFactor().compute(stub.dates[9], uni, stub)
    assert since9["000001.SZ"] == 2.0          # 距第 7 天的涨停 2 个交易日
    since7 = DaysSinceLimitUpFactor().compute(stub.dates[7], uni, stub)
    assert since7["000001.SZ"] == 0.0


def test_near_limit_rate_on_stub():
    from engine.research.factors.structure import NearLimitRateFactor
    stub = StubData()
    uni = stub.universe(stub.dates[7]).index
    rate = NearLimitRateFactor().compute(stub.dates[7], uni, stub)
    # 近 5 日（3..7）涨幅 [0, 0, +10%, +10%, +10%]，均 >= 0.9*10%
    assert rate["000001.SZ"] == pytest.approx(0.6)
    assert rate["000002.SZ"] == pytest.approx(1.0 / 5.0)


# ---------------------------------------------------------------------------
# 全量因子冒烟
# ---------------------------------------------------------------------------

def test_all_ready_factors_compute_clean_cross_sections(registry, data):
    asof = data.dates[-30]
    uni = _tradable_universe(data, asof)
    assert len(uni) > 30
    for rec in registry.list():
        f = registry.get(rec["name"])
        assert isinstance(f, Factor)
        if rec["availability"] != "ready":
            with pytest.raises(FactorUnavailableError):
                f.compute(asof, uni, data)
            continue
        out = f.compute(asof, uni, data)
        assert not out.empty, rec["name"]
        assert set(out.index).issubset(set(uni)), rec["name"]
        v = out.dropna()
        assert not v.empty, f"{rec['name']}: all-NaN cross section"
        assert np.isfinite(v.to_numpy()).all(), rec["name"]
        if rec["name"] not in ALLOW_CONSTANT and not rec["name"].startswith("mkt_"):
            assert v.nunique() > 1, f"{rec['name']}: constant cross section"
        assert not out.index.duplicated().any(), rec["name"]


def test_momentum_factor_monotone_in_window(data):
    """同一构造下 mom_5d 与 mom_20d 应高度相关（对实现合理性做健全性检查）。"""
    asof = data.dates[-20]
    uni = _tradable_universe(data, asof)
    m5 = _raw_unstandardized(data, asof, uni, 5)
    m20 = _raw_unstandardized(data, asof, uni, 20)
    both = pd.concat({"m5": m5, "m20": m20}, axis=1).dropna()
    assert len(both) > 20
    assert abs(np.corrcoef(both["m5"], both["m20"])[0, 1]) > 0.5


def _raw_unstandardized(data, asof, uni, window):
    from engine.research.factors.base import cross_section, group_apply, load_window
    panel = load_window(data, asof, uni, window + 1)
    mom = group_apply(panel["close"], lambda s: s / s.shift(window) - 1.0)
    return cross_section(mom, asof)


def test_rsi_14_bounded(registry, data):
    asof = data.dates[-20]
    uni = _tradable_universe(data, asof)
    v = registry.get("rsi_14").compute(asof, uni, data).dropna()
    assert ((v >= 0) & (v <= 100)).all()


def test_rps_20d_in_0_100(registry, data):
    asof = data.dates[-20]
    uni = _tradable_universe(data, asof)
    v = registry.get("rps_20d").compute(asof, uni, data).dropna()
    assert ((v >= 0) & (v <= 100)).all()
    assert v.nunique() > 10


def test_market_factors_broadcast_single_value(registry, data):
    asof = data.dates[-20]
    uni = _tradable_universe(data, asof)
    for name in ("mkt_breadth_5d", "mkt_vol_20d"):
        v = registry.get(name).compute(asof, uni, data).dropna()
        assert not v.empty
        assert v.nunique() == 1  # 全截面同值：横截面无区分度（设计如此）
