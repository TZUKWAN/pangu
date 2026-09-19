"""涨停/跌停微观结构因子（factors/structure.py）。

涨跌停幅度规则 limit_ratio：
- is_st -> 0.05
- sz.3 前缀（创业板 300/301）/ sh.68 前缀（科创板 688/689）-> 0.20
- 其余 -> 0.10

判定（A 股价格按 0.01 取整）：涨停价 = round(preclose*(1+ratio), 2)，
close >= 涨停价 - 1e-9 视为涨停。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FactorMeta, PanelFactor, cross_section

LIMIT_TOL = 1e-9


def limit_ratio(code: str, is_st: bool | None = None) -> float:
    """个股当日涨跌停幅度（is_st 优先）。"""
    if is_st:
        return 0.05
    c = str(code).upper()
    if c.endswith(".SZ") and c[:2] in ("30",):     # sz.3 前缀：创业板
        return 0.20
    if c.endswith(".SH") and c[:2] in ("68",):     # sh.68 前缀：科创板
        return 0.20
    return 0.10


def limit_up_flags(panel: pd.DataFrame) -> pd.Series:
    """逐行涨停标记（(date, code) MultiIndex，与 panel 对齐）。"""
    codes = panel.index.get_level_values("code")
    ratio = pd.Series([limit_ratio(c) for c in codes], index=panel.index)
    st = panel["is_st"].astype(bool)
    ratio = ratio.mask(st, 0.05)
    limit_price = (panel["preclose"] * (1.0 + ratio)).round(2)
    return panel["close"] >= limit_price - LIMIT_TOL


def _per_code(flags: pd.Series, func) -> pd.Series:
    return flags.groupby(level="code", group_keys=False).apply(func)


class LimitUpCountFactor(PanelFactor):
    """近 20 个交易日涨停天数： close 触及 round(preclose*(1+ratio),2)。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="limit_up_count_20d", version="1.0.0", family="structure",
            description="近20日涨停天数（按板块/ST自适应涨跌停幅度）",
            required_fields=("close", "preclose", "is_st"), lookback_days=20,
            economic_hypothesis="涨停是 A 股特有的一字限价与注意力机制：近期多次"
                                "涨停意味着强资金共识与动量（涨停动量假说）。",
            missing_rule="窗口内行缺失（停牌）按实际行计；无样本返回 NaN。",
            winsorize=None, standardize="none",  # 原始计数语义
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        flags = limit_up_flags(panel).astype(float)
        cnt = _per_code(flags, lambda s: s.rolling(self.meta.lookback_days,
                                                   min_periods=1).sum())
        return cross_section(cnt, asof)


class DaysSinceLimitUpFactor(PanelFactor):
    """距最近一次涨停的交易日数；回看窗口内无涨停 -> NaN（missing_rule）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="days_since_limit_up", version="1.0.0", family="structure",
            description="距最近一次涨停的交易日数",
            required_fields=("close", "preclose", "is_st"), lookback_days=20,
            economic_hypothesis="涨停后短线动量随时间衰减：距上一涨停越近，"
                                "情绪余温越强（涨停余温假说，预期与未来收益负相关—越近越强）。",
            missing_rule="回看窗口内无涨停记 NaN（不伪造）。",
            winsorize=None, standardize="none",  # 原始天数语义
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        flags = limit_up_flags(panel)

        def _since(s: pd.Series) -> pd.Series:
            vals = s.values.astype(bool)
            pos = np.flatnonzero(vals)
            out = np.full(len(vals), np.nan)
            if len(pos):
                out[:] = len(vals) - 1 - pos[-1]
            return pd.Series(out, index=s.index)

        return cross_section(_per_code(flags, _since), asof)


class ConsecLimitUpDaysFactor(PanelFactor):
    """截至 asof 的连续涨停天数（asof 当日未涨停则为 0）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="consec_limit_up_days", version="1.0.0", family="structure",
            description="连续涨停天数（连板数）",
            required_fields=("close", "preclose", "is_st"), lookback_days=20,
            economic_hypothesis="连板数刻画极致情绪与打板资金接力：连板越高"
                                "短线博弈越激烈（情绪周期假说，高位风险与人气并存）。",
            missing_rule="asof 当日无数据返回 NaN；asof 未涨停记 0。",
            winsorize=None, standardize="none",  # 原始天数语义
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        flags = limit_up_flags(panel)

        def _consec(s: pd.Series) -> pd.Series:
            vals = s.values.astype(bool)
            n = 0
            for v in vals[::-1]:
                if v:
                    n += 1
                else:
                    break
            out = np.zeros(len(vals))
            out[-1] = float(n)
            return pd.Series(out, index=s.index)

        return cross_section(_per_code(flags, _consec), asof)


class NearLimitRateFactor(PanelFactor):
    """近 5 日“接近涨停”频率：当日涨幅 >= 0.9*ratio 视为接近涨停。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="near_limit_rate_5d", version="1.0.0", family="structure",
            description="近5日接近涨停频率（当日涨幅≥0.9×涨跌停幅度）",
            required_fields=("close", "preclose", "is_st"), lookback_days=6,
            economic_hypothesis="频频冲高至涨停附近说明买方持续封板意愿，是"
                                "未确认的涨停前兆（封板压力假说）。",
            missing_rule="至少 3 个有效样本；不足返回 NaN。",
            winsorize=None, standardize="none",  # 原始频率语义
        )

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        codes = panel.index.get_level_values("code")
        ratio = pd.Series([limit_ratio(c) for c in codes], index=panel.index)
        ratio = ratio.mask(panel["is_st"].astype(bool), 0.05)
        gain = panel["close"] / panel["preclose"] - 1.0
        near = (gain >= 0.9 * ratio).astype(float)
        rate = _per_code(near, lambda s: s.rolling(5, min_periods=3).mean())
        return cross_section(rate, asof)
