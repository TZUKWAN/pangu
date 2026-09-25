"""市场状态（Phase 4 / Task 4.4）。

从面板与指数计算连续市场特征并给出状态标签；不同 regime 使用不同
因子权重剖面（禁止全市场一套固定权重）。全部特征仅用 asof 前数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd


class RegimeLabel(str, Enum):
    RISK_ON = "risk_on"          # 趋势向上、宽度扩散
    NEUTRAL = "neutral"          # 震荡
    RISK_OFF = "risk_off"        # 趋势向下/高波
    CRISIS = "crisis"            # 恐慌（高波+宽度崩塌）


@dataclass
class MarketRegime:
    label: RegimeLabel
    trend_above_ma20: Optional[bool]     # 指数 vs MA20
    breadth_above_ma20: float            # 个股高于 MA20 占比 0-1
    vol_percentile: float                # 指数 20d 波动的全样本分位 0-1
    limit_up_count: int
    limit_down_count: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"label": self.label.value,
             "trend_above_ma20": self.trend_above_ma20,
             "breadth_above_ma20": round(self.breadth_above_ma20, 4),
             "vol_percentile": round(self.vol_percentile, 4),
             "limit_up_count": self.limit_up_count,
             "limit_down_count": self.limit_down_count,
             "notes": self.notes}
        return d


# 因子权重剖面：不同 regime 下因子方向权重的缩放（相对基础权重）
REGIME_WEIGHT_PROFILES: Dict[RegimeLabel, Dict[str, float]] = {
    RegimeLabel.RISK_ON:   {"reversal": 0.8, "lowvol": 0.9, "illiquidity": 0.9,
                            "lowturnover": 0.9, "event": 1.0},
    RegimeLabel.NEUTRAL:   {"reversal": 1.0, "lowvol": 1.0, "illiquidity": 1.0,
                            "lowturnover": 1.0, "event": 1.0},
    RegimeLabel.RISK_OFF:  {"reversal": 0.7, "lowvol": 1.3, "illiquidity": 0.8,
                            "lowturnover": 1.2, "event": 0.8},
    RegimeLabel.CRISIS:    {"reversal": 0.5, "lowvol": 1.5, "illiquidity": 0.6,
                            "lowturnover": 1.3, "event": 0.6},
}


def compute_regime(panel: pd.DataFrame, index_close: Optional[pd.Series],
                   asof: str) -> MarketRegime:
    """只用 <= asof 的数据判定市场状态。

    panel: MultiIndex(date, code) 需含 close/preclose/amount。
    index_close: 基准指数收盘（date 索引，升序）。
    """
    dates = panel.index.get_level_values("date").unique().sort_values()
    dates = dates[dates <= asof]
    if len(dates) == 0:
        return MarketRegime(RegimeLabel.NEUTRAL, None, 0.5, 0.5, 0, 0,
                            notes=["empty panel → neutral fallback"])
    notes: List[str] = []
    close = panel["close"].unstack("code").sort_index().loc[:asof]

    # 宽度：个股高于自身 MA20 的比例
    ma20 = close.rolling(20, min_periods=10).mean()
    last_close, last_ma = close.iloc[-1], ma20.iloc[-1]
    valid = last_close.notna() & last_ma.notna()
    breadth = float((last_close[valid] > last_ma[valid]).mean()) if valid.any() else 0.5

    # 指数趋势与波动
    trend_above: Optional[bool] = None
    vol_pct = 0.5
    if index_close is not None and len(index_close) >= 25:
        idx = index_close.sort_index().loc[:asof]
        ma20i = idx.rolling(20, min_periods=15).mean()
        trend_above = bool(idx.iloc[-1] > ma20i.iloc[-1])
        ret = idx.pct_change().dropna()
        vol20 = ret.rolling(20).std().dropna()
        if len(vol20) >= 30:
            vol_pct = float((vol20 <= vol20.iloc[-1]).mean())

    # 涨跌停结构（最近一日，按 preclose ±10% 简化口径，仅供 regime 参考）
    lu = ld = 0
    if len(close) >= 2 and close.index[-1] is not None:
        pre = close.iloc[-2]
        cur = close.iloc[-1]
        both = pre.notna() & cur.notna()
        lu = int((cur[both] >= pre[both] * 1.0985).sum())
        ld = int((cur[both] <= pre[both] * 0.9015).sum())

    # 标签
    if trend_above is False and (breadth < 0.35 or vol_pct > 0.9):
        label = RegimeLabel.CRISIS
        notes.append("趋势向下且宽度崩塌/高波 → crisis")
    elif trend_above is False or vol_pct > 0.8:
        label = RegimeLabel.RISK_OFF
    elif trend_above is True and breadth > 0.55:
        label = RegimeLabel.RISK_ON
    else:
        label = RegimeLabel.NEUTRAL

    return MarketRegime(label=label, trend_above_ma20=trend_above,
                        breadth_above_ma20=breadth, vol_percentile=vol_pct,
                        limit_up_count=lu, limit_down_count=ld, notes=notes)
