"""持有周期与退出计划（Phase 5 / Task 5.1-5.4）。

每票独立持有周期：由 alpha horizon（研究窗各 horizon 的 IC 强度）、
事件半衰期、波动率、市场状态共同决定；禁止全组合统一固定天数。
退出计划输出 7 类条件，可在次日按同一函数重新评估。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.decision.regime import MarketRegime, RegimeLabel

# 研究窗 OOS 各 horizon 相对强度（来自 factor study 的 IC 随 horizon 收敛形态；
# 反转族 h5/h10 最强 → 默认 3-5 日；事件驱动按事件半衰期叠加）。
HORIZON_IC_SHAPE = {1: 0.55, 3: 0.90, 5: 1.00, 10: 0.95, 20: 0.80}
DEFAULT_HOLDING_RANGE: Dict[int, Tuple[int, int]] = {
    1: (1, 2), 3: (2, 4), 5: (3, 7), 10: (5, 12), 20: (10, 20)}


@dataclass
class ExitPlan:
    hard_stop: Optional[float]
    profit_target: Optional[float]
    trailing_condition: str
    time_stop_days: int
    news_invalidation: str
    market_invalidation: str
    sector_invalidation: str
    factor_invalidation: str

    def to_conditions(self) -> List[str]:
        conds = []
        if self.hard_stop is not None:
            conds.append(f"hard_stop@{self.hard_stop:.2f}")
        if self.profit_target is not None:
            conds.append(f"profit_target@{self.profit_target:.2f}")
        conds.append(self.trailing_condition)
        conds.append(f"time_stop@{self.time_stop_days}d")
        conds.append("news_invalidation:" + self.news_invalidation)
        conds.append("market_invalidation:" + self.market_invalidation)
        conds.append("sector_invalidation:" + self.sector_invalidation)
        conds.append("factor_invalidation:" + self.factor_invalidation)
        return conds


@dataclass
class HoldingPlan:
    expected_holding_days: int
    holding_range: List[int]
    horizon_used: int
    rationale: List[str] = field(default_factory=list)
    exit_plan: Optional[ExitPlan] = None


def choose_horizon(event_half_life_days: Optional[float],
                   vol20: Optional[float],
                   regime: MarketRegime) -> Tuple[int, List[str]]:
    """选择最合理 horizon（天）。事件半衰期强 → 短；高波/危机 → 缩短。"""
    rationale: List[str] = []
    horizon = 5
    if event_half_life_days is not None and event_half_life_days > 0:
        # 事件alpha主要落在半衰期的 0.5~1.5 倍窗口内
        best = min(HORIZON_IC_SHAPE,
                   key=lambda h: abs(h - min(max(event_half_life_days, 1), 20))
                   + (1 - HORIZON_IC_SHAPE[h]) * 5)
        horizon = best
        rationale.append(f"事件半衰期 {event_half_life_days:.0f}d → horizon {horizon}d")
    if vol20 is not None and vol20 > 0.05:          # 日波动 >5% 的极端波动
        horizon = max(1, min(horizon, 3))
        rationale.append("个股波动极高 → 缩短持有")
    if regime.label in (RegimeLabel.RISK_OFF, RegimeLabel.CRISIS):
        horizon = max(1, min(horizon, 3))
        rationale.append(f"市场状态 {regime.label.value} → 缩短持有")
    rationale.append("horizon 形态来自研究窗 OOS IC（h5 最强）")
    return horizon, rationale


def build_holding_plan(code: str, close: pd.Series, vol20: Optional[float],
                       event_half_life_days: Optional[float],
                       regime: MarketRegime,
                       support: Optional[float] = None) -> HoldingPlan:
    """生成持有计划与 7 类退出条件（全部数值可被次日复评）。"""
    horizon, rationale = choose_horizon(event_half_life_days, vol20, regime)
    lo, hi = DEFAULT_HOLDING_RANGE[horizon]
    last = float(close.dropna().iloc[-1]) if close is not None and close.notna().any() else np.nan
    atr = _atr_proxy(close)
    if support is None or (isinstance(support, float) and np.isnan(support)):
        support = float(np.nanmin(close.dropna().iloc[-20:])) if close is not None else np.nan
    stop = float(min(support * 0.99, last - 2.0 * atr)) if np.isfinite(last) and np.isfinite(atr) else None
    target = float(last + 2.0 * (last - stop)) if stop is not None and np.isfinite(last) else None
    time_stop = hi
    exit_plan = ExitPlan(
        hard_stop=round(stop, 2) if stop else None,
        profit_target=round(target, 2) if target else None,
        trailing_condition="收盘跌破 MA10 或自高点回撤 >6% → 减半",
        time_stop_days=time_stop,
        news_invalidation="持仓期内出现 negative/risk 方向的 direct 事件（decay>0.3）",
        market_invalidation="regime 转为 crisis 或指数收盘跌破 MA20",
        sector_invalidation="所属板块 inherit 事件方向转 negative",
        factor_invalidation="主导因子方向失效（rolling IC 20日 <0 且连续为负）")
    return HoldingPlan(expected_holding_days=horizon,
                       holding_range=[lo, hi], horizon_used=horizon,
                       rationale=rationale, exit_plan=exit_plan)


def _atr_proxy(close: pd.Series) -> float:
    """日收盘近似 ATR：|ret| 均值 × 价格（无高低价依赖的保守估计）。"""
    ret = close.pct_change().dropna().iloc[-20:]
    if ret.empty:
        return float("nan")
    last = float(close.dropna().iloc[-1])
    return float(ret.abs().mean() * last)
