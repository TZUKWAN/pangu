"""FactorMeta / Factor 抽象基类 / 横截面预处理（P2-001）。

硬性规则（P2-002 / P0-007）：
- 因子输出一律是 raw_score（原始打分，经 winsorize/zscore 或 rank-zscore
  的横截面标准化），任何命名/文案不得暗示“概率/probability”。
- compute() 严格 point-in-time：只允许读 asof 当日及以前的数据。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # 避免运行时循环依赖
    from ..data_interface import ResearchData


class FactorUnavailableError(RuntimeError):
    """因子诚实声明不可用（如无基本面数据源），绝不伪造数据。"""


@dataclass(frozen=True)
class FactorMeta:
    """因子元数据（注册表按此登记；code_hash 由 registry 计算）。"""

    name: str
    version: str
    family: str
    description: str
    required_fields: tuple[str, ...]
    lookback_days: int                    # 需要的历史交易日数（含 asof）
    economic_hypothesis: str              # 经济学假设（必填，评审用）
    missing_rule: str                     # 数据缺失时的处理约定
    winsorize: tuple[float, float] | None = (0.01, 0.99)
    standardize: str = "zscore"           # zscore | rank_zscore | none


# ---------------------------------------------------------------------------
# 横截面预处理
# ---------------------------------------------------------------------------

def winsorize_zscore(series: pd.Series,
                     bounds: tuple[float, float] | None = (0.01, 0.99)) -> pd.Series:
    """缩尾后横截面 z 分数。bounds=None 表示只做 zscore。常数列返回全 0。"""
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return s
    if bounds is not None:
        lo, hi = s.quantile(bounds[0]), s.quantile(bounds[1])
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            s = s.clip(lo, hi)
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def rank_zscore(series: pd.Series) -> pd.Series:
    """横截面百分位排名的 z 分数（对离群值天然稳健，保序）。"""
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return s
    r = s.rank(pct=True)
    sd = r.std()
    if not np.isfinite(sd) or sd == 0:
        return r * 0.0
    return (r - r.mean()) / sd


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class Factor(ABC):
    """单因子：compute(asof, universe, data) -> 以 code 为索引的 raw_score。"""

    meta: FactorMeta

    @abstractmethod
    def compute(self, asof: str, universe: pd.Index,
                data: "ResearchData") -> pd.Series:
        """返回 index=code 的 raw_score 序列。只读 asof 及更早的数据。"""

    # -- 标准化出口 -----------------------------------------------------------

    def finalize(self, series: pd.Series) -> pd.Series:
        """按 meta.winsorize / meta.standardize 做横截面预处理。"""
        std = self.meta.standardize
        if std == "zscore":
            return winsorize_zscore(series, self.meta.winsorize)
        if std == "rank_zscore":
            return rank_zscore(series)
        if std in ("none", "raw"):
            s = series.astype(float).replace([np.inf, -np.inf], np.nan)
            return s
        raise ValueError(f"unknown standardize rule: {std!r}")

    def __repr__(self) -> str:  # 便于日志/注册表调试
        m = self.meta
        return f"<Factor {m.name} v{m.version} family={m.family}>"


# ---------------------------------------------------------------------------
# 面板小工具（library / structure / univariate 共用）
# ---------------------------------------------------------------------------

def window_trading_days(data: "ResearchData", asof: str, lookback: int,
                        calendar_pad: int = 20) -> list[str]:
    """取 asof 之前（含 asof）至多 lookback 个交易日。

    先按日历日粗估再向 data 索取真实交易日，停牌/节假日由面板行缺失体现。
    """
    guess_start = (pd.Timestamp(asof) - pd.Timedelta(days=int(lookback * 2.5) + calendar_pad)
                   ).strftime("%Y-%m-%d")
    days = data.trading_days(guess_start, asof)
    if not days:
        return []
    return days[-lookback:]


def load_window(data: "ResearchData", asof: str, universe: pd.Index,
                lookback: int) -> pd.DataFrame:
    """加载 [asof-lookback, asof] 的面板（严格 <= asof，PIT 兜底再切一刀）。"""
    days = window_trading_days(data, asof, lookback)
    if not days:
        return pd.DataFrame(columns=list(getattr(data, "PANEL_COLUMNS", [])) or None, dtype=float)
    panel = data.daily_panel(days[0], asof, symbols=[str(c) for c in universe])
    if not panel.empty:  # 双保险：任何实现违反 STRICT end 也拦在这里
        panel = panel[panel.index.get_level_values("date") <= asof]
    return panel


def at_date(panel: pd.DataFrame | pd.Series, asof: str):
    """取 date 层 == asof 的行。"""
    return panel[panel.index.get_level_values("date") == asof]


def cross_section(panel: pd.DataFrame | pd.Series, asof: str) -> pd.Series:
    """取 asof 截面并压平为 index=code 的 Series。"""
    cs = at_date(panel, asof)
    out = pd.Series(cs.values, index=cs.index.get_level_values("code"))
    out.index.name = "code"
    return out


def group_apply(series: pd.Series, func: Callable[[pd.Series], pd.Series]) -> pd.Series:
    """按 code 分组做逐组时间序列运算（组内保持日期升序）。"""
    return series.groupby(level="code", group_keys=False).apply(func)


def roll(series: pd.Series, window: int, agg: str = "mean",
         min_periods: int | None = None) -> pd.Series:
    """按 code 分组的滚动聚合（mean/sum/std/max/min/count）。"""
    mp = window if min_periods is None else min_periods
    return group_apply(series, lambda s: getattr(s.rolling(window, min_periods=mp), agg)())


class PanelFactor(Factor):
    """库内因子的公共骨架：加载回看窗口 -> _raw -> 对齐 universe -> finalize。"""

    def compute(self, asof: str, universe: pd.Index, data: "ResearchData") -> pd.Series:
        panel = load_window(data, asof, universe, self.meta.lookback_days)
        if panel.empty:
            return pd.Series(dtype=float, name=self.meta.name)
        raw = self._raw(asof, panel)
        raw = raw.reindex(pd.Index([str(c) for c in universe]))
        return self.finalize(raw)

    def _raw(self, asof: str, panel: pd.DataFrame) -> pd.Series:
        """在回看窗口面板上计算 asof 截面的原始值（index=code）。"""
        raise NotImplementedError
