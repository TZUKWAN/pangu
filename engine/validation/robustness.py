"""Pangu 2.0 稳健性检验：参数邻域、成本压力、延迟压力、子期表现。"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Mapping, Sequence

import pandas as pd


def _metric_of(result) -> float:
    """从 run 结果中提取总收益（dict.summary / dict.total_return / obj.summary）。"""
    if isinstance(result, Mapping):
        summary = result.get("summary")
        if isinstance(summary, Mapping) and "total_return" in summary:
            return float(summary["total_return"])
        if "total_return" in result:
            return float(result["total_return"])
        return 0.0
    summary = getattr(result, "summary", None)
    if isinstance(summary, Mapping) and "total_return" in summary:
        return float(summary["total_return"])
    return 0.0


def param_neighborhood(run_fn: Callable, base_params: Mapping,
                       keys: Sequence[str], fracs=(0.8, 0.9, 1.1, 1.2)) -> dict:
    """对指定参数做 ±10%/±20% 邻域扫描。

    fragile = 仅精确基准参数盈利（所有扰动均不盈利）。
    """
    base_params = dict(base_params)
    base_metric = _metric_of(run_fn(**base_params))
    results: dict[str, dict] = {}
    variants: list[float] = []
    for key in keys:
        per_key: dict[str, float] = {}
        for frac in fracs:
            params = dict(base_params)
            params[key] = base_params[key] * float(frac)
            m = _metric_of(run_fn(**params))
            per_key[str(frac)] = m
            variants.append(m)
        results[str(key)] = per_key
    fragile = bool(base_metric > 0) and all(v <= 0 for v in variants)
    return {"base": base_metric, "results": results, "fragile": fragile}


def cost_stress(run_fn: Callable, multipliers=(1, 2, 3)) -> dict:
    """成本压力：run_fn(cost_mult)（实现方按倍数放大滑点/费率）。"""
    results = {str(m): _metric_of(run_fn(m)) for m in multipliers}
    out = {"results": results}
    out["survives_2x"] = (results["2"] > 0) if "2" in results else None
    return out


def delay_stress(run_fn: Callable, delays=(0, 1)) -> dict:
    """延迟压力：run_fn(delay)，delay 为策略决策顺延的交易日数。"""
    return {"results": {str(d): _metric_of(run_fn(d)) for d in delays}}


class DelayedStrategy:
    """把策略决策日期顺延 k 个交易日的包装器（配合 BacktestV2 使用）。

    在日期 index=i 被引擎询问时，实际用 index=i-k 的决策与受限视图
    （history.asof_view 保证其仍看不到 > i-k 的数据）。
    """

    def __init__(self, strategy, trading_days: Sequence[str], delay: int):
        self.strategy = strategy
        self.days = [str(d) for d in trading_days]
        self.delay = int(delay)

    def rebalance(self, decision_date: str, history) -> list:
        if self.delay <= 0:
            return self.strategy.rebalance(decision_date, history)
        try:
            i = self.days.index(str(decision_date))
        except ValueError:
            return []
        j = i - self.delay
        if j < 0:
            return []
        return self.strategy.rebalance(self.days[j], history.asof_view(self.days[j]))


def subperiod_metrics(equity, regime_fn: Callable | None = None) -> dict:
    """按 regime 与按日历年的子期表现 {by_regime, by_year}。"""
    if isinstance(equity, pd.DataFrame):
        s = equity["equity"].astype(float)
    else:
        s = pd.Series(equity, dtype=float)
    s = s.dropna()

    def _year(idx_val) -> str:
        if isinstance(idx_val, str):
            return idx_val[:4]
        return str(pd.Timestamp(idx_val).year)

    def _stats(sub: pd.Series) -> dict:
        if len(sub) < 2:
            return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                    "n_days": max(0, len(sub) - 1)}
        rets = (sub / sub.shift(1) - 1.0).dropna()
        std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
        peak = sub.cummax()
        dd = (peak - sub) / peak
        return {
            "total_return": float(sub.iloc[-1] / sub.iloc[0] - 1.0),
            "sharpe": float(rets.mean() / std * math.sqrt(252)) if std > 0 else 0.0,
            "max_drawdown": float(dd.max()) if len(dd) else 0.0,
            "n_days": int(len(rets)),
        }

    by_year = {y: _stats(g) for y, g in s.groupby([_year(i) for i in s.index])}
    by_regime: dict[str, dict] = {}
    if regime_fn is not None:
        labels = [str(regime_fn(i)) for i in s.index]
        grouped: dict[str, list] = defaultdict(list)
        for lab, (idx, val) in zip(labels, s.items()):
            grouped[lab].append(val)
        for lab, vals in grouped.items():
            by_regime[lab] = _stats(pd.Series(vals, dtype=float))
    return {"by_regime": by_regime, "by_year": by_year}
