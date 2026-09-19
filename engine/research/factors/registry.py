"""因子注册表：register/get/list，带 availability 诚实标注与 code hash。

code_hash = sha256(inspect.getsource(type(factor)))，用于实验留痕
（因子代码一旦改动，hash 变化，历史报告可追溯到旧版本）。
availability:
- ready：可用
- degraded_no_source：数据源未接入，compute() 抛 FactorUnavailableError
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Optional

from .base import Factor


def code_hash(factor: Factor) -> str:
    try:
        src = inspect.getsource(type(factor))
        return hashlib.sha256(src.encode("utf-8")).hexdigest()
    except (OSError, TypeError):  # REPL / 动态定义类无法取源码
        return "source_unavailable"


class FactorRegistry:
    def __init__(self):
        self._items: dict[str, dict] = {}

    def register(self, factor: Factor, availability: str = "ready") -> None:
        name = factor.meta.name
        if name in self._items:
            raise ValueError(f"duplicate factor registration: {name}")
        if availability not in ("ready", "degraded_no_source", "experimental"):
            raise ValueError(f"unknown availability: {availability}")
        self._items[name] = {
            "factor": factor,
            "availability": availability,
            "code_hash": code_hash(factor),
        }

    def get(self, name: str) -> Factor:
        if name not in self._items:
            raise KeyError(f"factor not registered: {name}")
        return self._items[name]["factor"]

    def availability(self, name: str) -> str:
        if name not in self._items:
            raise KeyError(f"factor not registered: {name}")
        return self._items[name]["availability"]

    def list(self) -> list[dict]:
        out = []
        for name, item in self._items.items():
            m = item["factor"].meta
            out.append({
                "name": name,
                "version": m.version,
                "family": m.family,
                "availability": item["availability"],
                "code_hash": item["code_hash"],
                "lookback_days": m.lookback_days,
            })
        return sorted(out, key=lambda d: (d["family"], d["name"]))

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


def build_default_registry() -> FactorRegistry:
    """注册全部库内因子（单例实例）。"""
    from . import library as L
    from . import structure as S

    reg = FactorRegistry()
    ready = [
        L.MomentumFactor(5), L.MomentumFactor(10), L.MomentumFactor(20), L.MomentumFactor(60),
        L.Rps20dFactor(), L.Ma20SlopeFactor(), L.Breakout20dFactor(),
        L.High52wProximityFactor(), L.VolAdjMom20dFactor(), L.TrendPersistence20dFactor(),
        L.ReversalFactor(1), L.ReversalFactor(3), L.ReversalFactor(5),
        L.Rsi14Factor(), L.IndexAdjRev5dFactor(),
        L.RealizedVol20dFactor(), L.DownsideVol20dFactor(),
        L.Turnover20dAvgFactor(), L.Amihud20dFactor(), L.VolumeRatio5_20Factor(),
        L.AmountAccel5dFactor(), L.VolumePriceDiv20dFactor(), L.TurnoverPersistence10dFactor(),
        S.LimitUpCountFactor(), S.DaysSinceLimitUpFactor(),
        S.ConsecLimitUpDaysFactor(), S.NearLimitRateFactor(),
        L.MktBreadth5dFactor(), L.MktVol20dFactor(),
    ]
    for f in ready:
        reg.register(f, availability="ready")
    reg.register(L.EarningsQualityFactor(), availability="degraded_no_source")
    reg.register(L.ValuePEFactor(), availability="degraded_no_source")
    return reg
