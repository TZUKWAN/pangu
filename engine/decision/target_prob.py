"""+5% 目标命中概率（用户核心诉求的诚实工程化）。

用户目标："推荐周期结束卖出时收益率 ≥5%"。任何诚实系统都无法**保证**该收益，
但可以回答一个可验证的问题：**与当前候选同画像的历史 setup，在同等持有窗口、
同等止损纪律下，触及 +5% 目标的频率是多少？**（OOS、含 Wilson 95% 置信下界）

- 入场口径：tail_close（T 日收盘，尾盘决策）或 next_open（T+1 开盘，盘后决策）
- 触达判定：持有窗内最高价 ≥ entry×(1+target)；止损判定：最低价 ≤ entry×(1-stop)；
  聚合窗口下无法分辨同 bar 内先后 → **保守计为止损优先**（stop_first）
- 条件维度：regime × 反转z分位（全部仅用决策日已知信息）
- 每格样本 ≥ min_n 才可信，否则沿 (regime|rev_all) → (all|rev_all) 父格回退
- 大规模构建见 tools/build_target_table.py（向量化）；本模块负责容器/查询/持久化
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_TARGET_PCT = 0.05
DEFAULT_STOP_PCT = 0.05
DEFAULT_HORIZON = 5
DEFAULT_MIN_N = 30
TABLE_PATH = Path("data/decision/target_hit_table.json")
BUY_MIN_WILSON_LB = 0.30      # BUY 要求命中频率的 95% 置信下界 ≥ 30%
BUY_MIN_N = 30


def wilson_lb(hits: int, n: int, z: float = 1.96) -> float:
    """Wilson 置信下界（默认 95%）。n=0 → 0。"""
    if n <= 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def outcome_for_bars(entry: float, highs: np.ndarray, lows: np.ndarray,
                     target_pct: float, stop_pct: float) -> Tuple[str, float]:
    """单票单次决策结果：hit / stopped / timeout（stop_first 保守序）。"""
    if entry is None or not np.isfinite(entry) or entry <= 0:
        return "invalid", 0.0
    tgt = entry * (1 + target_pct)
    stp = entry * (1 - stop_pct)
    for hi, lo in zip(highs, lows):
        if lo <= stp:
            return "stopped", -stop_pct
        if hi >= tgt:
            return "hit", target_pct
    return "timeout", 0.0


def rev_bucket(rev_z: Optional[float]) -> str:
    """反转 z 分位桶（z 越大越超跌）。"""
    if rev_z is None or not np.isfinite(rev_z):
        return "rev_na"
    if rev_z < -1.0:
        return "rev_q1"
    if rev_z < 0.0:
        return "rev_q2"
    if rev_z < 1.0:
        return "rev_q3"
    if rev_z < 2.0:
        return "rev_q4"
    return "rev_q5"


def vol_bucket(vol20: Optional[float]) -> str:
    if vol20 is None or not np.isfinite(vol20):
        return "vol_na"
    if vol20 < 0.02:
        return "vol_low"
    if vol20 < 0.035:
        return "vol_mid"
    return "vol_high"


@dataclass
class HitEstimate:
    hits: int
    n: int
    rate: float
    wilson_lb: float
    cell: str
    fallback: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def buy_eligible(self) -> bool:
        """是否满足"证据支持 ≥5% 目标"的 BUY 证据线。"""
        return (not self.fallback) and self.n >= BUY_MIN_N \
            and self.wilson_lb >= BUY_MIN_WILSON_LB


class TargetHitTable:
    """条件命中表：cell 字符串 → [hits, n]，带父格回退查询。"""

    def __init__(self, cells: Optional[Dict[str, List[int]]] = None,
                 min_n: int = DEFAULT_MIN_N,
                 target_pct: float = DEFAULT_TARGET_PCT,
                 stop_pct: float = DEFAULT_STOP_PCT,
                 horizon: int = DEFAULT_HORIZON, built_range: str = ""):
        self.cells = cells or {}
        self.min_n = min_n
        self.target_pct = target_pct
        self.stop_pct = stop_pct
        self.horizon = horizon
        self.built_range = built_range

    # -- 小规模构建（测试用；生产构建走 tools/build_target_table.py 向量化） -- #
    def build(self, panel: pd.DataFrame, rev_z: pd.Series,
              regime_of: Dict[str, str], decision_dates: List[str],
              entry_style: str = "next_open",
              horizon: Optional[int] = None) -> None:
        h = horizon or self.horizon
        close = panel["close"].unstack("code").sort_index()
        open_ = panel["open"].unstack("code").sort_index()
        high = panel["high"].unstack("code").sort_index()
        low = panel["low"].unstack("code").sort_index()
        dates = list(close.index)
        pos = {d: i for i, d in enumerate(dates)}
        for day in decision_dates:
            i = pos.get(day)
            if i is None or i + h >= len(dates):
                continue
            regime = regime_of.get(day, "neutral")
            entries = close.iloc[i] if entry_style == "tail_close" \
                else open_.iloc[i + 1]
            for code in close.columns:
                entry = entries.get(code)
                if entry is None or not np.isfinite(entry):
                    continue
                rz = None
                try:
                    rz = float(rev_z.loc[(day, code)])
                except (KeyError, TypeError, ValueError):
                    pass
                highs = high[code].values[i + 1:i + 1 + h]
                lows = low[code].values[i + 1:i + 1 + h]
                res, _ = outcome_for_bars(float(entry), highs, lows,
                                          self.target_pct, self.stop_pct)
                if res == "invalid":
                    continue
                cell = f"{regime}|{rev_bucket(rz)}"
                c = self.cells.setdefault(cell, [0, 0])
                c[1] += 1
                if res == "hit":
                    c[0] += 1

    # -- 查询 ------------------------------------------------------------ #
    def lookup(self, regime: str, rev_z) -> HitEstimate:
        rb = rev_z if isinstance(rev_z, str) and rev_z.startswith("rev_")             else rev_bucket(rev_z)
        exact = f"{regime}|{rb}"
        for cell in (exact, f"{regime}|rev_all", "all|rev_all"):
            hits, n = self.cells.get(cell, [0, 0])
            if n >= self.min_n:
                return HitEstimate(hits=hits, n=n,
                                   rate=round(hits / n, 4),
                                   wilson_lb=round(wilson_lb(hits, n), 4),
                                   cell=cell, fallback=(cell != exact))
        hits, n = self.cells.get(exact, [0, 0])
        return HitEstimate(hits=hits, n=n,
                           rate=round(hits / n, 4) if n else 0.0,
                           wilson_lb=0.0, cell=exact, fallback=True)

    def merge_parent_cells(self) -> None:
        """构建完成后由工具调用：把 (regime|rev_q*) 汇总出 (regime|rev_all)
        与 (all|rev_all) 父格，供回退查询。"""
        per_regime: Dict[str, List[int]] = {}
        all_cell = [0, 0]
        for cell, (hits, n) in self.cells.items():
            if cell.endswith("|rev_all"):
                continue
            regime = cell.split("|")[0]
            c = per_regime.setdefault(f"{regime}|rev_all", [0, 0])
            c[0] += hits
            c[1] += n
            all_cell[0] += hits
            all_cell[1] += n
        for k, v in per_regime.items():
            self.cells[k] = v
        self.cells["all|rev_all"] = all_cell

    # -- 持久化 ---------------------------------------------------------- #
    def save(self, path: Path = TABLE_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "cells": self.cells, "min_n": self.min_n,
            "target_pct": self.target_pct, "stop_pct": self.stop_pct,
            "horizon": self.horizon, "built_range": self.built_range,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = TABLE_PATH) -> "TargetHitTable":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return cls(cells=d.get("cells", {}), min_n=d.get("min_n", DEFAULT_MIN_N),
                       target_pct=d.get("target_pct", DEFAULT_TARGET_PCT),
                       stop_pct=d.get("stop_pct", DEFAULT_STOP_PCT),
                       horizon=d.get("horizon", DEFAULT_HORIZON),
                       built_range=d.get("built_range", ""))
        except (json.JSONDecodeError, OSError):
            return cls()
