"""pangu_ranker_v2 确认：等权 6 因子、N=10、20 日持有，BacktestV2 全成本精验。

来源：combo_search（90 组合网格）+ lowdim_ml（等权胜过 IC 加权与 LightGBM）。
本脚本用事件驱动引擎在真实约束（T+1/涨跌停/停牌/费用/容量）下复验，
分研究窗与 2025 OOS 段报告，并输出 +5% 触及率。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from engine.data.pit_store import PITStore  # noqa: E402
from engine.validation.backtest_v2 import (BacktestConfig,  # noqa: E402
                                           BacktestV2)
from engine.validation.experiment_registry import ExperimentRegistry  # noqa: E402

START, END = "2022-01-04", "2025-12-31"
OOS_START = "2025-01-02"
TOP_N, GROSS, HOLD = 10, 0.9, 20


class CachedStore:
    def __init__(self, inner, start, end):
        self._inner = inner
        self._panel = inner.daily_panel(start, end)
        self._index = inner.index_daily("sh.000300", start, end)
        self._full = inner.trading_days("1990-01-01", "2099-12-31")
        self._uni: dict = {}

    def daily_panel(self, start, end, symbols=None):
        m = self._panel.index.get_level_values("date")
        return self._panel[(m >= start) & (m <= end)]

    def index_daily(self, code, start, end):
        s = self._index["close"] if isinstance(self._index, pd.DataFrame) else self._index
        m = s.index
        return pd.DataFrame({"close": s[(m >= start) & (m <= end)]})

    def universe(self, date):
        if date not in self._uni:
            self._uni[date] = self._inner.universe(date)
        return self._uni[date]

    def trading_days(self, start, end):
        return [d for d in self._full
                if start.replace("-", "") <= d.replace("-", "") <= end.replace("-", "")]


class EqualWeight6:
    """等权 6 因子 Top10，每 20 交易日调仓（含缓冲：跌出 Top30 才卖）。"""

    def __init__(self, z: dict[str, pd.DataFrame], top_n=TOP_N, hold=HOLD,
                 keep_n=30):
        self.z = z
        self.top_n = top_n
        self.hold = hold
        self.keep_n = keep_n
        self._counter = 0
        self._held: set = set()

    def rebalance(self, decision_date, history):
        # 计数器独立于 history（HistoryView 禁止越界日历查询）
        self._counter += 1
        if (self._counter - 1) % self.hold != 0:
            return []
        frames = []
        for nm, dfz in self.z.items():
            if decision_date in dfz.index:
                frames.append(dfz.loc[decision_date].rank(pct=True))
        if len(frames) < 3:
            return []
        score = sum(frames) / len(frames)
        ranked = score.dropna().sort_values(ascending=False)
        if len(ranked) < self.top_n + 10:
            return []
        keep_set = set(ranked.index[: self.keep_n])
        buys = list(ranked.index[: self.top_n])
        held = [c for c in self._held if c in keep_set]
        final = list(dict.fromkeys(held + buys))[: self.keep_n]
        w = GROSS / len(final)
        targets = [{"symbol": c, "side": "BUY", "weight": w}
                   for c in final if c not in self._held]
        for c in self._held - set(final):
            targets.append({"symbol": c, "side": "SELL", "weight": 0})
        self._held = set(final)
        return targets


def main() -> None:
    t0 = time.time()
    store = PITStore()
    panel = store.daily_panel(START, END)
    from tools.lowdim_ml_ranker import factors_from
    z = factors_from(panel)
    cached = CachedStore(store, START, END)

    cfg = BacktestConfig(slippage_bps=10.0)
    bt = BacktestV2(cached, cfg)
    days = cached.trading_days(START, END)
    oos_days = [d for d in days if d >= OOS_START]

    results = {}
    segments = {"full": (days[80], days[-1])}
    if oos_days:
        segments["oos_2025"] = (oos_days[0], oos_days[-1])
    for label, (a, b) in segments.items():
        strat = EqualWeight6(z)
        res = bt.run(strat, a, b)
        s = res.summary
        daily = res.equity_curve["equity"].pct_change().dropna()
        # +5% 触及率（从成交明细：持有期最高 vs 入场价 +5%）
        hits = sum(1 for tr in res.trades
                   if tr.get("close_reason") == "profit_target"
                   or (tr.get("ret") or 0) >= 0.05)
        closed = sum(1 for tr in res.trades if tr.get("close_reason"))
        results[label] = {
            "total_return": s.get("total_return"), "sharpe": s.get("sharpe"),
            "profit_factor": s.get("profit_factor"),
            "max_drawdown": s.get("max_drawdown"),
            "n_trades": s.get("n_trades"), "execution_rate": s.get("execution_rate"),
            "turnover": s.get("turnover"),
            "win_rate_trades": s.get("win_rate_trades"),
            "target5_touch_proxy": round(hits / closed, 4) if closed else None,
        }
        print(label, json.dumps(results[label], default=str), flush=True)

    Path("data/experiments/strategies/pangu_ranker_v2_confirm.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8")

    reg = ExperimentRegistry()
    reg.register({
        "experiment_id": "ranker_v2_equal6_confirm",
        "hypothesis": "equal-weight 6-factor Top10/20d (winner of combo search + lowdim study) confirms OOS in event-driven engine with full costs",
        "economic_rationale": "reversal/low-vol/low-turnover/illiquidity/vpr/mom60-reversal blend",
        "data": "PITStore full archive", "pit_status": "strict asof; HistoryView guarded",
        "universe": "panel rows; liquidity floor by engine", "decision_time": "15:05 T",
        "execution_time": "T+1 open ±10bp, 20d rebalance with Top30 buffer",
        "features": list(z.keys()), "label": "realized PnL",
        "train_range": "combo search 2022→2024; OOS confirm 2025",
        "validation_range": "none",
        "test_range": f"full {START}→{END} and OOS 2025 separately",
        "costs": "3bp+stamp+transfer", "slippage": "10bp", "capacity": "2% ADV",
        "baseline": "combo_search screen hit 48.3% (2025 OOS Top10)",
        "parameters": {"top_n": TOP_N, "gross": GROSS, "hold": HOLD, "keep_n": 30},
        "optimization_method": "selected by prior studies (no refit here)",
        "n_variants_tried": 1,
        "metrics": results,
        "leakage_audit": {"weights_frozen": True},
        "independent_backtest": "event-driven engine vs screen-level numbers cross-check",
        "conclusion": _conclude(results),
        "status": "evaluated", "family": "ranker_validation"})
    print("registered", flush=True, )


def _conclude(results: dict) -> str:
    oos = results.get("oos_2025", {})
    if (oos.get("total_return") or 0) > 0 and (oos.get("profit_factor") or 0) > 1:
        return "oos_2025_positive_pf_above_1"
    if abs(oos.get("total_return") or 1) < 0.05:
        return "oos_2025_near_flat_after_full_costs"
    return "oos_2025_negative_after_costs"


if __name__ == "__main__":
    main()
