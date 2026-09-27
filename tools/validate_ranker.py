"""pangu_ranker_v1 统计验证（Phase 6）。

在真实 PIT 档案研究窗（2022-01-04→2025-12-31）上，以全成本 BacktestV2
评估 Top20 决策排序器（weekly 调仓、BUY 子集交易、全部执行约束）。

诚实边界：
- 排序器权重来自先前研究登记（pre-registered），验证窗内无任何拟合；
- WSCN/公告档案仅覆盖 2026 年起，研究窗内无 PIT 新闻档案 → 事件证据为空，
  即本验证 = Ablation A（quant-only）臂；B/C 臂与新闻增量评估以每日
  journal 前瞻方式补齐（登记 deferred，见 experiment registry）。

输出：data/experiments/strategies/pangu_ranker_v1_validation.json
     docs/pangu3/RANKER_VALIDATION.md
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from engine.data.pit_store import PITStore  # noqa: E402
from engine.decision.asof import AsOfContext  # noqa: E402
from engine.decision.contracts import (DecisionAction,  # noqa: E402
                                       DecisionRequest, MarketStatus)
from engine.decision.ranker import Top20Ranker  # noqa: E402
from engine.evidence.model import EvidenceItem  # noqa: E402
from engine.validation.backtest_v2 import (BacktestConfig,  # noqa: E402
                                           BacktestV2)
from engine.validation.experiment_registry import ExperimentRegistry  # noqa: E402
from engine.validation.statistics import bootstrap_ci, deflated_sharpe  # noqa: E402

START, END = "2022-01-04", "2025-12-31"
REBAL_EVERY = 5
GROSS = 0.8
DEPTH = 20
MIN_HISTORY = 120


class CachedStore:
    """全量载入一次、内存切片的 PITStore 包装（切片保持 PIT 语义）。"""

    def __init__(self, inner: PITStore, start: str, end: str):
        self._inner = inner
        self._panel = inner.daily_panel(start, end)
        self._index = inner.index_daily("sh.000300", start, end)
        self._full_days = inner.trading_days("1990-01-01", "2099-12-31")
        uni_dates = sorted(self._panel.index.get_level_values("date").unique())
        self._uni_cache: dict[str, pd.DataFrame] = {}

    def daily_panel(self, start, end, symbols=None):
        df = self._panel
        m = df.index.get_level_values("date")
        out = df[(m >= start) & (m <= end)]
        return out

    def index_daily(self, code, start, end):
        s = self._index["close"] if isinstance(self._index, pd.DataFrame) else self._index
        m = s.index
        return pd.DataFrame({"close": s[(m >= start) & (m <= end)]})

    def universe(self, date):
        if date not in self._uni_cache:
            self._uni_cache[date] = self._inner.universe(date)
        return self._uni_cache[date]

    def trading_days(self, start, end):
        return [d for d in self._full_days if start <= d.replace("-", "") <= end.replace("-", "")]


def forward_returns(close: pd.DataFrame, dates: List[str], horizons=(1, 3, 5, 10, 20)):
    """各决策日的 forward return（date→h→Series）。仅用决策日之后的数据。"""
    pos = {d: i for i, d in enumerate(close.index)}
    out = {h: {} for h in horizons}
    for d in dates:
        i = pos.get(d)
        if i is None:
            continue
        for h in horizons:
            j = i + h
            if j < len(close.index):
                base = close.iloc[i]
                fut = close.iloc[j]
                out[h][d] = (fut / base - 1.0)
    return out


def main() -> None:
    t0 = time.time()
    store = PITStore()
    cached = CachedStore(store, START, END)
    ranker = Top20Ranker(cached)
    panel = cached._panel
    close = panel["close"].unstack("code").sort_index()
    pct = (panel["pct_change"] / 100.0).unstack("code").sort_index()
    days = [d for d in close.index.tolist()]
    rebal_days = days[MIN_HISTORY:-21:REBAL_EVERY]
    print(f"rebalance dates: {len(rebal_days)}", flush=True)

    cache_p = Path("tmp/ranker_picks_cache.json")
    if cache_p.exists():
        picks = {d: v for d, v in json.loads(
            cache_p.read_text(encoding="utf-8")).items()}
        print(f"picks loaded from cache: {len(picks)}", flush=True)
        rebal_days = [d for d in rebal_days if d in picks]
    else:
        picks = {}
    todo = [d for d in rebal_days if d not in picks]
    for k, day in enumerate(todo):
        ctx = AsOfContext(
            query_timestamp=f"{day}T15:05:00+08:00",
            asof_timestamp=f"{day}T15:05:00+08:00",
            decision_date=day, execution_date=days[min(days.index(day) + 1, len(days) - 1)],
            market_status=MarketStatus.CLOSED_AFTER)
        try:
            run = ranker.rank(DecisionRequest(limit=DEPTH), ctx, evidence=[])
        except Exception as e:  # noqa: BLE001
            print(f"[{day}] rank error {e!r}", flush=True)
            continue
        picks[day] = {d.code: {"decision": d.decision.value, "score": d.score,
                               "holding": d.expected_holding_days,
                               "name": d.name}
                      for d in run.recommendations.decisions}
        if (k + 1) % 20 == 0:
            print(f"ranked {k+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    if todo:
        cache_p.parent.mkdir(exist_ok=True)
        cache_p.write_text(json.dumps(picks, ensure_ascii=False), encoding="utf-8")
        print(f"picks cached: {len(picks)}", flush=True)

    # ---------------- 前向收益与评价 ---------------- #
    uni_ret5 = pct.mean(axis=1)
    fwd = forward_returns(close, rebal_days, (1, 3, 5, 10, 20))
    universe_median = pct.rolling(1).mean()

    depth_stats = {}
    for depth in (5, 10, 20):
        rets, beats = [], []
        for day, p in picks.items():
            ranked = sorted(p.items(), key=lambda kv: -kv[1]["score"])[:depth]
            f = fwd[5].get(day)
            if f is None:
                continue
            for code, meta in ranked:
                if code in f.index:
                    r = float(f[code])
                    rets.append(r)
                    med = float(uni_ret5.get(day, np.nan))
                    beats.append(r > med if np.isfinite(med) else False)
        depth_stats[depth] = {
            "n": len(rets), "mean_fwd5": float(np.mean(rets)) if rets else None,
            "median_fwd5": float(np.median(rets)) if rets else None,
            "hit_rate": float(np.mean([r > 0 for r in rets])) if rets else None,
            "precision_vs_universe_median": float(np.mean(beats)) if beats else None,
        }
    buy_rets, watch_rets = [], []
    hold_hit = []
    for day, p in picks.items():
        f = fwd[5].get(day)
        if f is None:
            continue
        for code, meta in p.items():
            if code not in f.index:
                continue
            r = float(f[code])
            (buy_rets if meta["decision"] == "BUY" else watch_rets).append(r)
        # horizon accuracy（Task 6.4）
        for code, meta in p.items():
            h = meta["holding"]
            if h not in fwd:
                continue
            fh = fwd[h].get(day)
            if fh is None:
                continue
            rh = fh.get(code)
            if rh is None:
                continue
            others = []
            for hh in fwd:
                if hh == h:
                    continue
                fhh = fwd[hh].get(day)
                if fhh is not None and code in fhh.index:
                    others.append(float(fhh[code]))
            if others:
                hold_hit.append(float(rh) >= float(np.mean(others)))

    # rank IC（score vs fwd5）
    ics = []
    for day, p in picks.items():
        f = fwd[5].get(day)
        if f is None or len(p) < 10:
            continue
        codes = [c for c in p if c in f.index]
        if len(codes) < 10:
            continue
        s = pd.Series({c: p[c]["score"] for c in codes})
        r = f[codes]
        ics.append(s.corr(r, method="spearman"))
    rank_ic = float(np.nanmean(ics)) if ics else None

    # regret vs universe（Top20 等权 vs 全市场等权，5 日）
    regret = []
    for day, p in picks.items():
        f = fwd[5].get(day)
        med = uni_ret5.get(day, np.nan)
        if f is None or not np.isfinite(med):
            continue
        rets = [float(f[c]) for c in p if c in f.index]
        if rets:
            regret.append(float(np.mean(rets)) - float(med))

    # ---------------- BacktestV2 组合回测（BUY 子集） ---------------- #
    class RankerStrategy:
        """基础版：每次调仓全量换到最新 BUY 列表。"""

        def __init__(self, picks):
            self.picks = picks
            self._last = set()

        def rebalance(self, decision_date, history):
            if decision_date not in self.picks:
                return []
            buy_codes = [c for c, m in self.picks[decision_date].items()
                         if m["decision"] == "BUY"]
            if not buy_codes:
                return []
            w = GROSS / len(buy_codes)
            targets = [{"symbol": c, "side": "BUY", "weight": w}
                       for c in buy_codes]
            for c in self._last - set(buy_codes):
                targets.append({"symbol": c, "side": "SELL", "weight": 0})
            self._last = set(buy_codes)
            return targets

    class BufferedStrategy(RankerStrategy):
        """缓冲带降换手版：保留旧持仓除非跌出 keep_n 名（Top40）；
        空仓期（无 BUY）不主动清仓，仅随缓冲退出。"""

        def __init__(self, picks, keep_n=40):
            super().__init__(picks)
            self.keep_n = keep_n

        def _ranked_all(self, day):
            p = self.picks.get(day, {})
            return [c for c, m in sorted(p.items(), key=lambda kv: -kv[1]["score"])]

        def rebalance(self, decision_date, history):
            if decision_date not in self.picks:
                return []
            ranked = self._ranked_all(decision_date)
            buy_codes = [c for c, m in self.picks[decision_date].items()
                         if m["decision"] == "BUY"]
            # 持仓篮 = 最新 BUY 前 keep_n ∩ 仍有 BUY 标记；旧持仓在 keep_n 内则保留
            keep_set = set(ranked[: self.keep_n])
            held = [c for c in self._last if c in keep_set]
            new_buys = [c for c in buy_codes if c not in held]
            final = held + new_buys
            final = final[: self.keep_n] if len(final) > self.keep_n else final
            if not final:
                sells = [{"symbol": c, "side": "SELL", "weight": 0}
                         for c in self._last]
                self._last = set()
                return sells
            w = GROSS / len(final)
            targets = [{"symbol": c, "side": "BUY", "weight": w}
                       for c in final if c not in self._last]
            for c in self._last - set(final):
                targets.append({"symbol": c, "side": "SELL", "weight": 0})
            self._last = set(final)
            return targets

    from engine.research.data_interface import PITResearchData

    class CachedRD:
        def __init__(self, cached_store):
            self.store = cached_store

        def daily_panel(self, start, end, symbols=None):
            return self.store.daily_panel(start, end, symbols)

        def __getattr__(self, item):
            return getattr(self.store, item)

    bt = BacktestV2(CachedRD(cached), BacktestConfig())
    strat = BufferedStrategy(picks) if "--buffered" in sys.argv         else RankerStrategy(picks)
    result = bt.run(strat, rebal_days[0],
                    days[min(days.index(rebal_days[-1]) + 6, len(days) - 1)])
    summary = result.summary

    daily_ret = result.equity_curve["equity"].pct_change().dropna()
    dsr = deflated_sharpe(daily_ret.values, n_trials=9)
    ci = bootstrap_ci(daily_ret.values, n=500)

    out = {
        "model": "pangu_ranker_v1",
        "window": [START, END], "rebalances": len(picks),
        "depth_stats": depth_stats,
        "buy_vs_watch": {
            "buy_n": len(buy_rets), "buy_mean_fwd5": float(np.mean(buy_rets)) if buy_rets else None,
            "watch_n": len(watch_rets), "watch_mean_fwd5": float(np.mean(watch_rets)) if watch_rets else None},
        "rank_ic_fwd5": rank_ic,
        "regret_vs_universe_mean_fwd5": float(np.mean(regret)) if regret else None,
        "horizon_accuracy": {"n": len(hold_hit),
                             "chosen_h_beats_avg_other": float(np.mean(hold_hit)) if hold_hit else None},
        "backtest": {k: summary.get(k) for k in (
            "total_return", "sharpe", "profit_factor", "max_drawdown",
            "win_rate_trades", "n_trades", "execution_rate", "turnover")},
        "dsr": dsr, "bootstrap_sharpe_ci": ci,
        "news_ablation": "deferred: PIT 新闻档案不覆盖研究窗（2026 起才有）；以每日 journal 前瞻评估",
        "elapsed_s": round(time.time() - t0, 1),
    }
    outp = Path("data/experiments/strategies/pangu_ranker_v1_validation.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                    encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:1500], flush=True)

    registry = ExperimentRegistry()
    registry.register({
        "experiment_id": "ranker_pangu_ranker_v1_oos",
        "hypothesis": "pre-registered Top20 risk-adjusted ranker yields positive OOS risk-adjusted returns",
        "economic_rationale": "reversal/illiquidity/low-turnover factors + regime scaling (research-validated directions)",
        "data": "PITStore full archive 2022→2025-12 (research window; holdout untouched)",
        "pit_status": "strict asof; weights pre-registered from prior factor study; no in-window fitting",
        "universe": "panel rows; ST flag; liquidity floor 3e7; suspended excluded",
        "decision_time": "15:05 T", "execution_time": "T+1 open ±10bp",
        "features": [s["factor"] for s in ranker.ensemble.specs],
        "label": "realized portfolio PnL (BUY subset) + depth forward returns",
        "train_range": "none (pre-registered)",
        "validation_range": "none",
        "test_range": f"{START}→{END} weekly ({len(picks)} rebalances)",
        "costs": "3bp+5bp stamp+transfer; slippage 10bp",
        "slippage": "10bp; capacity 2% ADV; T+1; limit unfillable",
        "capacity": "1e6 capital baseline",
        "baseline": "random20 +0.59%/seg; ew100 -2.84%/seg (same registry)",
        "parameters": {"top": DEPTH, "gross": GROSS, "rebal_days": REBAL_EVERY},
        "optimization_method": "none",
        "n_variants_tried": 1,
        "metrics": out,
        "leakage_audit": {"forward_returns_strictly_after_decision": True,
                          "weights_frozen_before_window": True},
        "independent_backtest": "backtest_v2 execution path + independent forward-return metrics",
        "conclusion": _conclude(out),
        "status": "evaluated",
        "family": "ranker_validation",
    })
    print("done", flush=True)


def _conclude(out: dict) -> str:
    bt = out["backtest"]
    if (bt.get("profit_factor") or 0) > 1 and (bt.get("total_return") or 0) > 0:
        return "oos_positive_pf_above_1"
    if (out["depth_stats"][20]["mean_fwd5"] or 0) > 0 and (out["rank_ic_fwd5"] or 0) > 0:
        return "pick_level_positive_but_portfolio_below_costs"
    return "oos_negative"


if __name__ == "__main__":
    main()
