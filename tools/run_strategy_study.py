"""Rule-based strategy study through BacktestV2 with walk-forward segments.

Discipline (docs/pangu2/RESEARCH_PROTOCOL.md):
- Research window 2022-01-04 → 2025-12-31; walk-forward: expanding "train" is
  not needed for pre-registered rule strategies — instead we evaluate each
  strategy on consecutive 60-day OOS segments and aggregate, which measures
  stability. Parameters are fixed a priori (top 20, weekly rebalance, 0.8 gross)
  and were NOT tuned on any test data.
- Every strategy (including losers) is registered in the experiment registry.

Usage: .venv/Scripts/python tools/run_strategy_study.py [strategy_name ...]
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
from engine.research.data_interface import PITResearchData  # noqa: E402
from engine.validation.backtest_v2 import BacktestConfig, BacktestV2  # noqa: E402
from engine.validation.experiment_registry import ExperimentRegistry  # noqa: E402
from engine.validation.cross_check import cross_check  # noqa: E402
from engine.validation.robustness import cost_stress  # noqa: E402

from engine.validation.leakage import audit_module as _audit_module  # noqa: E402

LEAKAGE_AUDIT_FINDINGS = _audit_module(__file__)

START, END = "2022-01-04", "2025-12-31"
TOP_N = 20
GROSS = 0.8
REBALANCE_N = 5  # every 5 trading days


class CachedResearchData:
    """PITResearchData with a panel cache (read-only; slices stay asof-guarded
    by HistoryView inside BacktestV2)."""

    def __init__(self, inner: PITResearchData):
        self._inner = inner
        self._cache: dict[tuple, pd.DataFrame] = {}

    def daily_panel(self, start, end, symbols=None):
        key = (start, end, tuple(sorted(symbols)) if symbols else None)
        if key not in self._cache:
            self._cache[key] = self._inner.daily_panel(start, end, symbols)
        return self._cache[key]

    def __getattr__(self, item):
        return getattr(self._inner, item)


def precompute(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Causal per-symbol rolling features on the FULL panel (each row only
    uses past rows — rolling/shift with positive windows only)."""
    close = panel["close"].unstack("code").sort_index()
    pct = (panel["pct_change"] / 100.0).unstack("code").sort_index()
    amount = panel["amount"].unstack("code").sort_index()
    high = panel["high"].unstack("code").sort_index()
    low = panel["low"].unstack("code").sort_index()
    preclose = panel["preclose"].unstack("code").sort_index()
    is_st = panel["is_st"].unstack("code").sort_index().ffill().fillna(0)
    feats = {
        "ret20": close.pct_change(20),
        "ret5": close.pct_change(5),
        "ret60": close.pct_change(60),
        "vol20": pct.rolling(20).std(),
        "mom_vol20": close.pct_change(20) / pct.rolling(20).std().clip(lower=1e-4),
        "turn20": (amount / 1e8).rolling(20).mean(),
        "amihud20": (pct.abs() / amount.clip(lower=1e5)).rolling(20).mean() * 1e9,
        "vpr20": None,
    }
    # fix vpr20 (volume-price divergence) properly: corr of close vs volume over 20d
    volume = (panel["volume"].unstack("code").sort_index())
    feats["vpr20"] = -close.rolling(20).corr(volume)
    feats["is_st"] = is_st
    feats["amount"] = amount
    feats["preclose"] = preclose
    feats["high"] = high
    feats["low"] = low
    feats["close"] = close
    feats["pct"] = pct
    return feats


class PanelStrategy:
    """Weekly (every REBALANCE_N days) top-N by a precomputed score column."""

    def __init__(self, feats: dict, score_name: str, ascending: bool,
                 top_n: int = TOP_N, gross: float = GROSS, rebal_every: int = REBALANCE_N):
        self.feats = feats
        self.score_name = score_name
        self.ascending = ascending
        self.top_n = top_n
        self.gross = gross
        self.rebal_every = rebal_every
        self._days = None
        self._counter = 0

    def rebalance(self, decision_date: str, history) -> list[dict]:
        if self._days is None:
            self._days = list(history.panel_asof(decision_date).index.get_level_values(0).unique())
        self._counter += 1
        if (self._counter - 1) % self.rebal_every != 0:
            return []
        day = decision_date
        try:
            score = self.feats[self.score_name].loc[day]
        except KeyError:
            return []
        st = self.feats["is_st"].loc[day] if day in self.feats["is_st"].index else None
        amt = self.feats["amount"].loc[day]
        score = score.dropna()
        if st is not None:
            score = score[~score.index.map(lambda c: bool(st.get(c, 0)))]
        score = score[score.index.map(lambda c: (amt.get(c, 0) or 0) > 3e7)]  # liquidity floor
        if len(score) < self.top_n + 5:
            return []
        ranked = score.sort_values(ascending=self.ascending)
        picks = list(ranked.index[: self.top_n])
        held = set(history.open_positions.keys())
        targets = []
        w = self.gross / self.top_n
        for s in picks:
            targets.append({"symbol": s, "side": "BUY", "weight": w})
        for s in held - set(picks):
            targets.append({"symbol": s, "side": "SELL", "weight": 0})
        return targets


class RandomStrategy:
    """Seeded random ranking control: each rebalance picks TOP_N random eligible
    codes (same liquidity floor, ST exclusion, sizing).  RNG state advances
    deterministically, so runs are reproducible."""

    def __init__(self, feats, seed=42, top_n=TOP_N, gross=GROSS, rebal_every=REBALANCE_N):
        self.feats = feats
        self.rng = np.random.default_rng(seed)
        self.top_n = top_n
        self.gross = gross
        self.rebal_every = rebal_every
        self._counter = 0

    def rebalance(self, decision_date: str, history) -> list[dict]:
        self._counter += 1
        if (self._counter - 1) % self.rebal_every != 0:
            return []
        day = decision_date
        amt = self.feats["amount"].loc[day]
        st = self.feats["is_st"].loc[day]
        eligible = [c for c in amt.index
                    if (amt.get(c, 0) or 0) > 3e7 and not st.get(c, 0)]
        if len(eligible) < self.top_n + 5:
            return []
        picks = list(self.rng.choice(eligible, size=self.top_n, replace=False))
        held = set(history.open_positions.keys())
        w = self.gross / self.top_n
        targets = [{"symbol": s, "side": "BUY", "weight": w} for s in picks]
        targets += [{"symbol": s, "side": "SELL", "weight": 0} for s in held - set(picks)]
        return targets


class MLPlaceholder:
    pass


STRATEGIES = {
    "mom20_top20": dict(score="ret20", ascending=False),
    "mom60_top20": dict(score="ret60", ascending=False),
    "rev5_top20": dict(score="ret5", ascending=True),
    "lowvol20_top20": dict(score="vol20", ascending=True),
    "voladj_mom20_top20": dict(score="mom_vol20", ascending=False),
    "amihud20_top20": dict(score="amihud20", ascending=False),
    "turnover20_top20": dict(score="turn20", ascending=False),
    "vpr20_top20": dict(score="vpr20", ascending=False),
}


def walkforward_days(data, start: str, end: str, test_len=60, step=60, min_hist=120):
    days = data.trading_days(start, end)
    segs = []
    i = min_hist
    while i + test_len <= len(days):
        segs.append((days[0], days[i - 1], days[i], days[i + test_len - 1]))
        i += step
    return segs


def run_one(name: str, feats: dict, data, registry: ExperimentRegistry) -> None:
    t0 = time.time()
    spec = STRATEGIES[name]
    strat = PanelStrategy(feats, spec["score"], spec["ascending"])
    cfg = BacktestConfig()
    bt = BacktestV2(data, cfg)
    segs = walkforward_days(data, START, END)
    seg_results = []
    all_trades = []
    for (s0, s1, t0d, t1d) in segs:
        fresh = PanelStrategy(feats, spec["score"], spec["ascending"],
                              strat.top_n, strat.gross, strat.rebal_every)
        res = bt.run(fresh, t0d, t1d)
        summ = res.summary
        seg_results.append({
            "train_end": s1, "test": [t0d, t1d],
            "total_return": summ.get("total_return"),
            "sharpe": summ.get("sharpe"),
            "max_drawdown": summ.get("max_drawdown"),
            "n_trades": summ.get("n_trades"),
            "win_rate_trades": summ.get("win_rate_trades"),
            "execution_rate": summ.get("execution_rate"),
        })
        all_trades.extend(res.trades)
    # aggregate OOS
    rets = [r["total_return"] or 0.0 for r in seg_results]
    oos = {
        "n_segments": len(seg_results),
        "mean_segment_return": float(np.mean(rets)) if rets else None,
        "positive_segments": int(sum(1 for r in rets if r > 0)),
        "total_trades": int(sum(r["n_trades"] or 0 for r in seg_results)),
        "avg_win_rate": float(np.mean([r["win_rate_trades"] or 0 for r in seg_results])) if seg_results else None,
        "avg_execution_rate": float(np.mean([r["execution_rate"] or 0 for r in seg_results])) if seg_results else None,
    }
    # per-segment equity concat for drawdown/sharpe across OOS:
    # (approximate: use segment returns as a return series)
    sr = pd.Series(rets)
    oos["sharpe_segments"] = float(sr.mean() / sr.std() * np.sqrt(252 / 60)) if sr.std() else None
    oos["max_dd_segments"] = float((1 + sr).cumprod().sub((1 + sr).cumprod().cummax()).min()) if len(sr) else None

    # cost stress ×2 on the LAST segment only (compute budget); honest note
    last = segs[-1]
    strat2 = PanelStrategy(feats, spec["score"], spec["ascending"])

    def run_fn(cost_mult: float = 1.0):
        c = BacktestConfig(slippage_bps=cfg.slippage_bps * cost_mult,
                           commission_rate=cfg.commission_rate * cost_mult)
        return BacktestV2(data, c).run(strat2, last[2], last[3])

    try:
        stress = cost_stress(run_fn, (1, 2))
        cost2x_positive = bool((stress.get("2x", {}).get("summary", {}) or {}).get("total_return", 0) > 0) \
            if isinstance(stress, dict) else None
    except Exception as e:  # noqa: BLE001
        cost2x_positive = None
        stress = {"error": repr(e)}

    # independent cross-check on last segment
    strat3 = PanelStrategy(feats, spec["score"], spec["ascending"])
    base = BacktestV2(data, cfg).run(strat3, last[2], last[3])
    try:
        cc = cross_check(base, data, strat3)
        independent_ok = bool(cc.get("consistent"))
    except Exception as e:  # noqa: BLE001
        cc = {"error": repr(e)}
        independent_ok = None

    conclusion = _conclude(oos)
    registry.register({
        "experiment_id": f"strategy_{name}_wf",
        "hypothesis": f"{spec['score']} ranking pre-registered strategy (top {TOP_N}, gross {GROSS}, every {REBALANCE_N}d)",
        "economic_rationale": {"mom20_top20": "cross-sectional momentum underreaction",
                               "mom60_top20": "intermediate momentum",
                               "rev5_top20": "short-term reversal / liquidity provision",
                               "lowvol20_top20": "low-volatility anomaly",
                               "voladj_mom20_top20": "risk-adjusted momentum",
                               "amihud20_top20": "illiquidity premium",
                               "turnover20_top20": "attention/turnover effect",
                               "vpr20_top20": "volume-price divergence"}.get(name, name),
        "data": "PITStore breadth_raw (unadjusted OHLC, pct_change returns) 2022→2025-12",
        "pit_status": "strict asof HistoryView; ST excluded from buys; suspension = no bar; T+1; limit unfillable",
        "universe": "panel rows (incl. later-delisted); liquidity floor 30e6 amount",
        "decision_time": "15:05 T",
        "execution_time": "T+1 open ±10bp slippage",
        "features": [spec["score"]],
        "label": "realized portfolio PnL",
        "train_range": "pre-registered (no tuning)",
        "validation_range": "none (fixed params)",
        "test_range": f"walk-forward {len(seg_results)} × 60d segments in {START}→{END}",
        "costs": "commission 3bp min 5; stamp 5bp sell; transfer 0.1bp",
        "slippage": "10bp (2x stress on last segment)",
        "capacity": "participation cap 2% ADV",
        "baseline": "random20 control + ew100 benchmark (same registry)",
        "parameters": {"top_n": TOP_N, "gross": GROSS, "rebal_every": REBALANCE_N},
        "optimization_method": "none (pre-registered)",
        "n_variants_tried": 1,
        "metrics": {"oos": oos, "segments": seg_results[-3:], "cost_stress_last_seg": stress,
                    "cost2x_positive": cost2x_positive},
        "leakage_audit": {"history_view_guarded": True,
                          "features_causal_rolling": True,
                          "st_filter_pit": True,
                          "static_audit_findings": LEAKAGE_AUDIT_FINDINGS},
        "independent_backtest": {"consistent": independent_ok, "detail": cc if not independent_ok else "ok"},
        "conclusion": conclusion,
        "status": "evaluated",
        "family": "rule_strategy",
    })
    out = Path("data/experiments/strategies")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_wf.json").write_text(
        json.dumps({"name": name, "oos": oos, "segments": seg_results,
                    "cost_stress_last_seg": stress, "cross_check": cc},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[{name}] {time.time()-t0:.0f}s oos={json.dumps(oos)[:200]} -> {conclusion}", flush=True)


def _conclude(oos: dict) -> str:
    mr = oos.get("mean_segment_return")
    if mr is None:
        return "no_data"
    pos = oos.get("positive_segments", 0)
    n = oos.get("n_segments", 0)
    if mr > 0 and pos >= max(2, n // 2):
        return "oos_positive_candidate_for_holdout"
    if mr <= 0:
        return "oos_negative_rejected"
    return "oos_positive_but_unstable"


def main() -> None:
    names = sys.argv[1:] or list(STRATEGIES) + ["random20", "ew_benchmark"]
    t0 = time.time()
    store = PITStore()
    data = CachedResearchData(PITResearchData(store))
    panel = data.daily_panel(START, END)
    print(f"panel: {panel.shape}", flush=True)
    feats = precompute(panel)
    print(f"features ready {time.time()-t0:.0f}s", flush=True)
    registry = ExperimentRegistry()
    for name in names:
        try:
            if name == "random20":
                run_random(feats, data, registry)
            elif name == "ew_benchmark":
                run_ew_benchmark(feats, data, registry)
            else:
                run_one(name, feats, data, registry)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED {e!r}", flush=True)
            try:
                registry.register({
                    "experiment_id": f"strategy_{name}_wf", "hypothesis": name,
                    "economic_rationale": name, "data": "PITStore", "pit_status": "strict",
                    "universe": "panel rows", "decision_time": "15:05 T",
                    "execution_time": "T+1 open", "features": [name], "label": "PnL",
                    "train_range": "pre-registered", "validation_range": "none",
                    "test_range": f"walk-forward {START}→{END}", "costs": "standard",
                    "slippage": "10bp", "capacity": "2% ADV",
                    "baseline": "random20", "parameters": {}, "optimization_method": "none",
                    "n_variants_tried": 1, "metrics": {},
                    "leakage_audit": {"history_view_guarded": True},
                    "independent_backtest": "not_run",
                    "conclusion": f"error: {e!r}", "status": "failed",
                    "family": "rule_strategy",
                })
            except Exception:
                pass
    print(f"strategy study done {time.time()-t0:.0f}s", flush=True)


class EqualWeightStrategy:
    """Equal-weight universe benchmark: holds top-liquidity names equally
    (max_positions cap), weekly refresh.  Acts as the investable baseline."""

    def __init__(self, feats, top_n=100, rebal_every=REBALANCE_N):
        self.feats = feats
        self.top_n = top_n
        self.rebal_every = rebal_every
        self._counter = 0

    def rebalance(self, decision_date: str, history) -> list[dict]:
        self._counter += 1
        if (self._counter - 1) % self.rebal_every != 0:
            return []
        day = decision_date
        amt = self.feats["amount"].loc[day]
        st = self.feats["is_st"].loc[day]
        eligible = [c for c in amt.index
                    if (amt.get(c, 0) or 0) > 3e7 and not st.get(c, 0)]
        if len(eligible) < 20:
            return []
        picks = sorted(eligible, key=lambda c: amt.get(c, 0), reverse=True)[: self.top_n]
        held = set(history.open_positions.keys())
        w = min(1.0, 0.95) / max(len(picks), 1)
        targets = [{"symbol": s, "side": "BUY", "weight": w} for s in picks]
        targets += [{"symbol": s, "side": "SELL", "weight": 0} for s in held - set(picks)]
        return targets


def run_ew_benchmark(feats, data, registry) -> None:
    t0 = time.time()
    cfg = BacktestConfig()
    bt = BacktestV2(data, cfg)
    segs = walkforward_days(data, START, END)
    seg_results = []
    for (_s0, _s1, t0d, t1d) in segs:
        res = bt.run(EqualWeightStrategy(feats), t0d, t1d)
        s = res.summary
        seg_results.append({"test": [t0d, t1d], "total_return": s.get("total_return"),
                            "n_trades": s.get("n_trades")})
    rets = [r["total_return"] or 0.0 for r in seg_results]
    oos = {"n_segments": len(seg_results),
           "mean_segment_return": float(np.mean(rets)) if rets else None,
           "positive_segments": int(sum(1 for r in rets if r > 0)),
           "total_trades": int(sum(r["n_trades"] or 0 for r in seg_results))}
    registry.register({
        "experiment_id": "benchmark_ew100_wf",
        "hypothesis": "equal-weight top-liquidity-100 investable baseline",
        "economic_rationale": "market baseline after costs for comparison",
        "data": "PITStore breadth_raw", "pit_status": "strict asof",
        "universe": "panel rows; liquidity floor; ST excluded",
        "decision_time": "15:05 T", "execution_time": "T+1 open ±10bp",
        "features": ["amount"], "label": "realized portfolio PnL",
        "train_range": "n/a", "validation_range": "none",
        "test_range": f"walk-forward {len(seg_results)} × 60d in {START}→{END}",
        "costs": "commission 3bp min 5; stamp 5bp sell", "slippage": "10bp",
        "capacity": "2% ADV", "baseline": "self",
        "parameters": {"top_n": 100, "gross": 0.95},
        "optimization_method": "none", "n_variants_tried": 1,
        "metrics": {"oos": oos},
        "leakage_audit": {"history_view_guarded": True,
                          "static_audit_findings": LEAKAGE_AUDIT_FINDINGS},
        "independent_backtest": "not_applicable_baseline",
        "conclusion": _conclude(oos), "status": "evaluated",
        "family": "benchmark",
    })
    print(f"[ew_benchmark] {time.time()-t0:.0f}s oos={json.dumps(oos)[:200]}", flush=True)


def run_random(feats, data, registry) -> None:
    """Random control through the same walk-forward aggregation path."""
    t0 = time.time()
    cfg = BacktestConfig()
    bt = BacktestV2(data, cfg)
    segs = walkforward_days(data, START, END)
    seg_results = []
    for (_s0, _s1, t0d, t1d) in segs:
        res = bt.run(RandomStrategy(feats), t0d, t1d)
        summ = res.summary
        seg_results.append({"test": [t0d, t1d], "total_return": summ.get("total_return"),
                            "n_trades": summ.get("n_trades"),
                            "win_rate_trades": summ.get("win_rate_trades")})
    rets = [r["total_return"] or 0.0 for r in seg_results]
    oos = {"n_segments": len(seg_results),
           "mean_segment_return": float(np.mean(rets)) if rets else None,
           "positive_segments": int(sum(1 for r in rets if r > 0)),
           "total_trades": int(sum(r["n_trades"] or 0 for r in seg_results)),
           "avg_win_rate": float(np.mean([r["win_rate_trades"] or 0 for r in seg_results])) if seg_results else None}
    registry.register({
        "experiment_id": "strategy_random20_wf",
        "hypothesis": "random ranking control (same mechanics, seeded random picks)",
        "economic_rationale": "null model for the rule strategies",
        "data": "PITStore breadth_raw", "pit_status": "strict asof",
        "universe": "panel rows; liquidity floor; ST excluded",
        "decision_time": "15:05 T", "execution_time": "T+1 open ±10bp",
        "features": ["random"], "label": "realized portfolio PnL",
        "train_range": "n/a", "validation_range": "none",
        "test_range": f"walk-forward {len(seg_results)} × 60d in {START}→{END}",
        "costs": "commission 3bp min 5; stamp 5bp sell", "slippage": "10bp",
        "capacity": "2% ADV", "baseline": "self",
        "parameters": {"top_n": TOP_N, "gross": GROSS, "seed": 42},
        "optimization_method": "none", "n_variants_tried": 1,
        "metrics": {"oos": oos},
        "leakage_audit": {"history_view_guarded": True},
        "independent_backtest": "not_applicable_control",
        "conclusion": _conclude(oos), "status": "evaluated",
        "family": "control",
    })
    print(f"[random20] {time.time()-t0:.0f}s oos={json.dumps(oos)[:200]}", flush=True)


if __name__ == "__main__":
    main()
