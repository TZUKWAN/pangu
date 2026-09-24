"""ML strategy study: Alpha158 + LightGBM walk-forward through BacktestV2.

Splits mirror tools/run_strategy_study.py: consecutive 60-trading-day test
segments over the research window; each model trains ONLY on data before its
test segment (embargo 5 trading days), predicts the segment, and the top-20
prediction portfolio is backtested with full cost/limit/T+1 semantics.
Every run is registered, including failures.
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
from engine.validation.leakage import audit_module as _audit_module  # noqa: E402

import qlib  # noqa: E402
from qlib.constant import REG_CN  # noqa: E402
from qlib.data import D  # noqa: E402
from qlib.contrib.data.handler import Alpha158  # noqa: E402

START, END = "2022-01-04", "2025-12-31"
TOP_N, GROSS, REBAL_EVERY, EMBARGO = 20, 0.8, 5, 5


class CachedResearchData:
    def __init__(self, inner):
        self._inner = inner
        self._cache = {}

    def daily_panel(self, start, end, symbols=None):
        key = (start, end, tuple(sorted(symbols)) if symbols else None)
        if key not in self._cache:
            self._cache[key] = self._inner.daily_panel(start, end, symbols)
        return self._cache[key]

    def __getattr__(self, item):
        return getattr(self._inner, item)


class PredStrategy:
    """Top-N by precomputed prediction frame (date × code); only rebalances on
    days that have predictions."""

    def __init__(self, preds: pd.DataFrame, top_n=TOP_N, gross=GROSS, rebal_every=REBAL_EVERY):
        self.preds = preds
        self.top_n = top_n
        self.gross = gross
        self.rebal_every = rebal_every
        self._counter = 0

    def rebalance(self, decision_date: str, history) -> list[dict]:
        self._counter += 1
        if (self._counter - 1) % self.rebal_every != 0:
            return []
        if decision_date not in self.preds.index:
            return []
        score = self.preds.loc[decision_date].dropna()
        if len(score) < self.top_n + 5:
            return []
        picks = list(score.sort_values(ascending=False).index[: self.top_n])
        held = set(history.open_positions.keys())
        w = self.gross / self.top_n
        targets = [{"symbol": s, "side": "BUY", "weight": w} for s in picks]
        targets += [{"symbol": s, "side": "SELL", "weight": 0} for s in held - set(picks)]
        return targets


def walkforward_days(data, start, end, test_len=60, step=60, min_hist=120):
    days = data.trading_days(start, end)
    segs = []
    i = min_hist
    while i + test_len <= len(days):
        segs.append((days[0], days[max(i - EMBARGO, 0)], days[i], days[i + test_len - 1]))
        i += step
    return segs


def to_qlib_symbol(code: str) -> str:
    s = str(code)
    if "." in s:
        market, num = s.split(".", 1)
        return ("SH" if market == "sh" else "SZ") + num
    num = s.zfill(6)
    return ("SH" if num.startswith(("6", "9", "5")) else "SZ") + num


def main() -> None:
    t_all = time.time()
    qlib.init(provider_uri="data/qlib_data", region=REG_CN, quiet=True)
    store = PITStore()
    data = CachedResearchData(PITResearchData(store))
    panel = data.daily_panel(START, END)
    code_map = {to_qlib_symbol(c): c for c in panel.index.get_level_values(1).unique()}
    segs = walkforward_days(data, START, END)
    print(f"segments: {len(segs)}", flush=True)

    registry = ExperimentRegistry()
    all_preds = {}
    seg_rows = []
    for k, (s0, train_end, t0d, t1d) in enumerate(segs):
        t0 = time.time()
        try:
            handler = Alpha158(start_time=s0, end_time=t1d,
                               fit_start_time=s0, fit_end_time=train_end,
                               instruments="all",
                               label=["Ref($close, -2)/Ref($close, -1) - 1"])
            ds_features = handler.fetch(col_set="feature")
            ds_labels = handler.fetch(col_set="label")
            idx = ds_features.index
            dates = idx.get_level_values(0).astype(str)
            tr_mask = dates <= train_end
            te_mask = (dates >= t0d) & (dates <= t1d)
            xtr, ytr = ds_features[tr_mask], ds_labels[tr_mask]
            xte = ds_features[te_mask]
            ytr = ytr.iloc[:, 0]
            ok = ytr.notna().values
            xtr, ytr = xtr[ok], ytr[ok]
            import lightgbm as lgb
            params = {"objective": "regression", "num_leaves": 32, "learning_rate": 0.05,
                      "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                      "min_data_in_leaf": 200, "lambda_l2": 1.0, "num_threads": 8, "verbose": -1}
            dtr = lgb.Dataset(xtr.values, label=ytr.values, free_raw_data=False)
            booster = lgb.train(params, dtr, num_boost_round=300)
            p = booster.predict(xte.values)
            pred_idx = xte.index
            preds = pd.Series(p, index=pred_idx)
            preds.index = preds.index.set_levels(
                preds.index.levels[0].astype(str), level=0)
            # map qlib symbols back to pangu codes; average dup dates
            df = preds.rename_axis(["date", "symbol"]).reset_index()
            df["code"] = df["symbol"].map(code_map)
            df = df.dropna(subset=["code"])
            daily_rank_ic = None
            lab = ds_labels[te_mask].iloc[:, 0]
            joined = pd.DataFrame({"p": preds, "y": lab}).dropna()
            ics = joined.groupby(level=0).apply(
                lambda g: g["p"].corr(g["y"], method="spearman") if len(g) > 20 else np.nan).dropna()
            daily_rank_ic = float(ics.mean()) if len(ics) else None
            frame = df.pivot_table(index="date", columns="code", values=0)
            all_preds[k] = frame
            seg_rows.append({"segment": k, "train_end": train_end, "test": [t0d, t1d],
                             "rank_ic": daily_rank_ic, "n_train": int(len(xtr)),
                             "elapsed_s": round(time.time() - t0, 1)})
            print(f"seg{k} train<={train_end} test[{t0d}..{t1d}] rank_ic={daily_rank_ic} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            seg_rows.append({"segment": k, "error": repr(e)})
            print(f"seg{k} FAILED {e!r}", flush=True)

    # backtest each segment's prediction frame
    results = []
    for k, frame in all_preds.items():
        t0d, t1d = segs[k][2], segs[k][3]
        strat = PredStrategy(frame)
        res = BacktestV2(data, BacktestConfig()).run(strat, t0d, t1d)
        s = res.summary
        results.append({"segment": k, "test": [t0d, t1d],
                        "total_return": s.get("total_return"), "sharpe": s.get("sharpe"),
                        "max_drawdown": s.get("max_drawdown"), "n_trades": s.get("n_trades"),
                        "win_rate_trades": s.get("win_rate_trades"),
                        "execution_rate": s.get("execution_rate")})
    rets = [r["total_return"] or 0.0 for r in results]
    oos = {"n_segments": len(results),
           "mean_segment_return": float(np.mean(rets)) if rets else None,
           "positive_segments": int(sum(1 for r in rets if r > 0)),
           "total_trades": int(sum(r["n_trades"] or 0 for r in results)),
           "avg_win_rate": float(np.mean([r["win_rate_trades"] or 0 for r in results])) if results else None,
           "avg_execution_rate": float(np.mean([r["execution_rate"] or 0 for r in results])) if results else None,
           "mean_seg_rank_ic": float(np.mean([r["rank_ic"] for r in seg_rows if "rank_ic" in r and r["rank_ic"] is not None])) if any("rank_ic" in r for r in seg_rows) else None}

    # cross-check on the last backtested segment
    independent_ok = None
    if all_preds:
        k_last = max(all_preds)
        t0d, t1d = segs[k_last][2], segs[k_last][3]
        base = BacktestV2(data, BacktestConfig()).run(PredStrategy(all_preds[k_last]), t0d, t1d)
        try:
            cc = cross_check(base, data, PredStrategy(all_preds[k_last]))
            independent_ok = bool(cc.get("consistent"))
        except Exception as e:  # noqa: BLE001
            cc = {"error": repr(e)}
    else:
        cc = {}

    conclusion = ("oos_positive_candidate_for_holdout"
                  if (oos["mean_segment_return"] or 0) > 0 and oos["positive_segments"] >= len(rets) / 2
                  else ("oos_negative_rejected" if (oos["mean_segment_return"] or 0) <= 0
                        else "oos_positive_but_unstable"))
    registry.register({
        "experiment_id": "strategy_lgb_alpha158_wf",
        "hypothesis": "LightGBM on Alpha158 cross-sectional features predicts next-day excess return",
        "economic_rationale": "nonlinear interactions of price/volume/liquidity signals",
        "data": "qlib dump of PITStore (PIT-safe hfq-style total-return prices)",
        "pit_status": "strict: per-segment expanding train with 5d embargo; no test peeking",
        "universe": "all instruments in qlib dump; top-20 by prediction",
        "decision_time": "15:05 T", "execution_time": "T+1 open ±10bp",
        "features": ["Alpha158 (158 features)"],
        "label": "Ref($close,-2)/Ref($close,-1)-1 (train); realized PnL (backtest)",
        "train_range": f"expanding per segment (embargo {EMBARGO}d)",
        "validation_range": "none (early stop via fixed rounds)",
        "test_range": f"walk-forward {len(results)} × 60d segments in {START}→{END}",
        "costs": "commission 3bp min 5; stamp 5bp sell", "slippage": "10bp",
        "capacity": "2% ADV", "baseline": "random20 + rule strategies",
        "parameters": {"num_leaves": 32, "lr": 0.05, "rounds": 300, "top_n": TOP_N},
        "optimization_method": "fixed hyperparams (no tuning)",
        "n_variants_tried": 1,
        "metrics": {"oos": oos, "segments": seg_rows},
        "leakage_audit": {"expanding_train_only": True, "embargo_days": EMBARGO,
                          "label_shift_guarded": True,
                          "static_audit_findings": _audit_module(__file__)},
        "independent_backtest": {"consistent": independent_ok, "detail": cc if not independent_ok else "ok"},
        "conclusion": conclusion, "status": "evaluated",
        "family": "ml_strategy",
    })
    out = Path("data/experiments/strategies")
    out.mkdir(parents=True, exist_ok=True)
    (out / "lgb_alpha158_wf.json").write_text(
        json.dumps({"oos": oos, "segments": seg_rows, "backtest": results},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"ml study done {time.time()-t_all:.0f}s oos={json.dumps(oos)[:220]}", flush=True)


if __name__ == "__main__":
    main()
