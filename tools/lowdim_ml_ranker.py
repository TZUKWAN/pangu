"""低维 ML 排序：在 6 个已验证方向的因子 z 上训练 LightGBM（walk-forward）。

与 Pangu 2.0 的 Alpha158（158 维、过拟合）不同：这里只有 6 维输入、
purged walk-forward、rank 目标，检验"低维学习"能否超过等权/IC 加权。

对照（同一测试段）：
  A. 等权 6 因子
  B. 训练段 IC 加权
  C. LightGBM（每段重训，仅用段前数据）
评估：测试段 Rank IC + Top20 命中率（+5% 触及，止损优先）。
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
from engine.validation.experiment_registry import ExperimentRegistry  # noqa: E402

START, END = "2022-01-04", "2025-12-31"
OOS_START = "2025-01-02"
TARGET, STOP = 0.05, 0.05
HOLD = 20
TOP_N = 10


def factors_from(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from tools.strategy_search import zscore_x
    close = panel["close"].unstack("code").sort_index()
    pct = (panel["pct_change"] / 100.0).unstack("code").sort_index()
    amount = panel["amount"].unstack("code").sort_index()
    volume = panel["volume"].unstack("code").sort_index()
    z = {
        "rev5": zscore_x(-close.pct_change(5, fill_method=None)),
        "lowvol": zscore_x(-pct.rolling(20, min_periods=10).std()),
        "lowturn": zscore_x(-(amount.rolling(20, min_periods=10).mean() / close * 1e4)),
        "illiq": zscore_x((pct.abs() / amount.clip(lower=1e5))
                          .rolling(20, min_periods=10).mean() * 1e9),
        "vpr": zscore_x(-close.rolling(20, min_periods=10).corr(volume)),
        "mom60": zscore_x(-close.pct_change(60, fill_method=None)),
    }
    return z


def main() -> None:
    t0 = time.time()
    import lightgbm as lgb
    store = PITStore()
    panel = store.daily_panel(START, END)
    z = factors_from(panel)
    close = panel["close"].unstack("code").sort_index()
    open_ = panel["open"].unstack("code").sort_index()
    high = panel["high"].unstack("code").sort_index()
    low = panel["low"].unstack("code").sort_index()
    dates = list(close.index)
    entry = open_.shift(-1)
    win_high = high.rolling(HOLD, min_periods=HOLD).max().shift(-HOLD)
    win_low = low.rolling(HOLD, min_periods=HOLD).min().shift(-HOLD)

    # 目标：触及 +5%（stop_first 保守）→ 二分类
    y_hit = ((win_low <= entry * (1 - STOP)).pipe(lambda d: ~d) &
             (win_high >= entry * (1 + TARGET))).astype(float)
    y_hit = y_hit.where(entry.notna() & win_high.notna())

    names = list(z.keys())
    feat_stack = np.stack([z[n].values for n in names], axis=-1)  # (T, C, F)

    def rows_for(day: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        i = dates.index(day)
        X = feat_stack[i]
        y = y_hit.values[i] if day in y_hit.index else np.full(X.shape[0], np.nan)
        codes = list(close.columns)
        m = ~np.isnan(X).all(axis=-1) & ~np.isnan(y)
        return X[m], y[m], [codes[j] for j in range(len(codes)) if m[j]]

    train_days = [d for d in dates[80:] if d < OOS_START]
    test_days = [d for d in dates[80:] if d >= OOS_START][: -HOLD]
    # 每月一个测试点，训练=该点前全部（embargo 25d）
    test_points = test_days[::20]
    print(f"train days {len(train_days)}, test points {len(test_points)}", flush=True)

    ics = {"equal": [], "icw": [], "lgb": []}
    hits = {"equal": 0, "icw": 0, "lgb": 0}
    n_eval = 0
    for k, td in enumerate(test_points):
        i = dates.index(td)
        embargo_i = max(i - 25, 0)
        tr_idx = list(range(80, embargo_i))
        tr_days = [dates[j] for j in tr_idx if j < len(dates) and dates[j] < OOS_START]
        if len(tr_days) < 200:
            continue
        Xs, ys = [], []
        for d in tr_days[::5]:                      # 抽样降载
            X, y, _ = rows_for(d)
            if len(X):
                Xs.append(X)
                ys.append(y)
        Xtr = np.concatenate(Xs)
        ytr = np.concatenate(ys)
        Xte, yte, codes = rows_for(td)
        if len(Xte) < TOP_N + 20:
            continue
        # A. 等权
        s_eq = np.nanmean(Xte, axis=-1)
        # B. IC 加权（训练段逐因子 IC）
        icw = {}
        for fi, nm in enumerate(names):
            ic_list = []
            for d in tr_days[::10]:
                X, y, _ = rows_for(d)
                if len(X) < 50:
                    continue
                f = X[:, fi]
                ic_list.append(np.corrcoef(f, y)[0, 1])
            icw[nm] = np.nanmean(ic_list) if ic_list else 0.0
        w = np.array([max(icw[nm], 0) for nm in names])
        w = w / (w.sum() or 1)
        s_icw = np.nansum(Xte * w, axis=-1)
        # C. LightGBM
        m = lgb.train({"objective": "binary", "num_leaves": 15,
                       "learning_rate": 0.05, "min_data_in_leaf": 500,
                       "feature_fraction": 0.9, "num_threads": 8, "verbose": -1},
                      lgb.Dataset(Xtr, label=ytr), num_boost_round=200)
        s_lgb = m.predict(Xte)
        # 评估
        top = np.argsort(-s_lgb)[:TOP_N]
        f_all = {c: (win_high.loc[td, c] if td in win_high.index else np.nan)
                 for c in codes}
        e_all = {c: (entry.loc[td, c] if td in entry.index else np.nan) for c in codes}
        l_all = {c: (win_low.loc[td, c] if td in win_low.index else np.nan) for c in codes}
        def hit_of(c):
            e, hi, lo = e_all.get(c), f_all.get(c), l_all.get(c)
            if e is None or hi is None or lo is None or not (np.isfinite(e) and np.isfinite(hi) and np.isfinite(lo)):
                return None
            if lo <= e * (1 - STOP):
                return 0
            return 1 if hi >= e * (1 + TARGET) else 0
        for key, scores in (("equal", s_eq), ("icw", s_icw), ("lgb", s_lgb)):
            pick = [codes[j] for j in np.argsort(-scores)[:TOP_N]]
            hs = [hit_of(c) for c in pick]
            hs = [h for h in hs if h is not None]
            hits[key] += sum(hs)
        n_eval += 1
        # Rank IC（lgb）
        valid = ~np.isnan(yte)
        if valid.sum() > 50:
            r = pd.Series(s_lgb[valid]).rank()
            ry = pd.Series(yte[valid]).rank()
            ics["lgb"].append(float(r.corr(ry)))
        if (k + 1) % 3 == 0:
            print(f"point {k+1}/{len(test_points)} td={td} ({time.time()-t0:.0f}s)", flush=True)

    out = {"hold": HOLD, "top_n": TOP_N, "test_points": n_eval,
           "hit_rates": {k: (v / (n_eval * TOP_N) if n_eval else None)
                         for k, v in hits.items()},
           "lgb_rank_ic_mean": float(np.nanmean(ics["lgb"])) if ics["lgb"] else None,
           "elapsed_s": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1), flush=True)
    Path("data/experiments/strategies/lowdim_ml_ranker.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")

    reg = ExperimentRegistry()
    reg.register({
        "experiment_id": "lowdim_ml_ranker_wf",
        "hypothesis": "LightGBM on 6 validated factor-z dims beats equal/IC weighting on +5% touch rate",
        "economic_rationale": "low-dim learning on pre-validated signals (contrast: Alpha158 158-dim overfit)",
        "data": "PITStore 2022→2025-12", "pit_status": "walk-forward, embargo 25d, refit per point",
        "universe": "panel rows", "decision_time": "15:05 T",
        "execution_time": "T+1 open, 20d window",
        "features": names, "label": "touch +5% before -5% within 20d (binary)",
        "train_range": "expanding to each test point",
        "validation_range": "none (regularized, low-dim)",
        "test_range": f"2025 monthly points ×{n_eval}",
        "costs": "n/a (ranking study)", "slippage": "n/a", "capacity": "n/a",
        "baseline": "equal-weight 6 factors; train-IC weighting",
        "parameters": {"num_leaves": 15, "rounds": 200, "lr": 0.05,
                       "min_data_in_leaf": 500},
        "optimization_method": "fixed hyperparams",
        "n_variants_tried": 3,
        "metrics": out,
        "leakage_audit": {"walk_forward_refit": True, "embargo_days": 25,
                          "no_oos_refit": True},
        "independent_backtest": "not_applicable_ranking_study",
        "conclusion": _conclude(out),
        "status": "evaluated", "family": "ml_ranker"})
    print("registered", flush=True)


def _conclude(out: dict) -> str:
    hr = out["hit_rates"]
    lgb, eq = hr.get("lgb"), hr.get("equal")
    if lgb is None or eq is None:
        return "insufficient_data"
    if lgb > eq + 0.02:
        return f"lgb_beats_equal ({lgb:.1%} vs {eq:.1%})"
    return f"no_edge_over_equal ({lgb:.1%} vs {eq:.1%})"


if __name__ == "__main__":
    main()
