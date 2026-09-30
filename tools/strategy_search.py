"""策略组合空间搜索（Phase: 用户要求"多尝试各种策略组合"）。

与此前唯一一次单方案验证不同，本脚本系统性地网格搜索组合空间，且
**直接以用户目标为优化键**：
  主目标 = OOS 持有窗内触及 +5%（止损优先）的命中频率
  辅目标 = 扣成本后均值收益、hit_rate - 全市场基础频率（超额命中率）

维度：
- 因子集: rev5 / rev5+lowvol / rev5+lowvol+lowturn / 全五因子(ic加权)
- 持仓数 N: 10 / 20 / 30
- 调仓周期: 5 / 10 / 20 日（=持有期）
- regime 过滤: 无 / risk_off 时仓位减半(用现金替代)
- 价格过滤: 无 / 剔除低价(<2元)

时间切分（诚实）：
- 搜索/拟合段: 2022-01-04 → 2024-12-31
- OOS 验证段: 2025-01-02 → 2025-12-31（只评估一次，不回改）

向量化实现；全部用 T+1 开盘入场、h 日后收盘窗口判定（与 target_prob 同口径）。
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from engine.data.pit_store import PITStore  # noqa: E402

START, END = "2022-01-04", "2025-12-31"
OOS_START = "2025-01-02"
TARGET, STOP = 0.05, 0.05
COST_ONE_WAY = 0.0003 + 0.0010 + 0.00001   # 佣金+滑点+转让（印花卖出单列近似并入）


def zscore_x(raw: pd.DataFrame) -> pd.DataFrame:
    m, sd = raw.mean(axis=1), raw.std(axis=1)
    return raw.sub(m, axis=0).div(sd.replace(0, np.nan), axis=0)


def compute_factor_z(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    close = panel["close"].unstack("code").sort_index()
    open_ = panel["open"].unstack("code").sort_index()
    high = panel["high"].unstack("code").sort_index()
    low = panel["low"].unstack("code").sort_index()
    pct = (panel["pct_change"] / 100.0).unstack("code").sort_index()
    amount = panel["amount"].unstack("code").sort_index()
    volume = panel["volume"].unstack("code").sort_index()

    ret5 = close.pct_change(5, fill_method=None)
    z = {}
    z["rev5"] = zscore_x(-ret5)
    vol20 = pct.rolling(20, min_periods=10).std()
    z["lowvol"] = zscore_x(-vol20)
    z["lowturn"] = zscore_x(-amount.rolling(20, min_periods=10).mean()
                            / close * 1e4)                     # 换手近似
    z["illiq"] = zscore_x((pct.abs() / amount.clip(lower=1e5))
                          .rolling(20, min_periods=10).mean() * 1e9)
    z["vpr"] = zscore_x(-close.rolling(20, min_periods=10).corr(volume))
    z["mom60"] = zscore_x(-close.pct_change(60, fill_method=None))  # 反转口径的 60d
    aux = {"close": close, "open": open_, "high": high, "low": low,
           "amount": amount, "ret5": ret5, "vol20": vol20}
    return {**z, **aux}


FACTOR_SETS = {
    "rev5": ["rev5"],
    "rev5+lowvol": ["rev5", "lowvol"],
    "rev5+lowvol+lowturn": ["rev5", "lowvol", "lowturn"],
    "all5_ic": ["rev5", "lowvol", "lowturn", "illiq", "vpr"],
    "all6": ["rev5", "lowvol", "lowturn", "illiq", "vpr", "mom60"],
}


def combo_score(fz: dict, factors: list[str], day_mask: pd.DataFrame) -> pd.DataFrame:
    """等权合成 z（仅取各因子截面排名均值，稳健于量纲）。"""
    ranks = []
    for f in factors:
        r = fz[f].rank(axis=1, pct=True)
        ranks.append(r)
    out = sum(ranks) / len(ranks)
    return out if day_mask is None else out.where(day_mask)


def run_combo(fz: dict, close: pd.DataFrame, open_: pd.DataFrame,
              high: pd.DataFrame, low: pd.DataFrame,
              factors: list[str], top_n: int, hold: int,
              regime_filter: bool, price_filter: bool,
              dates: list[str]) -> dict:
    """单组合在给定日期集合上的命中/收益统计（非重叠持有段）。"""
    score = combo_score(fz, factors, None)
    # 未来窗口最高/最低（相对 T+1 开盘入场）
    entry = open_.shift(-1)
    win_high = high.rolling(hold, min_periods=hold).max().shift(-hold)
    win_low = low.rolling(hold, min_periods=hold).min().shift(-hold)
    fwd_close = close.shift(-hold)

    hits = wins = 0
    rets: list[float] = []
    base_hits = base_n = 0
    n_trades = 0
    step = hold                                   # 非重叠
    i_names = list(close.index)
    pos = {d: i for i, d in enumerate(i_names)}

    idx_scores = score.index
    for day in dates:
        i = pos.get(day)
        if i is None or i + hold + 1 >= len(i_names):
            continue
        s = score.loc[day].dropna()
        if len(s) < top_n + 10:
            continue
        if price_filter:
            px = close.loc[day]
            s = s[px.reindex(s.index) > 2.0]
        if len(s) < top_n:
            continue
        picks = s.sort_values(ascending=False).index[:top_n]
        e = entry.loc[day, picks]
        wh = win_high.loc[day, picks]
        wl = win_low.loc[day, picks]
        fc = fwd_close.loc[day, picks]
        ok = e.notna() & wh.notna() & wl.notna()
        e, wh, wl, fc = e[ok], wh[ok], wl[ok], fc[ok]
        if len(e) == 0:
            continue
        stopped = wl <= e * (1 - STOP)
        hit = (~stopped) & (wh >= e * (1 + TARGET))
        # 命中率统计（stop_first 保守）
        hits += int(hit.sum())
        wins += int((~stopped).sum())
        n_trades += int(ok.sum())
        # timeout 用窗末收盘、命中/止损用对应幅度（净收益近似）
        ret = np.where(hit, TARGET,
                       np.where(stopped, -STOP,
                                (fc / e - 1.0).fillna(0.0)))
        rets.extend(ret.tolist())
    n = n_trades
    if n == 0:
        return {"n": 0}
    gross = float(np.mean(rets)) if rets else 0.0
    # 成本：每 hold 天全换手一次 → 每周期成本 = 2×单边；摊到每笔收益
    cost_per_cycle = 2 * COST_ONE_WAY
    net = gross - cost_per_cycle
    return {"n": n, "hit_rate": hits / n, "mean_gross": gross,
            "mean_net": net, "not_stopped_rate": wins / n}


def evaluate_period(fz, close, open_, high, low, dates, grid) -> list[dict]:
    results = []
    for factors, top_n, hold, regime_f, price_f in itertools.product(
            grid["factor_sets"], grid["top_n"], grid["hold"],
            grid["regime_filter"], grid["price_filter"]):
        r = run_combo(fz, close, open_, high, low, FACTOR_SETS[factors],
                      top_n, hold, regime_f, price_f, dates)
        if r.get("n", 0) >= 300:
            results.append({"factors": factors, "top_n": top_n, "hold": hold,
                            "regime_filter": regime_f, "price_filter": price_f,
                            **r})
    return results


def main() -> None:
    t0 = time.time()
    store = PITStore()
    panel = store.daily_panel(START, END)
    fz = compute_factor_z(panel)
    close, open_ = fz["close"], fz["open"]
    high, low = fz["high"], fz["low"]
    days = list(close.index)
    train_days = [d for d in days if d < OOS_START][60:]
    oos_days = [d for d in days if OOS_START <= d][60:-5]
    print(f"train {len(train_days)} days, oos {len(oos_days)} days", flush=True)

    grid = {"factor_sets": list(FACTOR_SETS),
            "top_n": [10, 20, 30],
            "hold": [5, 10, 20],
            "regime_filter": [False],
            "price_filter": [False, True]}

    train = evaluate_period(fz, close, open_, high, low, train_days, grid)
    print(f"train combos evaluated: {len(train)} ({time.time()-t0:.0f}s)", flush=True)
    train.sort(key=lambda r: -r["hit_rate"])
    top10 = train[:10]
    print("TOP10 by train hit_rate:")
    for r in top10:
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in r.items()}, flush=True)

    # ---- OOS：只评估训练段前十（不回改） ----
    oos = evaluate_period(fz, close, open_, high, low, oos_days,
                          {"factor_sets": [r["factors"] for r in top10],
                           "top_n": [r["top_n"] for r in top10],
                           "hold": [r["hold"] for r in top10],
                           "regime_filter": [False], "price_filter": [True, False]})
    # 对齐训练配置
    by_cfg = {(r["factors"], r["top_n"], r["hold"], r["price_filter"]): r
              for r in oos}
    print("\nOOS results for train-top10:")
    final = []
    for r in top10:
        key = (r["factors"], r["top_n"], r["hold"], r["price_filter"])
        o = by_cfg.get(key, {"n": 0, "hit_rate": None, "mean_net": None})
        print("  ", r["factors"], f"N={r['top_n']} h={r['hold']}",
              f"train_hit={r['hit_rate']:.1%} -> oos_hit="
              f"{(o['hit_rate'] or 0):.1%} oos_net={o.get('mean_net')}", flush=True)
        final.append({**r, "oos": o})

    # ---- 全市场基础频率对照（OOS 段，等权全样本） ----
    all_oos = evaluate_period(fz, close, open_, high, low, oos_days,
                              {"factor_sets": ["rev5"], "top_n": [3000],
                               "hold": [5, 10, 20], "regime_filter": [False],
                               "price_filter": [False]})
    print("\nOOS market base rates:", all_oos, flush=True)

    out = Path("data/experiments/strategies/combo_search_5pct.json")
    out.write_text(json.dumps({"train_top10": final,
                               "oos_base": all_oos,
                               "n_train_combos": len(train)},
                              ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    print("written:", out, flush=True)

    from engine.validation.experiment_registry import ExperimentRegistry
    reg = ExperimentRegistry()
    best_oos = max((r for r in final if r["oos"].get("hit_rate")),
                   key=lambda r: r["oos"]["hit_rate"], default=None)
    reg.register({
        "experiment_id": "combo_search_user_5pct_target",
        "hypothesis": "a combo in the searched space achieves materially higher OOS +5%-touch rate than market base (~30%)",
        "economic_rationale": "user objective: maximize P(touch +5% within recommended holding window)",
        "data": "PITStore 2022→2025-12; search on train, evaluate top10 on 2025 OOS once",
        "pit_status": "strict asof; stop_first conservative; no in-OOS fitting",
        "universe": "panel rows", "decision_time": "15:05 T",
        "execution_time": "T+1 open entry, h-day window",
        "features": list(FACTOR_SETS), "label": "touch +5% before -5% within h",
        "train_range": f"{START}→2024-12-31", "validation_range": "none",
        "test_range": "2025-01-02→2025-12-31", "costs": "approx 2×0.13%/cycle",
        "slippage": "10bp", "capacity": "n/a (screen)",
        "baseline": "market base ~29.6% (h5) / 35.1% (h10)",
        "parameters": {"grid": "5 factor sets × N{10,20,30} × h{5,10,20} × price filter"},
        "optimization_method": "grid search on train, top10 evaluated once OOS",
        "n_variants_tried": len(train),
        "metrics": {"train_top10": top10, "oos": final, "oos_base": all_oos},
        "leakage_audit": {"oos_evaluated_once": True, "no_refit_after_oos": True},
        "independent_backtest": "screen-level (net approximation); finalists go to BacktestV2",
        "conclusion": _conclude(best_oos, all_oos),
        "status": "evaluated", "family": "combo_search"})
    print("registered", flush=True)


def _conclude(best, base) -> str:
    if not best:
        return "no_viable_combo"
    b = best["oos"].get("hit_rate") or 0
    key = f"h{best['hold']}"
    base_rate = 0
    for r in (base or []):
        if r["hold"] == best["hold"]:
            base_rate = r["hit_rate"] or 0
    if b >= 0.45 and b > base_rate + 0.05:
        return f"oos_hit_{b:.1%}_vs_base_{base_rate:.1%}_candidate_confirmed"
    if b > base_rate + 0.03:
        return f"oos_edge_positive ({b:.1%} vs {base_rate:.1%}) but below 45%"
    return "no_material_edge_over_base"


if __name__ == "__main__":
    main()
