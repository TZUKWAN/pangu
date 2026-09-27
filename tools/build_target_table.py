"""构建 +5% 目标命中表（向量化，研究窗 + OOS 校准）。

产出：
- data/decision/target_hit_table.json（全研究窗构建，供线上 lookup）
- OOS 校准：用 2022→2024-12 构建的表去预测 2025 年各桶实际命中率，
  输出校准误差并登记 experiment（诚实评估条件命中率的可泛化性）。

用法：.venv/Scripts/python tools/build_target_table.py [--horizon 5] [--target 0.05] [--stop 0.05]
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
from engine.decision.target_prob import (DEFAULT_HORIZON,  # noqa: E402
                                         DEFAULT_STOP_PCT, DEFAULT_TARGET_PCT,
                                         TargetHitTable, rev_bucket, wilson_lb)
from engine.validation.experiment_registry import ExperimentRegistry  # noqa: E402

START, END = "2022-01-04", "2025-12-31"
CALIB_OOS_START = "2025-01-02"


def regime_series(close: pd.DataFrame) -> pd.Series:
    """因果市场状态：等权指数 vs MA20 + 宽度中位数 → risk_on/neutral/risk_off。"""
    idx = close.mean(axis=1)
    ma20 = idx.rolling(20, min_periods=15).mean()
    above = close > close.rolling(20, min_periods=15).mean()
    breadth = above.mean(axis=1)
    labels = pd.Series("neutral", index=close.index)
    labels[(idx > ma20) & (breadth > 0.55)] = "risk_on"
    labels[(idx < ma20) & (breadth < 0.45)] = "risk_off"
    return labels


def build_cells(panel: pd.DataFrame, dates: List[str], horizon: int,
                target_pct: float, stop_pct: float, entry_style: str
                ) -> Dict[str, List[int]]:
    close = panel["close"].unstack("code").sort_index()
    open_ = panel["open"].unstack("code").sort_index()
    high = panel["high"].unstack("code").sort_index()
    low = panel["low"].unstack("code").sort_index()
    ret5 = close.pct_change(5, fill_method=None)
    rmean, rstd = ret5.mean(axis=1), ret5.std(axis=1)
    rz = ret5.sub(rmean, axis=0).div(rstd.replace(0, np.nan), axis=0)
    regimes = regime_series(close)

    if entry_style == "tail_close":
        entry = close
    else:
        entry = open_.shift(-1)
    win_high = high.rolling(horizon, min_periods=horizon).max().shift(-horizon)
    win_low = low.rolling(horizon, min_periods=horizon).min().shift(-horizon)

    cells: Dict[str, List[int]] = {}
    for day in dates:
        if day not in entry.index:
            continue
        e = entry.loc[day]
        tgt_lvl = e * (1 + target_pct)
        stp_lvl = e * (1 - stop_pct)
        wh = win_high.loc[day]
        wl = win_low.loc[day]
        rz_day = rz.loc[day]
        regime = regimes.get(day, "neutral")
        ok = e.notna() & wh.notna() & wl.notna() & rz_day.notna()
        if not ok.any():
            continue
        e, tgt_lvl, stp_lvl = e[ok], tgt_lvl[ok], stp_lvl[ok]
        wh, wl, rz_day = wh[ok], wl[ok], rz_day[ok]
        stopped = wl <= stp_lvl                     # stop_first 保守
        hit = (~stopped) & (wh >= tgt_lvl)
        for code in e.index:
            rb = rev_bucket(float(rz_day[code]))
            cell = f"{regime}|{rb}"
            c = cells.setdefault(cell, [0, 0])
            c[1] += 1
            if hit[code]:
                c[0] += 1
    return cells


def main() -> None:
    t0 = time.time()
    horizon = int(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else DEFAULT_HORIZON
    target_pct = float(sys.argv[sys.argv.index("--target") + 1]) if "--target" in sys.argv else DEFAULT_TARGET_PCT
    stop_pct = float(sys.argv[sys.argv.index("--stop") + 1]) if "--stop" in sys.argv else DEFAULT_STOP_PCT
    entry_style = sys.argv[sys.argv.index("--entry") + 1] if "--entry" in sys.argv else "next_open"
    out_suffix = sys.argv[sys.argv.index("--out-suffix") + 1] if "--out-suffix" in sys.argv else ""

    store = PITStore()
    panel = store.daily_panel(START, END)
    close = panel["close"].unstack("code").sort_index()
    days = close.index.tolist()
    print(f"panel {panel.shape}, days {len(days)}", flush=True)

    # ---- 多持有期表（h=5/10/20）：为"拿多久才能 ≥5%"提供分级证据 ----
    for h in (5, 10, 20):
        cells_h = build_cells(panel, days[60:-h], h, target_pct, stop_pct,
                              entry_style)
        t_h = TargetHitTable(cells=cells_h, target_pct=target_pct,
                             stop_pct=stop_pct, horizon=h,
                             built_range=f"{START}→{END}")
        t_h.merge_parent_cells()
        out = Path(f"data/decision/target_hit_table_h{h}{out_suffix}.json")
        t_h.save(out)
        base = t_h.lookup("all", "rev_all")
        print(f"h={h} cells={len(cells_h)} base_rate={base.rate:.1%} "
              f"wilson_lb={base.wilson_lb:.1%} -> {out}", flush=True)

    # 兼容主表 = h5
    cells_full = build_cells(panel, days[60:-horizon], horizon, target_pct,
                             stop_pct, "next_open")
    table = TargetHitTable(cells=cells_full, target_pct=target_pct,
                           stop_pct=stop_pct, horizon=horizon,
                           built_range=f"{START}→{END}")
    table.merge_parent_cells()
    table.save()
    print(f"full table cells={len(table.cells)} saved", flush=True)

    # ---- OOS 校准：2022→2024 建，2025 验 ----
    train_days = [d for d in days if d < CALIB_OOS_START]
    eval_days = [d for d in days if CALIB_OOS_START <= d <= END][60:-horizon]
    train_cells = build_cells(panel, train_days[60:-horizon], horizon,
                              target_pct, stop_pct, "next_open")
    train_table = TargetHitTable(cells=train_cells, min_n=30,
                                 target_pct=target_pct, stop_pct=stop_pct,
                                 horizon=horizon)
    train_table.merge_parent_cells()
    eval_cells = build_cells(panel, eval_days, horizon, target_pct,
                             stop_pct, "next_open")
    calib = []
    for cell, (hits, n) in eval_cells.items():
        if n < 30:
            continue
        est = train_table.lookup(*cell.split("|"))
        if est.n < 30:
            continue
        calib.append({"cell": cell, "predicted": est.rate, "actual": hits / n,
                      "n_eval": n, "n_train": est.n})
    mae = float(np.mean([abs(c["predicted"] - c["actual"]) for c in calib])) \
        if calib else None
    print(f"calibration buckets={len(calib)} MAE={mae}", flush=True)

    registry = ExperimentRegistry()
    registry.register({
        "experiment_id": "target_hit_table_v1",
        "hypothesis": "conditional +5% target-hit rates (regime × reversal-z) are stable enough OOS to gate BUYs",
        "economic_rationale": "user requires ≥5% return at end of recommended holding window; we gate BUY on OOS-validated conditional hit frequency",
        "data": "PITStore full archive 2022→2025-12",
        "pit_status": "strict asof; forward window only for labels; features = decision-date regime/rev-z",
        "universe": "panel rows (no ST filter for rate estimation; BUY gate applies elsewhere)",
        "decision_time": "15:05 T", "execution_time": "T+1 open entry, h-day window",
        "features": ["regime(causal)", "rev5_z_bucket"],
        "label": f"touch +{target_pct:.0%} before stop {stop_pct:.0%} within {horizon}d (stop_first)",
        "train_range": f"{START}→2024-12-31 (production table uses full window)",
        "validation_range": "none",
        "test_range": f"calibration on 2025 ({len(eval_days)} days)",
        "costs": "n/a (probability study)", "slippage": "n/a", "capacity": "n/a",
        "baseline": "unconditional hit rate",
        "parameters": {"horizon": horizon, "target": target_pct, "stop": stop_pct,
                       "min_n": 30},
        "optimization_method": "none",
        "n_variants_tried": 1,
        "metrics": {"calibration": calib, "calibration_mae": mae,
                    "full_table_cells": len(table.cells),
                    "full_table_total_n": sum(v[1] for v in table.cells.values())},
        "leakage_audit": {"labels_strictly_after_decision": True,
                          "calibration_train_eval_disjoint": True},
        "independent_backtest": "not_applicable_probability_study",
        "conclusion": ("calibrated_mae_acceptable" if mae is not None and mae <= 0.08
                       else "calibration_mae_high_use_with_caution") if mae is not None
        else "insufficient_eval_buckets",
        "status": "evaluated", "family": "target_probability",
    })
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
