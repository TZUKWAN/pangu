"""P3-002/003: Alpha158 + LightGBM/XGBoost baseline on the Pangu qlib dump.

Split discipline (docs/pangu2/RESEARCH_PROTOCOL.md):
  train  : 2022-01-04 → 2025-06-30
  valid  : 2025-07-01 → 2025-12-31   (model selection / early stopping ONLY)
  test   : 2026-01-01 → 2026-05-29   (evaluated ONCE at final gate)
  HOLDOUT: 2026-06-01 → 2026-09-04   (NOT touched by this script)

Label: Ref($close,-2)/Ref($close,-1) - 1  (T+1 close → T+2 close, no T-close entry).
Output: data/experiments/ml_baselines/<model>_<ts>.json with valid IC / Rank IC / top-bottom
spread; every run is appended to data/experiments/registry.jsonl via the standard registry
(if engine.validation.experiment_registry exists).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.data import D

TRAIN = ("2022-01-04", "2025-06-30")
VALID = ("2025-07-01", "2025-12-31")

LABEL = ["Ref($close, -2)/Ref($close, -1) - 1"]


def init() -> None:
    qlib.init(provider_uri="data/qlib_data", region=REG_CN, quiet=True)


def build_dataset(model_name: str):
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.loader import QlibDataLoader

    handler_conf = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": TRAIN[0],
            "end_time": VALID[1],
            "fit_start_time": TRAIN[0],
            "fit_end_time": TRAIN[1],
            "instruments": "all",
            "label": LABEL,
        },
    }
    # split via DatasetH segments
    dataset = DatasetH(
        handler=Alpha158(**handler_conf["kwargs"]),
        segments={
            "train": (TRAIN[0], TRAIN[1]),
            "valid": (VALID[0], VALID[1]),
        },
    )
    return dataset


def rank_ic(pred: pd.Series, label: pd.Series) -> dict:
    df = pd.concat([pred, label], axis=1)
    df.columns = ["pred", "label"]
    df = df.dropna()
    daily = df.groupby(level=0).apply(
        lambda g: g["pred"].corr(g["label"], method="spearman") if len(g) > 5 else np.nan
    )
    daily_p = df.groupby(level=0).apply(
        lambda g: g["pred"].corr(g["label"]) if len(g) > 5 else np.nan
    )
    daily = daily.dropna()
    daily_p = daily_p.dropna()
    # long-bottom-quantile vs top-quantile spread using pred ranks
    spreads = []
    for dt_, g in df.groupby(level=0):
        if len(g) < 100:
            continue
        q = g["pred"].rank(pct=True)
        spreads.append(g.loc[q >= 0.8, "label"].mean() - g.loc[q <= 0.2, "label"].mean())
    return {
        "ic_rank_mean": float(daily.mean()),
        "ic_rank_std": float(daily.std()),
        "icir_rank": float(daily.mean() / daily.std()) if daily.std() else 0.0,
        "n_days": int(len(daily)),
        "ic_pearson_mean": float(daily_p.mean()),
        "top_bottom_spread_mean": float(np.nanmean(spreads)) if spreads else None,
        "top_bottom_n_days": len(spreads),
    }


def run_model(model_name: str) -> dict:
    t0 = time.time()
    dataset = build_dataset(model_name)
    x_train = dataset.prepare("train", col_set="feature")
    y_train = dataset.prepare("train", col_set="label")
    x_valid = dataset.prepare("valid", col_set="feature")
    y_valid = dataset.prepare("valid", col_set="label")
    print(f"{model_name}: train {x_train.shape}, valid {x_valid.shape}", flush=True)

    label_col = y_train.columns[0]
    # clean labels/features (forward-ref labels are NaN at segment tails)
    tr_mask = y_train.iloc[:, 0].notna().values
    va_mask = y_valid.iloc[:, 0].notna().values
    y_train = y_train.loc[tr_mask]
    x_train = x_train.loc[tr_mask]
    y_valid_c = y_valid.loc[va_mask]
    x_valid_c = x_valid.loc[va_mask]
    if model_name == "lgb":
        import lightgbm as lgb

        xtr = x_train.replace([np.inf, -np.inf], np.nan)
        xva = x_valid_c.replace([np.inf, -np.inf], np.nan)
        dtr = lgb.Dataset(xtr.values, label=y_train.iloc[:, 0].values,
                          feature_name=list(xtr.columns), free_raw_data=False)
        dva = lgb.Dataset(xva.values, label=y_valid_c.iloc[:, 0].values,
                          feature_name=list(xva.columns), reference=dtr)
        params = {"objective": "regression", "num_leaves": 64, "learning_rate": 0.05,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                  "min_data_in_leaf": 200, "lambda_l1": 0.1, "lambda_l2": 1.0,
                  "num_threads": 8, "verbose": -1}
        booster = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = pd.Series(booster.predict(xva.values), index=x_valid_c.index)
        met = rank_ic(pred, y_valid_c.iloc[:, 0])
        met.update(model=model_name, train=x_train.shape, valid=x_valid.shape,
                   best_iter=booster.best_iteration,
                   elapsed_s=round(time.time() - t0, 1))
        return met
    elif model_name == "xgb":
        import xgboost as xgb

        xtr = x_train.replace([np.inf, -np.inf], np.nan)
        xva = x_valid_c.replace([np.inf, -np.inf], np.nan)
        dtr = xgb.DMatrix(xtr, label=y_train.iloc[:, 0])
        dva = xgb.DMatrix(xva, label=y_valid_c.iloc[:, 0])
        params = {"max_depth": 8, "eta": 0.05, "objective": "reg:squarederror",
                  "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 100,
                  "eval_metric": "rmse", "nthread": 8}
        bst = xgb.train(params, dtr, num_boost_round=600, evals=[(dva, "valid")],
                        early_stopping_rounds=50, verbose_eval=100)
        pred = pd.Series(bst.predict(dva), index=x_valid_c.index)
        met = rank_ic(pred, y_valid_c.iloc[:, 0])
        met.update(model=model_name, train=x_train.shape, valid=x_valid.shape,
                   elapsed_s=round(time.time() - t0, 1))
        return met
    elif model_name == "linear":
        from sklearn.linear_model import Ridge

        xtr = x_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        xva = x_valid_c.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        reg = Ridge(alpha=100.0)
        reg.fit(xtr.values, y_train.iloc[:, 0].values)
        pred = pd.Series(reg.predict(xva.values), index=x_valid_c.index)
        met = rank_ic(pred, y_valid_c.iloc[:, 0])
        met.update(model=model_name, train=x_train.shape, valid=x_valid.shape,
                   elapsed_s=round(time.time() - t0, 1))
        return met
    else:
        raise ValueError(model_name)

    raise RuntimeError("unreachable")


def main() -> None:
    init()
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else ["linear", "lgb", "xgb"]
    out_dir = Path("data/experiments/ml_baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for m in models:
        try:
            results[m] = run_model(m)
            print(m, json.dumps(results[m], indent=2), flush=True)
        except Exception as e:  # noqa: BLE001
            results[m] = {"model": m, "error": repr(e)}
            print(f"{m} FAILED: {e!r}", flush=True)
    out = out_dir / f"alpha158_valid_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written:", out)
    # register into the append-only experiment registry (audit trail)
    try:
        from engine.validation.experiment_registry import ExperimentRegistry
        reg = ExperimentRegistry()
        for m, r in results.items():
            if "error" in r:
                reg.register({
                    "experiment_id": f"ml_alpha158_{m}_baseline",
                    "hypothesis": f"Alpha158 + {m} baseline predicts next-day return",
                    "economic_rationale": "standard qlib benchmark features",
                    "data": "qlib dump of PITStore", "pit_status": "train/valid split, holdout untouched",
                    "universe": "all instruments", "decision_time": "15:05 T",
                    "execution_time": "T+1 close→T+2 close label",
                    "features": ["Alpha158"], "label": "Ref($close,-2)/Ref($close,-1)-1",
                    "train_range": str(TRAIN), "validation_range": str(VALID),
                    "test_range": "not touched", "costs": "n/a (IC study)",
                    "slippage": "n/a", "capacity": "n/a", "baseline": "zero-IC null",
                    "parameters": {}, "optimization_method": "default hyperparams",
                    "n_variants_tried": 1, "metrics": {"error": r["error"]},
                    "leakage_audit": {"fit_window_train_only": True},
                    "independent_backtest": "not_applicable_ic_study",
                    "conclusion": f"error: {r['error']}", "status": "failed",
                    "family": "ml_baseline"})
            else:
                reg.register({
                    "experiment_id": f"ml_alpha158_{m}_baseline",
                    "hypothesis": f"Alpha158 + {m} baseline predicts next-day return",
                    "economic_rationale": "standard qlib benchmark features",
                    "data": "qlib dump of PITStore", "pit_status": "train/valid split, holdout untouched",
                    "universe": "all instruments", "decision_time": "15:05 T",
                    "execution_time": "T+1 close→T+2 close label",
                    "features": ["Alpha158"], "label": "Ref($close,-2)/Ref($close,-1)-1",
                    "train_range": str(TRAIN), "validation_range": str(VALID),
                    "test_range": "not touched", "costs": "n/a (IC study)",
                    "slippage": "n/a", "capacity": "n/a", "baseline": "zero-IC null",
                    "parameters": {}, "optimization_method": "default hyperparams",
                    "n_variants_tried": 1, "metrics": r,
                    "leakage_audit": {"fit_window_train_only": True},
                    "independent_backtest": "not_applicable_ic_study",
                    "conclusion": "weak_positive_ic_baseline" if (r.get("ic_rank_mean") or 0) > 0 else "no_ic",
                    "status": "evaluated", "family": "ml_baseline"})
    except Exception as e:  # noqa: BLE001
        print("registry append failed:", repr(e))


if __name__ == "__main__":
    main()
