"""Pangu 2.0 统计验证（P5-010 系列）。

- :func:`deflated_sharpe`：Bailey / López de Prado DSR——SR 减去 n_trials 下
  期望最大值（Euler–Mascheroni 近似），方差用 LOO 或外部给定。
- :func:`bootstrap_ci`：移动块自助（block=5）置信区间。
- :func:`pbo_cscv`：组合对称交叉验证 PBO。
- :func:`multiple_hypothesis_report`：多重假设计数 + BH 校正 q 值。
- :func:`daily_return_stats`：日收益统计字典。
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_max_std(n: int) -> float:
    """E[max of n iid N(0,1)] 的 Euler–Mascheroni 近似。"""
    if n <= 1:
        return 0.0
    ln = math.log(n)
    root = math.sqrt(2.0 * ln)
    return root - (math.log(ln) + math.log(4.0 * math.pi)) / (2.0 * root) \
        + EULER_GAMMA / root


def deflated_sharpe(returns: Sequence[float], n_trials: int = 1,
                    sr_variance: float | None = None) -> float:
    """DSR = P(真实 SR > SR*)；纯噪声 + 大量试验 → 接近 0。"""
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return 0.0
    sr = float(r.mean()) / sd
    if sr_variance is None:
        # LOO：留一法 SR 估计的样本方差
        srs = []
        for i in range(n):
            sub = np.delete(r, i)
            s = float(sub.std(ddof=1))
            srs.append(float(sub.mean()) / s if s > 0 else 0.0)
        var = float(np.var(srs, ddof=1)) if n > 2 else 0.0
    else:
        var = float(sr_variance)
    sr0 = math.sqrt(max(var, 0.0)) * _expected_max_std(max(int(n_trials), 1))
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurt()) + 3.0        # pandas kurt 为超额峰度
    denom = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr0) * math.sqrt(n - 1) / denom
    return float(_norm_cdf(z))


def bootstrap_ci(returns: Sequence[float], stat: Callable | None = None,
                 n: int = 2000, seed: int = 42, block: int = 5) -> dict:
    """圆形移动块自助；返回 {lo, hi, point, n, block}（2.5% / 97.5% 分位）。"""
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    if stat is None:
        stat = np.mean
    point = float(stat(r)) if r.size else 0.0
    if r.size == 0 or n <= 0:
        return {"lo": point, "hi": point, "point": point, "n": 0, "block": int(block)}
    T = r.size
    block = max(1, min(int(block), T))
    n_blocks = int(math.ceil(T / block))
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n))
    for k in range(int(n)):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]
        stats[k] = stat(r[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "point": point,
            "n": int(n), "block": int(block)}


def pbo_cscv(perf_matrix, S: int = 6) -> dict:
    """组合对称交叉验证 PBO。perf_matrix: T×N（行=日收益，列=策略/参数）。

    对每个 S 选 S/2 的组合：IS 最优策略在 OOS 的相对秩 logit <= 0 计为一次
    "回撤过拟合"。返回 {"pbo", "n_combinations"}。
    """
    df = np.asarray(perf_matrix, dtype=float)
    if df.ndim != 2 or df.shape[1] < 2:
        return {"pbo": 0.0, "n_combinations": 0}
    T, N = df.shape
    if T < S:
        raise ValueError(f"perf_matrix rows {T} < S {S}")

    def _sharpe(x: np.ndarray) -> np.ndarray:
        sd = x.std(axis=0, ddof=1)
        return np.divide(x.mean(axis=0), sd, out=np.zeros_like(sd), where=sd > 0)

    blocks = np.array_split(np.arange(T), S)
    logits: list[float] = []
    for combo in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in combo])
        if oos_idx.size == 0:
            continue
        best = int(np.argmax(_sharpe(df[is_idx])))
        oos = _sharpe(df[oos_idx])
        # 相对秩从"最差=1"到"最好=N"：IS 最优在 OOS 掉入较差一半 → 过拟合
        better = int((oos > oos[best]).sum())
        omega = (N - better) / (N + 1.0)
        logits.append(math.log(omega / (1.0 - omega)))
    pbo = float(np.mean([1.0 if l <= 0 else 0.0 for l in logits])) if logits else 0.0
    return {"pbo": pbo, "n_combinations": len(logits)}


def multiple_hypothesis_report(experiments: Iterable[dict]) -> dict:
    """计数 + Benjamini–Hochberg 校正 q 值（p_value 存在的实验）。"""
    exps = [dict(e) for e in experiments]
    pairs = [(i, float(e["p_value"])) for i, e in enumerate(exps)
             if e.get("p_value") is not None]
    m = len(pairs)
    q = [None] * len(exps)
    if m:
        ordered = sorted(pairs, key=lambda x: x[1])
        raw = [p * m / (k + 1) for k, (_, p) in enumerate(ordered)]
        # 从大到小做单调化（step-up）
        running = raw[-1]
        for k in range(m - 1, -1, -1):
            running = min(running, raw[k])
            q[ordered[k][0]] = min(1.0, running)
    return {
        "n_experiments": len(exps),
        "n_with_p": m,
        "n_significant_raw": sum(1 for _, p in pairs if p < 0.05),
        "n_significant_bh": sum(1 for v in q if v is not None and v < 0.05),
        "method": "benjamini-hochberg",
        "experiments": [{**e, "q_value_bh": q[i]} for i, e in enumerate(exps)],
    }


def daily_return_stats(equity_series) -> dict:
    """P5-010 日收益统计字典（equity: Series 或含 equity 列的 DataFrame）。"""
    if isinstance(equity_series, pd.DataFrame):
        s = equity_series["equity"].astype(float)
    else:
        s = pd.Series(equity_series, dtype=float)
    s = s.dropna()
    if len(s) < 2:
        return {"n_days": max(0, len(s) - 1), "total_return": 0.0,
                "annualized_return": 0.0, "annualized_vol": 0.0, "sharpe": 0.0,
                "sortino": 0.0, "max_drawdown": 0.0, "worst_day": 0.0,
                "var_1pct": 0.0, "es_1pct": 0.0,
                "positive_day_ratio": 0.0, "negative_day_ratio": 0.0,
                "skew": 0.0, "kurtosis": 0.0}
    rets = (s / s.shift(1) - 1.0).dropna()
    total = float(s.iloc[-1] / s.iloc[0] - 1.0)
    years = len(s) / TRADING_DAYS
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    std = float(rets.std(ddof=1))
    downside = rets[rets < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    peak = s.cummax()
    dd = (peak - s) / peak
    mdd_end = int(dd.values.argmax())
    mdd_start = int(s.values[:mdd_end + 1].argmax())
    var_1pct = float(rets.quantile(0.01))
    tail = rets[rets <= var_1pct]
    return {
        "n_days": int(len(rets)),
        "total_return": total,
        "annualized_return": float(ann),
        "annualized_vol": std * math.sqrt(TRADING_DAYS),
        "sharpe": float(rets.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0,
        "sortino": float(rets.mean() / dstd * math.sqrt(TRADING_DAYS)) if dstd > 0 else 0.0,
        "max_drawdown": float(dd.max()),
        "mdd_start": str(s.index[mdd_start]),
        "mdd_end": str(s.index[mdd_end]),
        "worst_day": float(rets.min()),
        "var_1pct": var_1pct,
        "es_1pct": float(tail.mean()) if len(tail) else 0.0,
        "positive_day_ratio": float((rets > 0).mean()),
        "negative_day_ratio": float((rets < 0).mean()),
        "skew": float(rets.skew()),
        "kurtosis": float(rets.kurt()),
    }
