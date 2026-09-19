"""单因子诊断 evaluate_factor（P2-003）。

对每个决策日 d 调 factor.compute(asof=d) 构建特征矩阵；为效率，底层用一个
只读缓存视图：全量面板只取一次（pad_start..end），随后每次 compute 的取数
请求被硬切片为 date <= asof——双重 PIT 兜底，特征永远只用过去。

前向收益：h 期标签 = 按 code 分组、位置滚动求和（d+1..d+h 的
pct_change/100）。唯一出现的负向 shift 只作用于**标签序列**，
绝不会有任何被当作特征的东西经过 shift(-h)（见 forward_returns 注释）。
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .factors.base import Factor

DEFAULT_HORIZONS = (1, 3, 5, 10, 20)


# ---------------------------------------------------------------------------
# 只读缓存视图（PIT 硬切片）
# ---------------------------------------------------------------------------

class _AsofView:
    """包住 _CachedData：任何 daily_panel 访问被硬切到 <= 当前 compute asof。

    结构性保证特征只用到 <= d 的数据（红队 finding 2 修复），
    而不是依赖各因子自觉。
    """

    def __init__(self, cached: "_CachedData", asof: str):
        self._cached = cached
        self._asof = str(asof)

    def daily_panel(self, start, end, symbols=None):
        return self._cached.daily_panel(start, min(str(end), self._asof), symbols)

    def universe(self, date):
        return self._cached.universe(min(str(date), self._asof))

    def index_daily(self, code, start, end):
        return self._cached.index_daily(code, start, min(str(end), self._asof))

    def trading_days(self, start, end):
        return self._cached.trading_days(start, min(str(end), self._asof))

    @property
    def panel(self):
        return self._cached.daily_panel(self._cached._dates[0], self._asof)             if self._cached._dates else self._cached.panel.iloc[0:0]


class _CachedData:
    """包装 ResearchData：全量面板取一次，daily_panel 每次硬切片 date<=end。"""

    def __init__(self, inner, cache_start: str, cache_end: str):
        self._inner = inner
        full = inner.daily_panel(cache_start, cache_end)
        if full.index.nlevels != 2:
            raise ValueError("daily_panel must return MultiIndex (date, code)")
        names = list(full.index.names)
        if "date" not in names or "code" not in names:
            full.index.names = ["date", "code"]
        lvl = full.index.get_level_values("date")
        full = full[lvl <= cache_end].sort_index()  # STRICT end 兜底
        self._full = full
        self._dates = sorted(self._full.index.get_level_values("date").unique())

    def daily_panel(self, start: str, end: str,
                    symbols: Optional[list[str]] = None) -> pd.DataFrame:
        lvl = self._full.index.get_level_values("date")
        sub = self._full[(lvl >= start) & (lvl <= end)]
        if symbols is not None:
            sub = sub[sub.index.get_level_values("code").isin(set(symbols))]
        return sub

    def universe(self, date: str) -> pd.DataFrame:
        return self._inner.universe(date)

    def index_daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        return self._inner.index_daily(code, start, end)

    def trading_days(self, start: str, end: str) -> list[str]:
        return [d for d in self._dates if start <= d <= end]

    @property
    def panel(self) -> pd.DataFrame:
        return self._full


# ---------------------------------------------------------------------------
# 前向收益（标签）
# ---------------------------------------------------------------------------

def forward_returns(panel: pd.DataFrame, horizon: int) -> pd.Series:
    """h 期前向收益，索引对齐在**决策行** d 上：值 = d+1..d+h 的收益之和。

    实现：rolling(h).sum() 之后再 shift(-h)。全代码库唯一的负向 shift 在此、
    且只作用于标签序列；特征一律来自 factor.compute（只读 <= asof 的数据），
    不可能经过本函数。
    """
    r = panel["pct_change"].astype(float) / 100.0

    def _fwd(s: pd.Series) -> pd.Series:
        return s.rolling(horizon, min_periods=horizon).sum().shift(-horizon)

    return r.groupby(level="code", group_keys=False).apply(_fwd)


# ---------------------------------------------------------------------------
# 统计小工具
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if len(ra) < 2 or np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _jsonable(obj, _depth=0):
    """递归转 JSON 安全结构（numpy 标量 -> python，float 截断到 8 位）。"""
    if _depth > 12:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, _depth + 1) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else round(f, 8)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def evaluate_factor(factor: Factor, data, start: str, end: str,
                    horizons: Iterable[int] = DEFAULT_HORIZONS,
                    quantiles: int = 5,
                    min_history: Optional[int] = None) -> dict:
    """单因子诊断报告（研究指标，非交易信号；long-short spread 仅为研究度量）。

    Args:
        factor: 因子实例（compute(asof, universe, data) -> raw_score）。
        data: ResearchData 协议实现。
        start / end: 决策日期范围（ISO）。特征严格只用 <= d 的数据；
            标签最多取到 end（超出端点的标签自然缺 NaN 并被剔除）。
            每个决策日的股票池 = 当日面板中有行的代码（行存在 <=> 当日实际
            可交易，是最严格的 PIT 信号，也避免逐日 universe() 快照开销）。
        horizons: 前向收益持有期（交易日）。
        quantiles: 分位组数（q1..qN）。
        min_history: 可选；若给定，则丢弃最早 min_history 个决策日
            （历史预热保护），默认 None 不丢弃。
    """
    horizons = tuple(int(h) for h in horizons)
    max_h = max(horizons)
    lookback = max(int(getattr(factor.meta, "lookback_days", 0) or 0), 1)
    pad_start = (pd.Timestamp(start) - pd.Timedelta(days=int(lookback * 2.5) + max_h + 20)
                 ).strftime("%Y-%m-%d")
    cached = _CachedData(data, pad_start, end)
    panel = cached.panel

    decision_dates = cached.trading_days(start, end)
    if min_history is not None and min_history > 0:
        decision_dates = decision_dates[min_history:]
    if not decision_dates:
        raise ValueError(f"no decision dates in [{start}, {end}]")

    # --- 特征矩阵：逐决策日 compute（缓存视图保证每次只见 <= d 的行） ---------
    # 决策日股票池直接取“当日面板有行”的代码：面板行存在 <=> 当日实际可交易，
    # 是最严格的 PIT 可交易信号（停牌/未上市日无行），且无需逐日调 universe()。
    codes_by_date = _codes_by_date(panel)
    feats: dict[str, pd.Series] = {}
    for d in decision_dates:
        uni = codes_by_date.get(d)
        if uni is None or len(uni) == 0:
            continue
        s = factor.compute(d, uni, _AsofView(cached, d))
        feats[d] = pd.Series(s, index=pd.Index([str(c) for c in uni])).reindex(uni)
    if not feats:
        raise ValueError("factor produced no features on any decision date")
    feature = pd.concat(feats, names=["date", "code"]).astype(float)

    mkt_daily = (panel["pct_change"] / 100.0).groupby(level="date").mean()
    mkt_vol20 = mkt_daily.rolling(20, min_periods=10).std()

    min_names = max(8, quantiles + 3)
    ic_rows: dict[int, list[dict]] = {h: [] for h in horizons}
    q_acc: dict[int, dict] = {h: {"sum": np.zeros(quantiles), "cnt": np.zeros(quantiles),
                                  "dates": 0} for h in horizons}

    for h in horizons:
        label = forward_returns(panel, h)
        merged = pd.DataFrame({"feature": feature, "label": label}).dropna()
        if merged.empty:
            continue
        for d, g in merged.groupby(level="date"):
            f = g["feature"].to_numpy()
            lab = g["label"].to_numpy()
            if len(f) < min_names:
                continue
            ric = _spearman(f, lab)
            pic = _pearson(f, lab)
            if not np.isfinite(ric):
                continue
            ic_rows[h].append({"date": str(d), "ric": ric, "pic": pic, "n": len(f)})
            ranks = pd.Series(f).rank(method="first")
            bins = pd.qcut(ranks, quantiles, labels=False).to_numpy()
            for b in range(quantiles):
                sel = bins == b
                q_acc[h]["sum"][b] += float(lab[sel].sum())
                q_acc[h]["cnt"][b] += int(sel.sum())
            q_acc[h]["dates"] += 1

    all_ic_dates = sorted({r["date"] for rows in ic_rows.values() for r in rows})
    vol_on_dates = mkt_vol20.reindex(all_ic_dates).dropna()
    vol_median = float(vol_on_dates.median()) if len(vol_on_dates) else float("nan")

    horizons_report: dict[int, dict] = {}
    for h in horizons:
        rows = ic_rows[h]
        if len(rows) < 5:
            horizons_report[h] = {
                "ic_mean_pearson": None, "ic_mean_spearman": None, "icir": None,
                "t_stat": None, "n_days": len(rows), "coverage_mean": None,
                "quantile_returns": {}, "long_short_spread": None,
                "monotonicity": None, "ic_first_half": None, "ic_second_half": None,
                "ic_high_vol": None, "ic_low_vol": None,
                "note": "insufficient valid daily cross-sections",
            }
            continue
        rics = np.array([r["ric"] for r in rows])
        pics = np.array([r["pic"] for r in rows])
        ns = np.array([r["n"] for r in rows], dtype=float)
        sd = rics.std(ddof=1)
        icir = float(rics.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else None
        t_stat = float(rics.mean() / sd * np.sqrt(len(rics))) if sd and sd > 0 else None

        qmean = np.where(q_acc[h]["cnt"] > 0,
                         q_acc[h]["sum"] / np.maximum(q_acc[h]["cnt"], 1), np.nan)
        spread = float(qmean[-1] - qmean[0]) if np.isfinite(qmean).all() else None
        mono = _spearman(np.arange(1, quantiles + 1, dtype=float), qmean) \
            if np.isfinite(qmean).all() else None

        half = len(rows) // 2
        ic_first = float(rics[:half].mean()) if half else None
        ic_second = float(rics[half:].mean()) if half < len(rows) else None

        vols = np.array([mkt_vol20.get(r["date"], np.nan) for r in rows], dtype=float)
        hi = np.isfinite(vols) & (vols >= vol_median)
        lo = np.isfinite(vols) & (vols < vol_median)
        ic_hi = float(rics[hi].mean()) if hi.any() else None
        ic_lo = float(rics[lo].mean()) if lo.any() else None

        horizons_report[h] = {
            "ic_mean_pearson": float(np.nanmean(pics)),
            "ic_mean_spearman": float(rics.mean()),
            "icir": icir,
            "t_stat": t_stat,
            "n_days": int(len(rows)),
            "coverage_mean": float(ns.mean()),
            "quantile_returns": {f"q{b + 1}": float(qmean[b]) for b in range(quantiles)},
            "long_short_spread": spread,       # 研究度量（q5-q1），非交易信号
            "monotonicity": mono,
            "ic_first_half": ic_first,
            "ic_second_half": ic_second,
            "ic_high_vol": ic_hi,
            "ic_low_vol": ic_lo,
        }

    used_dates = sorted(feats.keys())
    label_dates = sorted(set(panel.index.get_level_values("date")))
    report = {
        "factor": {
            "name": factor.meta.name,
            "version": factor.meta.version,
            "family": factor.meta.family,
            "description": factor.meta.description,
            "economic_hypothesis": factor.meta.economic_hypothesis,
        },
        "start": start,
        "end": end,
        "quantiles": quantiles,
        "horizons": horizons_report,
        "pit_checks": {
            "max_feature_date": max(used_dates) if used_dates else None,
            "max_label_date": max(label_dates) if label_dates else None,
            "feature_used_only_past": True,   # _AsofView 硬切片保证（结构性强制的）
        },
        "metric_notes": {
            "long_short_spread": "research metric only, not a tradable signal",
            "score_semantics": "raw_score; never a probability (P0-007)",
        },
    }
    return _jsonable(report)


def _codes_by_date(panel: pd.DataFrame) -> dict[str, pd.Index]:
    """每个交易日在面板中有行的代码（= 当日实际可交易的 PIT 股票池）。"""
    df = pd.DataFrame({
        "date": panel.index.get_level_values("date").to_numpy(),
        "code": panel.index.get_level_values("code").to_numpy(),
    })
    out: dict[str, pd.Index] = {}
    for date_str, g in df.groupby("date", sort=False):
        out[str(date_str)] = pd.Index(pd.unique(g["code"]), name="code")
    return out
