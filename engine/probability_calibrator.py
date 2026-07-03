"""股票上涨概率预测的历史校准器。

背景：推荐引擎会给出每只候选股的「上涨概率」，但该概率必须经过历史数据校准
才能避免拍脑袋。本模块离线统计「具备哪些特征的股票，未来 N 日上涨概率和平均
涨幅是多少」，并据此提供可解释的查表预测。

用法（离线校准，首次运行较慢）：
    from engine.data_loader import DataLoader
    from engine.probability_calibrator import calibrate, predict

    dl = DataLoader()
    calibrate(dl, end_date="20241231", months=6, workers=8)
    prob, ret, ci, flag = predict("000001", {"rps": 85, "breakout": 1, ...})

设计取舍：
1. 单特征分箱 + 关键二维交互（RPS × 20 日涨幅），避免高维组合稀疏。
2. 预测时多特征加权合成，样本少或置信区间宽时向基准收缩，并明确标注不确定性。
3. 校准结果存 SQLite，避免每次重算；提供 load_calibration() 加载。
4. 历史数据获取依赖 akshare，首次全量校准预计 10-30 分钟，属正常离线任务。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .data_loader import DataLoader, find_col as _find_col

logger = logging.getLogger("pangu.prob")

# --------------------------------------------------------------------------- #
# 常量：回看窗口、预测窗口、分箱定义
# --------------------------------------------------------------------------- #
RPS_WINDOW = 20
BREAKOUT_WINDOW = 20
ATR_PERIOD = 14
HORIZONS = (5, 10)

BENCH_CODE = "000001"  # 上证指数成分股之一，用于生成交易日序列

FEATURE_BINNERS = {
    "rps": lambda f: _bin_rps(f.rps),
    "ret_20d": lambda f: _bin_ret(f.ret_20d),
    "ma_bull": lambda f: "1" if f.ma_bull else "0",
    "breakout": lambda f: "1" if f.breakout else "0",
    "volume_ratio": lambda f: _bin_volume_ratio(f.volume_ratio),
    "fund_inflow_days": lambda f: _bin_fund(f.fund_inflow_days),
    "atr_pct": lambda f: _bin_atr_pct(f.atr_pct),
}

FEATURE_WEIGHTS = {
    "rps": 1.2,
    "ret_20d": 1.0,
    "ma_bull": 0.8,
    "breakout": 0.9,
    "volume_ratio": 0.7,
    "fund_inflow_days": 0.8,
    "atr_pct": 0.6,
}

MIN_SAMPLES_PER_BIN = 30
WIDE_CI_THRESHOLD = 0.30


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSnapshot:
    """某只股票在某个历史交易日的特征快照。"""

    code: str
    date: str
    rps: float
    ret_20d: float
    ma_bull: int
    breakout: int
    volume_ratio: float
    fund_inflow_days: int
    atr_pct: float

    def to_bins(self) -> dict[str, str]:
        return {name: binner(self) for name, binner in FEATURE_BINNERS.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "date": self.date,
            "rps": round(self.rps, 2),
            "ret_20d": round(self.ret_20d * 100, 2),
            "ma_bull": bool(self.ma_bull),
            "breakout": bool(self.breakout),
            "volume_ratio": round(self.volume_ratio, 2),
            "fund_inflow_days": self.fund_inflow_days,
            "atr_pct": round(self.atr_pct, 2),
        }


@dataclass
class PredictionResult:
    """预测输出，强调不确定性标注。"""

    prob_up: float          # 0-1
    predicted_return: float # 小数，如 0.03 表示 3%
    ci_low: float
    ci_high: float
    uncertainty_flag: str   # ok / low_sample / wide_ci / fallback_baseline
    sample_count: int
    contributions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prob_up": round(self.prob_up, 4),
            "predicted_return": round(self.predicted_return, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "uncertainty_flag": self.uncertainty_flag,
            "sample_count": self.sample_count,
            "contributions": self.contributions,
        }


# --------------------------------------------------------------------------- #
# 分箱函数
# --------------------------------------------------------------------------- #
def _bin_rps(v: float) -> str:
    if v >= 95:
        return "95-100"
    if v >= 90:
        return "90-95"
    if v >= 80:
        return "80-90"
    if v >= 70:
        return "70-80"
    if v >= 60:
        return "60-70"
    return "<60"


def _bin_ret(v: float) -> str:
    """20 日累计涨幅（小数）分箱。"""
    pct = v * 100
    if pct >= 20:
        return "20+"
    if pct >= 10:
        return "10-20"
    if pct >= 5:
        return "5-10"
    if pct >= 0:
        return "0-5"
    if pct >= -5:
        return "-5-0"
    if pct >= -10:
        return "-10--5"
    return "<-10"


def _bin_volume_ratio(v: float) -> str:
    if v >= 5:
        return "5+"
    if v >= 2.5:
        return "2.5-5"
    if v >= 1.5:
        return "1.5-2.5"
    if v >= 1.0:
        return "1.0-1.5"
    return "<1.0"


def _bin_fund(v: int) -> str:
    return str(min(int(v), 5))


def _bin_atr_pct(v: float) -> str:
    if v >= 12:
        return "12+"
    if v >= 7:
        return "7-12"
    if v >= 4:
        return "4-7"
    if v >= 2:
        return "2-4"
    return "<2"


# --------------------------------------------------------------------------- #
# 数据库
# --------------------------------------------------------------------------- #
def _db_path(db_path: str = "data/pangu.db") -> Path:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_table(db_path: str = "data/pangu.db") -> None:
    """建 calibration 相关表（如不存在）。"""
    with sqlite3.connect(_db_path(db_path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_meta (
                id INTEGER PRIMARY KEY,
                version TEXT,
                created_at TEXT,
                date_start TEXT,
                date_end TEXT,
                total_samples INTEGER,
                horizon INTEGER,
                avg_return_baseline REAL,
                up_baseline REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_bins (
                id INTEGER PRIMARY KEY,
                created_at TEXT,
                horizon INTEGER,
                feature TEXT,
                bin_label TEXT,
                bin_low REAL,
                bin_high REAL,
                bin_value_int INTEGER,
                prob_up REAL,
                avg_return REAL,
                sample_count INTEGER,
                ci_low REAL,
                ci_high REAL,
                uncertainty_flag TEXT,
                UNIQUE(created_at, horizon, feature, bin_label)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_pairs (
                id INTEGER PRIMARY KEY,
                created_at TEXT,
                horizon INTEGER,
                feature_x TEXT,
                feature_y TEXT,
                bin_x TEXT,
                bin_y TEXT,
                prob_up REAL,
                avg_return REAL,
                sample_count INTEGER,
                ci_low REAL,
                ci_high REAL,
                uncertainty_flag TEXT,
                UNIQUE(created_at, horizon, feature_x, feature_y, bin_x, bin_y)
            )
        """)
        conn.commit()


def _latest_run_id(db_path: str = "data/pangu.db") -> Optional[str]:
    """返回最近一次校准的 created_at。"""
    if not _db_path(db_path).exists():
        return None
    try:
        with sqlite3.connect(_db_path(db_path)) as conn:
            row = conn.execute(
                "SELECT MAX(created_at) FROM calibration_meta"
            ).fetchone()
            return row[0]
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# 指标计算工具
# --------------------------------------------------------------------------- #
def _numeric_series(k: pd.DataFrame, candidates: list[str], default_idx: int) -> pd.Series:
    col = _find_col(k, candidates)
    if col is None:
        # fallback：按列索引
        if len(k.columns) > default_idx:
            col = k.columns[default_idx]
        else:
            raise KeyError(f"无法找到列 {candidates}，且 DataFrame 只有 {len(k.columns)} 列")
    return pd.to_numeric(k[col], errors="coerce").dropna()


def _atr(highs: pd.Series, lows: pd.Series, closes: pd.Series, n: int = 14) -> Optional[float]:
    """平均真实波幅 ATR(n)。"""
    if len(highs) < n + 1 or len(lows) < n + 1 or len(closes) < n + 1:
        return None
    df = pd.DataFrame({"h": highs.values, "l": lows.values, "c": closes.values}).dropna()
    if len(df) < n + 1:
        return None
    prev_close = df["c"].shift(1)
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - prev_close).abs(),
        (df["l"] - prev_close).abs(),
    ], axis=1).max(axis=1).dropna()
    if len(tr) < n:
        return None
    return float(tr.iloc[-n:].mean())


def _fund_inflow_days(fund: pd.DataFrame, min_inflow: float = 0.0) -> int:
    """主力连续净流入天数（从最近一天往前数）。"""
    if len(fund) == 0:
        return 0
    col = _find_col(fund, ["主力净流入-净额", "主力净流入"])
    if col is None:
        col = fund.columns[0]
    net = pd.to_numeric(fund[col], errors="coerce").dropna()
    days = 0
    for v in net.iloc[::-1]:
        if pd.isna(v):
            continue
        if v > min_inflow:
            days += 1
        else:
            break
    return days


def _wilson_ci(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    p = count / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _mean_ci(values: pd.Series, z: float = 1.96) -> tuple[float, float]:
    """正态近似均值置信区间。"""
    n = len(values)
    if n <= 0:
        return 0.0, 0.0
    m = float(values.mean())
    if n < 2:
        return m, m
    se = float(values.std(ddof=1)) / sqrt(n)
    return m - z * se, m + z * se


# --------------------------------------------------------------------------- #
# 交易日序列与全市场代码
# --------------------------------------------------------------------------- #
def _trade_dates(
    dl: DataLoader,
    end_date: str,
    months: int = 6,
) -> list[str]:
    """生成 end_date 往前 months 个月的交易日序列（YYYYmmdd）。

    用 bench_code 的日 K 日期列作为交易日参考。
    """
    days = max(months * 22 + 20, 140)
    k = dl.daily_kline(BENCH_CODE, days=days, date=end_date)
    if len(k) == 0:
        raise RuntimeError(f"无法获取交易日序列（{BENCH_CODE} 在 {end_date} 无数据）")
    date_col = _find_col(k, ["日期", "date"])
    if date_col is None:
        date_col = k.columns[0]
    dates = [str(d).replace("-", "") for d in pd.to_datetime(k[date_col]).dt.strftime("%Y%m%d")]
    dates = sorted(set(dates))
    # 留出最后 10 个交易日，避免未来收益不够
    if len(dates) > 10:
        dates = dates[:-10]
    return dates


def _calendar_plus_trading_days(
    trade_dates: list[str], date: str, offset: int
) -> Optional[str]:
    """从交易日序列中查找 date + offset 个交易日的日历日期。"""
    try:
        idx = trade_dates.index(date)
    except ValueError:
        return None
    target = idx + offset
    if target >= len(trade_dates):
        return None
    return trade_dates[target]


def _all_codes(dl: DataLoader) -> list[str]:
    """当前全市场 A 股代码列表（幸存者偏差，校准用近似）。"""
    spot = dl.all_spot()
    if len(spot) == 0:
        return []
    code_col = _find_col(spot, ["代码", "code"])
    if code_col is None:
        code_col = spot.columns[1]
    codes = [str(x).strip() for x in spot[code_col] if str(x).strip().isdigit()]
    return sorted(set(codes))


# --------------------------------------------------------------------------- #
# 单只股票特征 + 未来收益
# --------------------------------------------------------------------------- #
def _compute_rps_for_date(
    dl: DataLoader,
    codes: list[str],
    date: str,
    workers: int = 8,
) -> dict[str, float]:
    """计算某交易日全市场 20 日 RPS，返回 {code: rps}。"""
    rets: dict[str, float] = {}

    def _one(code: str) -> tuple[str, Optional[float]]:
        try:
            k = dl.daily_kline(code, days=RPS_WINDOW + 5, date=date)
            close_col = _find_col(k, ["收盘", "close"])
            if close_col is None:
                close_col = k.columns[4] if len(k.columns) > 4 else k.columns[-1]
            closes = pd.to_numeric(k[close_col], errors="coerce").dropna()
            if len(closes) < RPS_WINDOW + 1:
                return code, None
            ret = float(closes.iloc[-1] / closes.iloc[-RPS_WINDOW - 1] - 1)
            return code, ret
        except Exception:  # noqa: BLE001
            return code, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, c): c for c in codes}
        for fut in as_completed(futures):
            code, ret = fut.result()
            if ret is not None:
                rets[code] = ret

    if not rets:
        return {}
    series = pd.Series(rets)
    ranks = series.rank(pct=True) * 100
    return {code: float(ranks.get(code, 50.0)) for code in rets}


def _compute_features_and_future(
    dl: DataLoader,
    code: str,
    date: str,
    rps_map: dict[str, float],
    future_dates: dict[int, Optional[str]],
) -> Optional[tuple[FeatureSnapshot, dict[int, float]]]:
    """计算单只股票在 date 的特征以及未来 horizon 收益。

    Returns:
        (FeatureSnapshot, {5: ret5, 10: ret10}) 或 None（数据不足）。
    """
    # 1. 以 date 为终点取历史 K 线
    k = dl.daily_kline(code, days=max(RPS_WINDOW, 30, ATR_PERIOD) + 10, date=date)
    if len(k) < 25:
        return None

    try:
        date_col = _find_col(k, ["日期", "date"])
        if date_col is None:
            date_col = k.columns[0]
        k["_date"] = pd.to_datetime(k[date_col]).dt.strftime("%Y%m%d")
        closes = _numeric_series(k, ["收盘", "close"], 4)
        highs = _numeric_series(k, ["最高", "high"], 2)
        lows = _numeric_series(k, ["最低", "low"], 3)
        vols = _numeric_series(k, ["成交量", "volume"], 5)
    except Exception:  # noqa: BLE001
        return None

    if len(closes) < 21 or len(highs) < 21 or len(lows) < 21 or len(vols) < 6:
        return None

    # 2. 特征计算
    ret_20d = float(closes.iloc[-1] / closes.iloc[-21] - 1)
    ma5 = float(closes.iloc[-5:].mean())
    ma10 = float(closes.iloc[-10:].mean())
    ma20 = float(closes.iloc[-20:].mean())
    ma_bull = int(ma5 > ma10 > ma20)
    breakout = int(float(closes.iloc[-1]) > float(highs.iloc[-21:-1].max()))
    avg_vol = float(vols.iloc[-6:-1].mean())
    volume_ratio = float(vols.iloc[-1]) / avg_vol if avg_vol > 0 else 0.0
    atr = _atr(highs, lows, closes, ATR_PERIOD)
    atr_pct = (atr / float(closes.iloc[-1]) * 100) if atr and closes.iloc[-1] > 0 else 0.0

    # 资金流
    try:
        fund = dl.individual_fund_flow(code, fast=True)
        fund_days = _fund_inflow_days(fund, min_inflow=0.0)
    except Exception:  # noqa: BLE001
        fund_days = 0

    feat = FeatureSnapshot(
        code=code,
        date=date,
        rps=rps_map.get(code, 50.0),
        ret_20d=ret_20d,
        ma_bull=ma_bull,
        breakout=breakout,
        volume_ratio=volume_ratio,
        fund_inflow_days=fund_days,
        atr_pct=atr_pct,
    )

    # 3. 未来收益：取未来最远日期（T+10）的 K 线，然后定位 T
    max_offset = max(future_dates.keys())
    future_cal = future_dates.get(max_offset)
    if not future_cal:
        return None

    try:
        kf = dl.daily_kline(code, days=30, date=future_cal)
        if len(kf) < max_offset + 1:
            return None
        date_col = _find_col(kf, ["日期", "date"])
        if date_col is None:
            date_col = kf.columns[0]
        kf["_date"] = pd.to_datetime(kf[date_col]).dt.strftime("%Y%m%d")
        kf = kf.set_index("_date")
        if date not in kf.index:
            return None
        idx = kf.index.get_loc(date)
        close_col = _find_col(kf, ["收盘", "close"])
        if close_col is None:
            close_col = kf.columns[4] if len(kf.columns) > 4 else kf.columns[-1]
        fcloses = pd.to_numeric(kf[close_col], errors="coerce").dropna()
        base_price = float(fcloses.iloc[idx])
        if base_price <= 0:
            return None
        future_rets: dict[int, float] = {}
        for h in HORIZONS:
            target_idx = idx + h
            if target_idx >= len(fcloses):
                continue
            future_rets[h] = float(fcloses.iloc[target_idx]) / base_price - 1
        if not future_rets:
            return None
    except Exception:  # noqa: BLE001
        return None

    return feat, future_rets


# --------------------------------------------------------------------------- #
# 单日校准任务
# --------------------------------------------------------------------------- #
def _calibrate_one_date(
    args: tuple[DataLoader, str, list[str], dict[int, Optional[str]], int]
) -> list[tuple[FeatureSnapshot, dict[int, float]]]:
    dl, date, codes, future_dates, workers = args
    # 先算当日全市场 RPS
    rps_map = _compute_rps_for_date(dl, codes, date, workers=workers)
    records: list[tuple[FeatureSnapshot, dict[int, float]]] = []
    for code in codes:
        res = _compute_features_and_future(dl, code, date, rps_map, future_dates)
        if res is not None:
            records.append(res)
    return records


# --------------------------------------------------------------------------- #
# 分箱统计
# --------------------------------------------------------------------------- #
def _build_bins(
    records: list[tuple[FeatureSnapshot, dict[int, float]]],
) -> dict[int, dict[str, pd.DataFrame]]:
    """按 horizon 和 feature 做分箱统计。"""
    # 展开成 DataFrame
    rows = []
    for feat, futs in records:
        for h, ret in futs.items():
            rows.append({
                **feat.to_bins(),
                "horizon": h,
                "return": ret,
                "up": int(ret > 0),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    result: dict[int, dict[str, pd.DataFrame]] = {}
    for h in HORIZONS:
        d = df[df["horizon"] == h]
        if d.empty:
            continue
        result[h] = {}
        for feature in FEATURE_BINNERS:
            grp = d.groupby(feature).agg(
                sample_count=("up", "size"),
                up_count=("up", "sum"),
                avg_return=("return", "mean"),
                std_return=("return", "std"),
            ).reset_index()
            grp = grp.rename(columns={feature: "bin_label"})
            grp["prob_up"] = grp["up_count"] / grp["sample_count"]
            grp["ci_low"], grp["ci_high"] = zip(
                *grp.apply(
                    lambda r: _wilson_ci(int(r["up_count"]), int(r["sample_count"])),
                    axis=1,
                )
            )
            mci_low, mci_high = [], []
            for _, r in grp.iterrows():
                sub = d[d[feature] == r["bin_label"]]["return"]
                lo, hi = _mean_ci(sub)
                mci_low.append(lo)
                mci_high.append(hi)
            grp["return_ci_low"] = mci_low
            grp["return_ci_high"] = mci_high
            result[h][feature] = grp
    return result


def _build_pairs(
    records: list[tuple[FeatureSnapshot, dict[int, float]]],
) -> dict[int, pd.DataFrame]:
    """构建 RPS × ret_20d 二维交互分箱。"""
    rows = []
    for feat, futs in records:
        for h, ret in futs.items():
            rows.append({
                "rps_bin": _bin_rps(feat.rps),
                "ret_bin": _bin_ret(feat.ret_20d),
                "horizon": h,
                "return": ret,
                "up": int(ret > 0),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    result: dict[int, pd.DataFrame] = {}
    for h in HORIZONS:
        d = df[df["horizon"] == h]
        if d.empty:
            continue
        grp = d.groupby(["rps_bin", "ret_bin"]).agg(
            sample_count=("up", "size"),
            up_count=("up", "sum"),
            avg_return=("return", "mean"),
        ).reset_index()
        grp["prob_up"] = grp["up_count"] / grp["sample_count"]
        grp["ci_low"], grp["ci_high"] = zip(
            *grp.apply(
                lambda r: _wilson_ci(int(r["up_count"]), int(r["sample_count"])),
                axis=1,
            )
        )
        grp = grp.rename(columns={"rps_bin": "bin_x", "ret_bin": "bin_y"})
        result[h] = grp
    return result


# --------------------------------------------------------------------------- #
# 写入 / 读取校准表
# --------------------------------------------------------------------------- #
def _bin_label_to_bounds(feature: str, label: str) -> tuple[Optional[float], Optional[float], Optional[int]]:
    """把 bin_label 解析成 (low, high, int_value)，用于数据库存储。"""
    if feature in ("ma_bull", "breakout", "fund_inflow_days"):
        try:
            return None, None, int(label)
        except ValueError:
            return None, None, None
    # 连续型解析
    label = label.strip()
    if label.startswith("<"):
        try:
            return float("-inf"), float(label[1:]), None
        except ValueError:
            return None, None, None
    if label.endswith("+"):
        try:
            return float(label[:-1]), float("inf"), None
        except ValueError:
            return None, None, None
    if "-" in label:
        parts = label.split("-")
        try:
            return float(parts[0]), float(parts[1]), None
        except ValueError:
            return None, None, None
    return None, None, None


def _write_calibration(
    db_path: str,
    run_id: str,
    date_start: str,
    date_end: str,
    bins: dict[int, dict[str, pd.DataFrame]],
    pairs: dict[int, pd.DataFrame],
    baselines: dict[int, tuple[float, float, int]],
) -> None:
    """把统计结果写入 SQLite。"""
    ensure_table(db_path)
    with sqlite3.connect(_db_path(db_path)) as conn:
        # 清旧 run（同 run_id 理论上不会出现，但保险）
        conn.execute("DELETE FROM calibration_bins WHERE created_at = ?", (run_id,))
        conn.execute("DELETE FROM calibration_pairs WHERE created_at = ?", (run_id,))
        conn.execute("DELETE FROM calibration_meta WHERE created_at = ?", (run_id,))

        for h, (avg_ret, up_prob, total) in baselines.items():
            conn.execute(
                """INSERT INTO calibration_meta
                   (version, created_at, date_start, date_end, total_samples, horizon,
                    avg_return_baseline, up_baseline)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("1.0", run_id, date_start, date_end, total, h, avg_ret, up_prob),
            )

        for h, fmap in bins.items():
            for feature, df in fmap.items():
                for _, r in df.iterrows():
                    label = str(r["bin_label"])
                    lo, hi, vi = _bin_label_to_bounds(feature, label)
                    n = int(r["sample_count"])
                    p = float(r["prob_up"])
                    ci_low, ci_high = float(r["ci_low"]), float(r["ci_high"])
                    flag = "ok"
                    if n < MIN_SAMPLES_PER_BIN:
                        flag = "low_sample"
                    elif ci_high - ci_low > WIDE_CI_THRESHOLD:
                        flag = "wide_ci"
                    conn.execute(
                        """INSERT INTO calibration_bins
                           (created_at, horizon, feature, bin_label, bin_low, bin_high,
                            bin_value_int, prob_up, avg_return, sample_count, ci_low,
                            ci_high, uncertainty_flag)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (run_id, h, feature, label, lo, hi, vi, p,
                         float(r["avg_return"]), n, ci_low, ci_high, flag),
                    )

        for h, df in pairs.items():
            for _, r in df.iterrows():
                n = int(r["sample_count"])
                p = float(r["prob_up"])
                ci_low, ci_high = float(r["ci_low"]), float(r["ci_high"])
                flag = "ok"
                if n < MIN_SAMPLES_PER_BIN:
                    flag = "low_sample"
                elif ci_high - ci_low > WIDE_CI_THRESHOLD:
                    flag = "wide_ci"
                conn.execute(
                    """INSERT INTO calibration_pairs
                       (created_at, horizon, feature_x, feature_y, bin_x, bin_y,
                        prob_up, avg_return, sample_count, ci_low, ci_high, uncertainty_flag)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, h, "rps", "ret_20d", str(r["bin_x"]), str(r["bin_y"]),
                     p, float(r["avg_return"]), n, ci_low, ci_high, flag),
                )
        conn.commit()


def load_calibration(
    db_path: str = "data/pangu.db",
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """加载最近一次（或指定 run_id）的校准表。

    Returns:
        {
            "meta": {horizon: {...}},
            "bins": {horizon: {feature: DataFrame}},
            "pairs": {horizon: DataFrame},
        }
    """
    run_id = run_id or _latest_run_id(db_path)
    if run_id is None:
        return {}

    with sqlite3.connect(_db_path(db_path)) as conn:
        meta = pd.read_sql(
            "SELECT * FROM calibration_meta WHERE created_at = ?", conn, params=(run_id,)
        )
        bins = pd.read_sql(
            "SELECT * FROM calibration_bins WHERE created_at = ?", conn, params=(run_id,)
        )
        pairs = pd.read_sql(
            "SELECT * FROM calibration_pairs WHERE created_at = ?", conn, params=(run_id,)
        )

    result: dict[str, Any] = {"meta": {}, "bins": {}, "pairs": {}}
    for _, r in meta.iterrows():
        result["meta"][int(r["horizon"])] = dict(r)
    for h in HORIZONS:
        b = bins[bins["horizon"] == h]
        if not b.empty:
            result["bins"][h] = {
                feat: g.reset_index(drop=True)
                for feat, g in b.groupby("feature")
            }
        p = pairs[pairs["horizon"] == h]
        if not p.empty:
            result["pairs"][h] = p.reset_index(drop=True)
    return result


# --------------------------------------------------------------------------- #
# 主入口：calibrate
# --------------------------------------------------------------------------- #
def calibrate(
    dl: DataLoader,
    end_date: Optional[str] = None,
    months: int = 6,
    workers: int = 8,
    db_path: str = "data/pangu.db",
    max_codes: Optional[int] = None,
) -> dict[str, Any]:
    """离线历史校准入口。

    Args:
        dl: DataLoader 实例
        end_date: 校准终点 YYYYMMDD，默认最近交易日
        months: 回看月数
        workers: 并行线程数
        db_path: SQLite 路径
        max_codes: 限制股票数量（None=全市场），用于快速测试

    Returns:
        {"run_id", "date_start", "date_end", "total_samples", "horizons"}
    """
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    except ImportError:  # pragma: no cover
        Progress = None  # type: ignore

    end_date = end_date or datetime.now().strftime("%Y%m%d")
    dates = _trade_dates(dl, end_date, months=months)
    if not dates:
        raise RuntimeError("未能生成交易日序列")

    codes = _all_codes(dl)
    if max_codes:
        codes = codes[:max_codes]

    logger.info(
        "开始概率校准：%s 至 %s，共 %d 个交易日，%d 只股票，%d 线程",
        dates[0], dates[-1], len(dates), len(codes), workers,
    )

    # 预计算每个交易日对应的未来日历日期
    date_future_map: dict[str, dict[int, Optional[str]]] = {}
    for d in dates:
        date_future_map[d] = {
            5: _calendar_plus_trading_days(dates, d, 5),
            10: _calendar_plus_trading_days(dates, d, 10),
        }

    all_records: list[tuple[FeatureSnapshot, dict[int, float]]] = []
    t0 = time.time()

    progress_ctx = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total} 交易日"),
        )
        if Progress
        else None
    )

    def _task_iter():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_calibrate_one_date, (dl, d, codes, date_future_map[d], workers)): d
                for d in dates
            }
            if progress_ctx:
                with progress_ctx as prog:
                    task = prog.add_task("历史概率校准", total=len(dates))
                    for fut in as_completed(futures):
                        d = futures[fut]
                        try:
                            recs = fut.result()
                            all_records.extend(recs)
                            prog.advance(task)
                            prog.update(task, description=f"已处理 {d}")
                        except Exception as e:  # noqa: BLE001
                            logger.warning("日期 %s 校准失败: %s", d, e)
                            prog.advance(task)
            else:
                for fut in as_completed(futures):
                    try:
                        all_records.extend(fut.result())
                    except Exception as e:  # noqa: BLE001
                        logger.warning("日期校准失败: %s", e)

    _task_iter()

    elapsed = time.time() - t0
    logger.info("特征收益收集完成：%d 条记录，耗时 %.1fs", len(all_records), elapsed)

    if not all_records:
        raise RuntimeError("未收集到任何有效特征-收益记录，请检查数据源或日期范围")

    # 分箱统计
    bins = _build_bins(all_records)
    pairs = _build_pairs(all_records)

    # 基准统计
    baselines: dict[int, tuple[float, float, int]] = {}
    for h in HORIZONS:
        rets = [futs[h] for _, futs in all_records if h in futs]
        if rets:
            avg_ret = float(pd.Series(rets).mean())
            up_prob = sum(1 for r in rets if r > 0) / len(rets)
            baselines[h] = (avg_ret, up_prob, len(rets))
        else:
            baselines[h] = (0.0, 0.5, 0)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_calibration(db_path, run_id, dates[0], dates[-1], bins, pairs, baselines)

    logger.info("校准结果已写入 %s，run_id=%s", db_path, run_id)
    return {
        "run_id": run_id,
        "date_start": dates[0],
        "date_end": dates[-1],
        "total_samples": len(all_records),
        "horizons": list(HORIZONS),
        "elapsed_seconds": round(elapsed, 1),
    }


# --------------------------------------------------------------------------- #
# 预测入口
# --------------------------------------------------------------------------- #
def predict(
    code: str,
    features: dict[str, Any],
    horizon: int = 5,
    db_path: str = "data/pangu.db",
    run_id: Optional[str] = None,
) -> PredictionResult:
    """基于历史校准表预测单只股票未来 horizon 日的上涨概率与涨幅。

    Args:
        code: 股票代码（仅用于日志/输出，不影响查表）
        features: {"rps": 85, "ret_20d": 0.08, "ma_bull": 1, "breakout": 1,
                   "volume_ratio": 2.1, "fund_inflow_days": 3, "atr_pct": 3.5}
        horizon: 5 或 10
        db_path: SQLite 路径
        run_id: 指定校准批次，None 用最新

    Returns:
        PredictionResult
    """
    cal = load_calibration(db_path, run_id)
    if not cal or horizon not in cal.get("meta", {}):
        return _fallback_baseline()

    meta = cal["meta"][horizon]
    baseline_prob = float(meta.get("up_baseline", 0.5))
    baseline_ret = float(meta.get("avg_return_baseline", 0.0))
    baseline_n = int(meta.get("total_samples", 0))

    # 构造一个临时 FeatureSnapshot 用于分箱
    try:
        tmp = FeatureSnapshot(
            code=code,
            date="",
            rps=float(features.get("rps", 50)),
            ret_20d=float(features.get("ret_20d", 0)),
            ma_bull=int(features.get("ma_bull", 0)),
            breakout=int(features.get("breakout", 0)),
            volume_ratio=float(features.get("volume_ratio", 1)),
            fund_inflow_days=int(features.get("fund_inflow_days", 0)),
            atr_pct=float(features.get("atr_pct", 0)),
        )
    except (TypeError, ValueError):
        return _fallback_baseline()

    target_bins = tmp.to_bins()
    bin_map = cal.get("bins", {}).get(horizon, {})

    weighted_probs: list[float] = []
    weighted_rets: list[float] = []
    weights: list[float] = []
    total_n = 0
    flags: set[str] = set()
    contributions: list[dict[str, Any]] = []

    for feature, weight in FEATURE_WEIGHTS.items():
        if feature not in bin_map:
            continue
        label = target_bins[feature]
        row = bin_map[feature][bin_map[feature]["bin_label"] == label]
        if row.empty:
            continue
        r = row.iloc[0]
        n = int(r["sample_count"])
        p = float(r["prob_up"])
        ret = float(r["avg_return"])
        ci_low = float(r["ci_low"])
        ci_high = float(r["ci_high"])
        flag = str(r["uncertainty_flag"])
        if flag != "ok":
            flags.add(flag)

        # 权重：样本数 × 置信度（CI 越窄权重越高）
        ci_width = ci_high - ci_low
        w = weight * sqrt(max(n, 1)) / (ci_width + 0.05)
        weighted_probs.append(p)
        weighted_rets.append(ret)
        weights.append(w)
        total_n += n
        contributions.append({
            "feature": feature,
            "bin": label,
            "prob_up": round(p, 4),
            "avg_return": round(ret, 4),
            "sample_count": n,
            "uncertainty_flag": flag,
        })

    # 加入 RPS × ret_20d 交互修正
    pairs_df = cal.get("pairs", {}).get(horizon)
    if pairs_df is not None:
        pair_row = pairs_df[
            (pairs_df["bin_x"] == target_bins["rps"])
            & (pairs_df["bin_y"] == target_bins["ret_20d"])
        ]
        if not pair_row.empty:
            r = pair_row.iloc[0]
            n = int(r["sample_count"])
            p = float(r["prob_up"])
            ret = float(r["avg_return"])
            ci_low = float(r["ci_low"])
            ci_high = float(r["ci_high"])
            flag = str(r["uncertainty_flag"])
            if flag != "ok":
                flags.add(flag)
            ci_width = ci_high - ci_low
            w = 1.5 * sqrt(max(n, 1)) / (ci_width + 0.05)
            weighted_probs.append(p)
            weighted_rets.append(ret)
            weights.append(w)
            total_n += n
            contributions.append({
                "feature": "rps_x_ret_20d",
                "bin": f"{target_bins['rps']}|{target_bins['ret_20d']}",
                "prob_up": round(p, 4),
                "avg_return": round(ret, 4),
                "sample_count": n,
                "uncertainty_flag": flag,
            })

    if not weights:
        return _fallback_baseline()

    total_w = sum(weights)
    prob = sum(p * w for p, w in zip(weighted_probs, weights)) / total_w
    pred_ret = sum(r * w for r, w in zip(weighted_rets, weights)) / total_w

    # 向基准收缩（样本不足或不确定性高时更保守）
    shrink = 0.0
    if total_n < 50:
        shrink = 0.6
    elif total_n < 200:
        shrink = 0.3
    elif flags:
        shrink = 0.15
    prob = prob * (1 - shrink) + baseline_prob * shrink
    pred_ret = pred_ret * (1 - shrink) + baseline_ret * shrink

    # 综合 CI：用加权有效样本近似 + Wilson
    effective_n = int(sum(w for w in weights))
    up_count = int(prob * effective_n)
    ci_low, ci_high = _wilson_ci(up_count, max(effective_n, 1))

    flag = "ok"
    if not flags and effective_n < MIN_SAMPLES_PER_BIN:
        flag = "low_sample"
    elif flags:
        flag = ",".join(sorted(flags))

    return PredictionResult(
        prob_up=prob,
        predicted_return=pred_ret,
        ci_low=ci_low,
        ci_high=ci_high,
        uncertainty_flag=flag,
        sample_count=effective_n,
        contributions=contributions,
    )


def _fallback_baseline() -> PredictionResult:
    return PredictionResult(
        prob_up=0.5,
        predicted_return=0.0,
        ci_low=0.0,
        ci_high=1.0,
        uncertainty_flag="fallback_baseline",
        sample_count=0,
    )
