"""概率校准器纯逻辑测试（mock 数据）。"""

import pandas as pd
import pytest

from engine.probability_calibrator import (
    _all_codes,
    _atr,
    _bin_atr_pct,
    _bin_fund,
    _bin_rps,
    _bin_ret,
    _bin_volume_ratio,
    _build_bins,
    _build_pairs,
    _calendar_plus_trading_days,
    _compute_features_and_future,
    _compute_rps_for_date,
    _fallback_baseline,
    _find_col,
    _fund_inflow_days,
    _mean_ci,
    _trade_dates,
    _wilson_ci,
    calibrate,
    ensure_table,
    load_calibration,
    predict,
)


# --------------------------------------------------------------------------- #
# 内存 Stub DataLoader
# --------------------------------------------------------------------------- #
class StubDL:
    """内存 DataLoader，返回预设 K 线与资金流。"""

    def __init__(self, klines, fund=None):
        self._klines = klines
        self._fund = fund or {}

    def all_spot(self):
        return pd.DataFrame({"代码": list(self._klines.keys())})

    def daily_kline(self, code, days=60, date=None, adjust="qfq"):
        df = self._klines.get(code, pd.DataFrame()).copy()
        if df.empty:
            return df
        df["_dt"] = pd.to_datetime(df["日期"])
        if date:
            end = pd.to_datetime(date)
            df = df[df["_dt"] <= end].copy()
        df = df.drop(columns=["_dt"], errors="ignore")
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)
        return df

    def individual_fund_flow(self, code, fast=False):
        return self._fund.get(code, pd.DataFrame())


def make_kline(prices, start="2024-01-01", volumes=None):
    """从收盘价序列构造含必要列的 K 线。"""
    n = len(prices)
    dates = pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d")
    if volumes is None:
        volumes = [10000] * n
    highs = [max(p * 1.01, p * 0.99) for p in prices]
    lows = [min(p * 1.01, p * 0.99) for p in prices]
    return pd.DataFrame({
        "日期": dates,
        "开盘": prices,
        "收盘": prices,
        "最高": highs,
        "最低": lows,
        "成交量": volumes,
    })


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_calib.db")


# --------------------------------------------------------------------------- #
# 1. 分箱边界
# --------------------------------------------------------------------------- #
def test_bin_rps_boundaries():
    assert _bin_rps(95) == "95-100"
    assert _bin_rps(94.9) == "90-95"
    assert _bin_rps(90) == "90-95"
    assert _bin_rps(89) == "80-90"
    assert _bin_rps(80) == "80-90"
    assert _bin_rps(79) == "70-80"
    assert _bin_rps(70) == "70-80"
    assert _bin_rps(69) == "60-70"
    assert _bin_rps(60) == "60-70"
    assert _bin_rps(59) == "<60"
    assert _bin_rps(0) == "<60"


def test_bin_ret_boundaries():
    assert _bin_ret(0.20) == "20+"
    assert _bin_ret(0.199) == "10-20"
    assert _bin_ret(0.05) == "5-10"
    assert _bin_ret(0.0) == "0-5"
    assert _bin_ret(-0.001) == "-5-0"
    assert _bin_ret(-0.05) == "-5-0"
    assert _bin_ret(-0.051) == "-10--5"
    assert _bin_ret(-0.10) == "-10--5"
    assert _bin_ret(-0.101) == "<-10"


def test_bin_volume_ratio_boundaries():
    assert _bin_volume_ratio(0.5) == "<1.0"
    assert _bin_volume_ratio(1.0) == "1.0-1.5"
    assert _bin_volume_ratio(1.5) == "1.5-2.5"
    assert _bin_volume_ratio(2.5) == "2.5-5"
    assert _bin_volume_ratio(5.0) == "5+"


def test_bin_fund_boundaries():
    assert _bin_fund(0) == "0"
    assert _bin_fund(3) == "3"
    assert _bin_fund(5) == "5"
    assert _bin_fund(10) == "5"


def test_bin_atr_pct_boundaries():
    assert _bin_atr_pct(1.5) == "<2"
    assert _bin_atr_pct(2.0) == "2-4"
    assert _bin_atr_pct(4.0) == "4-7"
    assert _bin_atr_pct(7.0) == "7-12"
    assert _bin_atr_pct(12.0) == "12+"


# --------------------------------------------------------------------------- #
# 2. 指标工具
# --------------------------------------------------------------------------- #
def test_atr_basic():
    highs = pd.Series([11, 12, 13, 12, 14])
    lows = pd.Series([9, 10, 11, 10, 12])
    closes = pd.Series([10, 11, 12, 11, 13])
    atr = _atr(highs, lows, closes, n=3)
    assert atr is not None
    assert atr > 0


def test_atr_insufficient_data():
    highs = pd.Series([11, 12])
    lows = pd.Series([9, 10])
    closes = pd.Series([10, 11])
    assert _atr(highs, lows, closes, n=3) is None


def test_fund_inflow_days():
    df = pd.DataFrame({"主力净流入-净额": [100, 200, -50, 300]})
    # 从最近一天往前：300>0, -50 中断 → 1 天
    assert _fund_inflow_days(df, min_inflow=0) == 1

    df2 = pd.DataFrame({"主力净流入-净额": [100, 200, 300, 400]})
    assert _fund_inflow_days(df2, min_inflow=0) == 4


def test_wilson_ci_extremes():
    assert _wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = _wilson_ci(10, 10)
    assert lo == pytest.approx(0.7225, abs=0.001)
    assert hi == 1.0


def test_mean_ci_basic():
    s = pd.Series([0.01, 0.02, 0.03, 0.04])
    lo, hi = _mean_ci(s)
    assert lo < hi
    assert lo < 0.025 < hi


# --------------------------------------------------------------------------- #
# 3. 交易日序列
# --------------------------------------------------------------------------- #
def test_trade_dates(tmp_db):
    # 构造足够长的 bench K 线
    k = make_kline([10] * 100, start="2024-01-01")
    dl = StubDL({"000001": k})
    dates = _trade_dates(dl, "20240320", months=2)
    assert len(dates) > 30
    # 最后留出 10 个交易日空白
    assert dates[-1] < "20240320"


def test_calendar_plus_trading_days():
    dates = ["20240101", "20240102", "20240103", "20240104", "20240105"]
    assert _calendar_plus_trading_days(dates, "20240101", 5) is None
    assert _calendar_plus_trading_days(dates, "20240101", 2) == "20240103"


def test_all_codes():
    dl = StubDL({"000001": pd.DataFrame(), "000002": pd.DataFrame()})
    codes = _all_codes(dl)
    assert set(codes) == {"000001", "000002"}


# --------------------------------------------------------------------------- #
# 4. 特征计算
# --------------------------------------------------------------------------- #
def test_compute_features_and_future():
    """构造一只均线多头、突破、放量的股票，验证特征值。"""
    # 60 天 K 线，从 2024-02-05（周一）开始上涨，特征日 2024-02-09（周五）已上涨 5 天
    prices = [10.0] * 60
    for i, idx in enumerate(range(35, 60)):
        prices[idx] = 10.0 + (i + 1) * 0.3
    volumes = [10000] * 60
    volumes[39] = 50000  # 特征日放量
    k = make_kline(prices, start="2024-01-01", volumes=volumes)
    dl = StubDL({"000001": k})

    rps_map = {"000001": 88.0}
    # 未来 5/10 日继续使用日历日（mock 数据不区分周末）
    feature_date = "20240209"
    future_dates = {5: "20240214", 10: "20240219"}
    res = _compute_features_and_future(dl, "000001", feature_date, rps_map, future_dates)
    assert res is not None
    feat, futs = res
    assert feat.ma_bull == 1
    assert feat.breakout == 1
    assert feat.volume_ratio == pytest.approx(5.0, abs=0.01)
    assert feat.rps == 88.0
    assert 5 in futs or 10 in futs


def test_compute_rps_for_date():
    """四只股票涨幅递增，RPS 排名应正确。"""
    base = [10] * 22
    klines = {
        "000001": make_kline(base[:-1] + [12], start="2024-01-01"),   # +20%
        "000002": make_kline(base[:-1] + [11], start="2024-01-01"),   # +10%
        "000003": make_kline(base[:-1] + [10.5], start="2024-01-01"), # +5%
        "000004": make_kline(base[:-1] + [10], start="2024-01-01"),   # 0%
    }
    dl = StubDL(klines)
    rps = _compute_rps_for_date(dl, list(klines.keys()), "20240122", workers=2)
    assert len(rps) == 4
    assert rps["000001"] == 100.0
    assert rps["000004"] == 25.0


# --------------------------------------------------------------------------- #
# 5. 分箱统计
# --------------------------------------------------------------------------- #
def test_build_bins():
    """构造两条记录，验证分箱统计结果。"""
    from engine.probability_calibrator import FeatureSnapshot

    feat1 = FeatureSnapshot(
        code="A", date="20240101", rps=85, ret_20d=0.08, ma_bull=1,
        breakout=1, volume_ratio=2.0, fund_inflow_days=3, atr_pct=3.0,
    )
    feat2 = FeatureSnapshot(
        code="B", date="20240101", rps=85, ret_20d=0.08, ma_bull=0,
        breakout=0, volume_ratio=1.0, fund_inflow_days=0, atr_pct=3.0,
    )
    records = [
        (feat1, {5: 0.05, 10: 0.08}),
        (feat2, {5: -0.02, 10: 0.01}),
    ]
    bins = _build_bins(records)
    assert 5 in bins
    assert "rps" in bins[5]
    rps_df = bins[5]["rps"]
    row = rps_df[rps_df["bin_label"] == "80-90"]
    assert not row.empty
    # 两只都在 80-90，1 涨 1 跌，概率 0.5，平均涨幅 (0.05-0.02)/2 = 0.015
    assert row.iloc[0]["prob_up"] == pytest.approx(0.5, abs=0.001)
    assert row.iloc[0]["avg_return"] == pytest.approx(0.015, abs=0.001)


def test_build_pairs():
    from engine.probability_calibrator import FeatureSnapshot

    feat1 = FeatureSnapshot(
        code="A", date="20240101", rps=85, ret_20d=0.08, ma_bull=1,
        breakout=1, volume_ratio=2.0, fund_inflow_days=3, atr_pct=3.0,
    )
    feat2 = FeatureSnapshot(
        code="B", date="20240101", rps=85, ret_20d=0.08, ma_bull=0,
        breakout=0, volume_ratio=1.0, fund_inflow_days=0, atr_pct=3.0,
    )
    records = [
        (feat1, {5: 0.05, 10: 0.08}),
        (feat2, {5: -0.02, 10: 0.01}),
    ]
    pairs = _build_pairs(records)
    assert 5 in pairs
    df = pairs[5]
    assert not df[(df["bin_x"] == "80-90") & (df["bin_y"] == "5-10")].empty


# --------------------------------------------------------------------------- #
# 6. 端到端：calibrate + predict
# --------------------------------------------------------------------------- #
def test_calibrate_and_predict(tmp_db):
    """用两只票、多个交易日做端到端校准并预测。"""
    # 覆盖 end_date 20240215 及未来 10 日，共 70 天
    dates = pd.date_range("2023-12-10", periods=70, freq="D").strftime("%Y-%m-%d")
    # 票 A：持续上涨，均线多头，放量
    prices_a = [10 + i * 0.08 for i in range(70)]
    volumes_a = [10000] * 69 + [80000]
    # 票 B：横盘震荡
    prices_b = [10 + (i % 5) * 0.05 for i in range(70)]
    volumes_b = [10000] * 70

    klines = {
        "000001": pd.DataFrame({
            "日期": dates,
            "开盘": prices_a,
            "收盘": prices_a,
            "最高": [p * 1.01 for p in prices_a],
            "最低": [p * 0.99 for p in prices_a],
            "成交量": volumes_a,
        }),
        "000002": pd.DataFrame({
            "日期": dates,
            "开盘": prices_b,
            "收盘": prices_b,
            "最高": [p * 1.01 for p in prices_b],
            "最低": [p * 0.99 for p in prices_b],
            "成交量": volumes_b,
        }),
    }
    dl = StubDL(klines)
    result = calibrate(dl, end_date="20240215", months=2, workers=2,
                       db_path=tmp_db, max_codes=2)
    assert result["total_samples"] > 0
    assert "run_id" in result

    pred = predict(
        "000001",
        {"rps": 90, "ret_20d": 0.10, "ma_bull": 1, "breakout": 1,
         "volume_ratio": 3.0, "fund_inflow_days": 3, "atr_pct": 2.5},
        horizon=5,
        db_path=tmp_db,
    )
    assert pred.prob_up > 0
    assert pred.sample_count > 0
    assert pred.uncertainty_flag in ("ok", "low_sample", "wide_ci")


def test_predict_fallback_when_no_calibration(tmp_db):
    """没有校准表时预测应回退到基准。"""
    ensure_table(tmp_db)
    pred = predict("000001", {"rps": 80}, horizon=5, db_path=tmp_db)
    assert pred.prob_up == 0.5
    assert pred.uncertainty_flag == "fallback_baseline"


def test_predict_low_sample_flag(tmp_db):
    """小样本区间应标记不确定性。"""
    from engine.probability_calibrator import FeatureSnapshot

    # 只构造 1 条记录，使其 rps 落入 80-90
    feat = FeatureSnapshot(
        code="A", date="20240101", rps=85, ret_20d=0.06, ma_bull=1,
        breakout=1, volume_ratio=2.0, fund_inflow_days=2, atr_pct=2.0,
    )
    records = [(feat, {5: 0.03})]
    bins = _build_bins(records)
    pairs = _build_pairs(records)
    baselines = {5: (0.0, 0.5, 1)}
    run_id = "test_run"
    from engine.probability_calibrator import _write_calibration
    _write_calibration(tmp_db, run_id, "20240101", "20240101", bins, pairs, baselines)

    pred = predict(
        "A",
        {"rps": 85, "ret_20d": 0.06, "ma_bull": 1, "breakout": 1,
         "volume_ratio": 2.0, "fund_inflow_days": 2, "atr_pct": 2.0},
        horizon=5,
        db_path=tmp_db,
        run_id=run_id,
    )
    assert pred.uncertainty_flag == "low_sample"
    # 样本极少时应向基准收缩
    assert pred.prob_up < 0.8


def test_load_calibration(tmp_db):
    """验证写入和读取校准表一致。"""
    from engine.probability_calibrator import FeatureSnapshot

    feat = FeatureSnapshot(
        code="A", date="20240101", rps=85, ret_20d=0.06, ma_bull=1,
        breakout=1, volume_ratio=2.0, fund_inflow_days=2, atr_pct=2.0,
    )
    records = [(feat, {5: 0.03, 10: 0.05})]
    bins = _build_bins(records)
    pairs = _build_pairs(records)
    baselines = {5: (0.03, 1.0, 1), 10: (0.05, 1.0, 1)}
    run_id = "load_test"
    from engine.probability_calibrator import _write_calibration
    _write_calibration(tmp_db, run_id, "20240101", "20240101", bins, pairs, baselines)

    cal = load_calibration(tmp_db, run_id=run_id)
    assert 5 in cal["meta"]
    assert 10 in cal["meta"]
    assert "rps" in cal["bins"][5]
    assert not cal["pairs"][5].empty


# --------------------------------------------------------------------------- #
# 7. fallback 与边界
# --------------------------------------------------------------------------- #
def test_fallback_baseline():
    pred = _fallback_baseline()
    assert pred.prob_up == 0.5
    assert pred.predicted_return == 0.0
    assert pred.ci_high == 1.0
    assert pred.uncertainty_flag == "fallback_baseline"


def test_find_col_fuzzy():
    df = pd.DataFrame({"涨跌幅.1": [1], "名称": ["x"], "成交量": [100]})
    assert _find_col(df, ["涨跌幅"]) == "涨跌幅.1"
    assert _find_col(df, ["成交量"]) == "成交量"
