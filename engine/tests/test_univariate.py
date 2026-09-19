"""P2-003 单因子诊断测试：标签构造 / 注入 alpha 的 regime 检出 / 噪声对照 / 前视守卫 / 落盘。"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine.research import (
    SyntheticResearchData,
    Factor,
    FactorMeta,
    evaluate_factor,
    forward_returns,
    write_factor_report,
)
from engine.research.factors.library import MomentumFactor

HORIZONS = (1, 3, 5, 10, 20)


# ---------------------------------------------------------------------------
# 测试因子
# ---------------------------------------------------------------------------

class OracleQuality(Factor):
    """直接揭示合成数据注入的 quality 隐变量（oracle，仅测试用）。"""

    def __init__(self, quality: np.ndarray, codes: list[str]):
        self._series = pd.Series(quality, index=pd.Index(codes, name="code"))
        self.meta = FactorMeta(
            name="oracle_quality", version="1.0.0", family="test",
            description="合成 quality 隐变量的 oracle 揭示因子",
            required_fields=(), lookback_days=1,
            economic_hypothesis="quality 注入期内与未来收益正相关（设计如此）。",
            missing_rule="universe 之外的股票 NaN。",
        )

    def compute(self, asof, universe, data):
        return self._series.reindex(pd.Index([str(c) for c in universe]))


class PureNoiseFactor(Factor):
    """按决策日确定性的纯噪声因子（对照：|IC| 应接近 0）。"""

    def __init__(self):
        self.meta = FactorMeta(
            name="pure_noise", version="1.0.0", family="test",
            description="确定性纯噪声对照因子",
            required_fields=(), lookback_days=1,
            economic_hypothesis="无（对照桩）。",
            missing_rule="无。",
        )

    def compute(self, asof, universe, data):
        rng = np.random.default_rng(int(asof.replace("-", "")) % 10 ** 8)
        return pd.Series(rng.normal(size=len(universe)),
                         index=pd.Index([str(c) for c in universe]))


# ---------------------------------------------------------------------------
# fixtures（合成数据生成与评估较重，模块级复用）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alpha_data():
    # alpha 只注入前 200 个交易日（regime 依赖）；提高 alpha_bp/降低波动
    # 是为了让 oracle IC 在可检测量级（默认 2bp 属真实弱信号，IC~0.04）。
    return SyntheticResearchData(seed=11, n_symbols=120, n_days=400,
                                 alpha_bp=25.0, alpha_days=200,
                                 sigma_lo=0.010, sigma_hi=0.018)


@pytest.fixture(scope="module")
def alpha_report(alpha_data):
    codes = list(alpha_data.codes)
    # 决策日横跨 alpha regime 边界（第 200 天）：前半有信号、后半无
    return evaluate_factor(OracleQuality(alpha_data.quality, codes),
                           alpha_data, alpha_data.dates[30], alpha_data.dates[350],
                           horizons=HORIZONS)


# ---------------------------------------------------------------------------
# 前向收益（标签）构造
# ---------------------------------------------------------------------------

def _tiny_panel():
    rows = []
    for code, rets in {"A": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       "B": [0.0, 2.0, -2.0, 0.0, 2.0, 2.0]}.items():
        for t, r in enumerate(rets):
            rows.append({"date": f"2024-01-{t + 2:02d}", "code": code,
                         "pct_change": r})
    df = pd.DataFrame(rows).set_index(["date", "code"]).sort_index()
    df.index.names = ["date", "code"]
    return df


def test_forward_returns_are_sum_of_next_h_days():
    panel = _tiny_panel()
    fwd = forward_returns(panel, 2)
    # A: 全 1% -> 每个决策行 = 后两天之和 = 2%（末两行为 NaN）
    a = fwd.xs("A", level="code")
    assert a.iloc[0] == pytest.approx(0.02)
    assert a.iloc[3] == pytest.approx(0.02)
    assert pd.isna(a.iloc[4]) and pd.isna(a.iloc[5])
    # B: 行1 = 第2/3天之和 = (-2% + 0%) = -2%；行0 = 第1/2天之和 = 0；
    #    行3 = 第4/5天之和 = (+2% + 2%) = +4%
    b = fwd.xs("B", level="code")
    assert b.iloc[1] == pytest.approx(-0.02)
    assert b.iloc[0] == pytest.approx(0.0)
    assert b.iloc[3] == pytest.approx(0.04)


def test_forward_returns_never_touch_feature_direction():
    """负向 shift 只出现在标签内部；标签值必须等于“未来”收益之和而非过去。"""
    panel = _tiny_panel()
    fwd = forward_returns(panel, 1)
    rets = panel["pct_change"] / 100.0
    for code in ("A", "B"):
        r = rets.xs(code, level="code")
        f = fwd.xs(code, level="code")
        assert f.iloc[0] == pytest.approx(r.iloc[1])  # 决策行 0 看到的是第 1 天收益
        assert f.iloc[1] == pytest.approx(r.iloc[2])


# ---------------------------------------------------------------------------
# 注入 alpha：regime 切分机制
# ---------------------------------------------------------------------------

def test_injected_alpha_detectable_in_first_half_only(alpha_report):
    for h in ("5", "10", "20"):
        hz = alpha_report["horizons"][h]
        assert hz["ic_mean_spearman"] is not None
        # 前半程（alpha regime）rank IC 显著为正
        assert hz["ic_first_half"] > 0.2, (h, hz["ic_first_half"])
        # regime 机制：前半 > 后半（后半程 alpha 已停注入）
        assert hz["ic_first_half"] > hz["ic_second_half"], (h, hz)


def test_injected_alpha_icir_and_quantile_machinery(alpha_report):
    hz = alpha_report["horizons"]["20"]
    assert hz["icir"] > 0.5
    assert hz["t_stat"] > 2.0
    qs = hz["quantile_returns"]
    assert set(qs) == {"q1", "q2", "q3", "q4", "q5"}
    # 高分位组合均值收益应高于低分位（研究度量，非交易信号）
    assert qs["q5"] > qs["q1"]
    assert hz["long_short_spread"] > 0
    assert hz["monotonicity"] is not None and hz["monotonicity"] > 0.5
    # 波动率 regime 两半均有记录（机制在跑，不预设方向）
    assert hz["ic_high_vol"] is not None
    assert hz["ic_low_vol"] is not None


# ---------------------------------------------------------------------------
# 纯噪声对照
# ---------------------------------------------------------------------------

def test_pure_noise_factor_has_no_mean_ic(alpha_data):
    report = evaluate_factor(PureNoiseFactor(), alpha_data,
                             alpha_data.dates[30], alpha_data.dates[350],
                             horizons=HORIZONS)
    for h, hz in report["horizons"].items():
        assert abs(hz["ic_mean_spearman"]) < 0.1, (h, hz["ic_mean_spearman"])
        assert abs(hz["ic_mean_pearson"]) < 0.1, (h, hz["ic_mean_pearson"])


# ---------------------------------------------------------------------------
# 前视守卫：evaluate 永不读 end 之后的行
# ---------------------------------------------------------------------------

class CorruptAfterEnd:
    """包装合成数据：把 end_target 之后的行全部污染为 NaN 再提供服务。

    若 evaluate_factor 以任何方式读取 end 之后的行，结果必然不同。
    """

    def __init__(self, inner: SyntheticResearchData, end_target: str):
        self._inner = inner
        self._end_target = end_target
        full = inner.daily_panel(inner.dates[0], inner.dates[-1])
        full = full.copy()
        full["is_st"] = full["is_st"].astype(float)
        bad = full.index.get_level_values("date") > end_target
        full.loc[bad, :] = np.nan
        self._full = full.sort_index()

    def daily_panel(self, start, end, symbols=None):
        lvl = self._full.index.get_level_values("date")
        sub = self._full[(lvl >= start) & (lvl <= end)]
        if symbols is not None:
            sub = sub[sub.index.get_level_values("code").isin(set(symbols))]
        return sub.copy()

    def universe(self, date):
        return self._inner.universe(date)

    def index_daily(self, code, start, end):
        return self._inner.index_daily(code, start, end)

    def trading_days(self, start, end):
        return self._inner.trading_days(start, end)


def test_evaluate_never_reads_after_end():
    clean = SyntheticResearchData(seed=21, n_symbols=60, n_days=300)
    start, end = clean.dates[120], clean.dates[280]
    factor = MomentumFactor(5)

    rep_clean = evaluate_factor(factor, clean, start, end, horizons=(1, 5))
    corrupted = CorruptAfterEnd(clean, end_target=end)
    rep_corrupt = evaluate_factor(factor, corrupted, start, end, horizons=(1, 5))

    assert json.dumps(rep_clean, sort_keys=True) == json.dumps(rep_corrupt, sort_keys=True)


def test_pit_checks_recorded(alpha_report, alpha_data):
    pit = alpha_report["pit_checks"]
    assert pit["feature_used_only_past"] is True
    assert pit["max_feature_date"] <= alpha_report["end"]
    assert pit["max_label_date"] <= alpha_report["end"]
    assert pit["max_feature_date"] >= alpha_report["start"]


# ---------------------------------------------------------------------------
# evaluate 报告结构与 min_history
# ---------------------------------------------------------------------------

def test_report_structure_and_horizon_keys(alpha_report):
    assert alpha_report["factor"]["name"] == "oracle_quality"
    assert set(alpha_report["horizons"]) == {str(h) for h in HORIZONS}
    hz = alpha_report["horizons"]["5"]
    for key in ("ic_mean_pearson", "ic_mean_spearman", "icir", "t_stat", "n_days",
                "coverage_mean", "quantile_returns", "long_short_spread",
                "monotonicity", "ic_first_half", "ic_second_half",
                "ic_high_vol", "ic_low_vol"):
        assert key in hz, key
    assert alpha_report["metric_notes"]["score_semantics"].startswith("raw_score")


def test_min_history_drops_early_decision_dates(alpha_data):
    codes = list(alpha_data.codes)
    f = OracleQuality(alpha_data.quality, codes)
    r1 = evaluate_factor(f, alpha_data, alpha_data.dates[30], alpha_data.dates[80],
                         horizons=(5,))
    r2 = evaluate_factor(f, alpha_data, alpha_data.dates[30], alpha_data.dates[80],
                         horizons=(5,), min_history=10)
    d1 = pd.Timestamp(r1["pit_checks"]["max_feature_date"])
    d2 = pd.Timestamp(r2["pit_checks"]["max_feature_date"])
    assert d2 <= d1  # 丢弃最早的决策日后，特征终点不后移
    assert r2["horizons"]["5"]["n_days"] <= r1["horizons"]["5"]["n_days"]


# ---------------------------------------------------------------------------
# 落盘 reporting
# ---------------------------------------------------------------------------

def test_write_factor_report_json_and_index(alpha_report, alpha_data, tmp_path):
    codes = list(alpha_data.codes)
    factor = OracleQuality(alpha_data.quality, codes)
    out_dir = tmp_path / "factors"
    res = write_factor_report(factor, alpha_report, out_dir=str(out_dir))

    report_path = out_dir / "oracle_quality.json"
    assert res["report_path"] == str(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["name"] == "oracle_quality"
    assert payload["pit_checks"]["feature_used_only_past"] is True
    assert "horizons" in payload

    index_path = tmp_path / "factor_index.jsonl"
    assert res["index_path"] == str(index_path)
    lines = [json.loads(x) for x in
             index_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 1
    line = lines[0]
    assert line["name"] == "oracle_quality"
    assert line["version"] == "1.0.0"
    assert line["family"] == "test"
    assert line["status"] == "evaluated"
    assert "ts" in line and "5" in line["horizons"]


def test_write_factor_report_refuses_report_without_pit_checks(tmp_path):
    factor = PureNoiseFactor()
    with pytest.raises(ValueError):
        write_factor_report(factor, {"horizons": {}}, out_dir=str(tmp_path / "f"))
