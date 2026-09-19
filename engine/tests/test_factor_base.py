"""P2-001 因子 API 基础设施测试：FactorMeta / Factor / 预处理 / 注册表。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.research.data_interface import PITResearchData, SyntheticResearchData
from engine.research.factors.base import (
    Factor,
    FactorMeta,
    FactorUnavailableError,
    rank_zscore,
    winsorize_zscore,
)
from engine.research.factors.registry import FactorRegistry, build_default_registry, code_hash


# ---------------------------------------------------------------------------
# 测试用哑因子
# ---------------------------------------------------------------------------

class _IdentityFactor(Factor):
    def __init__(self, standardize="zscore", winsorize=(0.01, 0.99)):
        self.meta = FactorMeta(
            name="identity_test", version="1.0.0", family="test",
            description="恒等哑因子（仅测试用）",
            required_fields=("close",), lookback_days=1,
            economic_hypothesis="无（测试桩）", missing_rule="不处理",
            winsorize=winsorize, standardize=standardize,
        )

    def compute(self, asof, universe, data):
        return pd.Series(1.0, index=pd.Index([str(c) for c in universe]))


class _UnavailableFactor(Factor):
    def __init__(self):
        self.meta = FactorMeta(
            name="never_available", version="1.0.0", family="test",
            description="总是不可用（测试桩）",
            required_fields=(), lookback_days=0,
            economic_hypothesis="无（测试桩）", missing_rule="直接抛错",
        )

    def compute(self, asof, universe, data):
        raise FactorUnavailableError("no source in test stub")


# ---------------------------------------------------------------------------
# FactorMeta
# ---------------------------------------------------------------------------

def test_factor_meta_is_frozen():
    meta = FactorMeta(
        name="x", version="1", family="t", description="d",
        required_fields=("close",), lookback_days=5,
        economic_hypothesis="h", missing_rule="m",
    )
    with pytest.raises(Exception):
        meta.name = "y"


def test_factor_is_abstract():
    with pytest.raises(TypeError):
        Factor()  # type: ignore[abstract]


def test_compute_returns_raw_score_series():
    f = _IdentityFactor()
    data = SyntheticResearchData(seed=3, n_symbols=12, n_days=60)
    asof = data.dates[-10]
    uni = data.universe(asof).index
    out = f.compute(asof, uni, data)
    assert isinstance(out, pd.Series)
    assert list(out.index) == [str(c) for c in uni]


def test_unavailable_factor_raises_honest_error():
    f = _UnavailableFactor()
    data = SyntheticResearchData(seed=3, n_symbols=12, n_days=60)
    asof = data.dates[-5]
    with pytest.raises(FactorUnavailableError):
        f.compute(asof, data.universe(asof).index, data)
    assert issubclass(FactorUnavailableError, RuntimeError)


# ---------------------------------------------------------------------------
# 预处理：winsorize_zscore / rank_zscore
# ---------------------------------------------------------------------------

def _sample_scores() -> pd.Series:
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(size=200))
    s.iloc[0] = 50.0    # 右侧离群
    s.iloc[1] = -50.0   # 左侧离群
    return s


def test_winsorize_zscore_clips_and_standardizes():
    raw = _sample_scores()
    out = winsorize_zscore(raw, (0.01, 0.99))
    assert out.notna().all()
    assert abs(out.mean()) < 1e-9
    assert abs(out.std() - 1.0) < 1e-9
    # 离群值被截到分位边界 -> 不再超过样本内极值
    lo, hi = raw.quantile(0.01), raw.quantile(0.99)
    assert out.max() <= ((hi - raw.clip(lo, hi).mean()) / raw.clip(lo, hi).std() + 1e-9)
    assert out.min() >= -abs(out.min()) - 1e-9


def test_winsorize_preserves_interior_order():
    raw = _sample_scores()
    out = winsorize_zscore(raw, (0.01, 0.99))
    masked = raw.between(raw.quantile(0.05), raw.quantile(0.95))
    a = raw[masked].rank()
    b = out[masked].rank()
    assert abs(np.corrcoef(a, b)[0, 1]) > 0.999  # 内部次序完全保留


def test_rank_zscore_is_order_preserving_and_standardized():
    rng = np.random.default_rng(7)
    raw = pd.Series(rng.lognormal(size=100))  # 重尾
    out = rank_zscore(raw)
    assert abs(np.corrcoef(raw.rank(), out)[0, 1]) > 0.9999
    assert abs(out.mean()) < 1e-9
    assert abs(out.std() - 1.0) < 1e-9


def test_zscore_of_constant_series_is_zeros_not_nan():
    s = pd.Series(3.14, index=[f"c{i}" for i in range(10)])
    for out in (winsorize_zscore(s), rank_zscore(s)):
        assert out.notna().all()
        assert (out == 0).all()


def test_finalize_honors_standardize_rule():
    data = SyntheticResearchData(seed=5, n_symbols=30, n_days=80)
    asof = data.dates[-5]
    uni = data.universe(asof).index

    f_raw = _IdentityFactor(standardize="none")
    assert (f_raw.finalize(pd.Series([1.0, 2.0])) == pd.Series([1.0, 2.0])).all()
    with pytest.raises(ValueError):
        _IdentityFactor(standardize="bogus").finalize(pd.Series([1.0, 2.0]))
    assert isinstance(f_raw.compute(asof, uni, data), pd.Series)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


def test_registry_names_never_imply_probability(registry):
    """P0-007：因子输出是 raw_score，命名绝不能暗示概率。"""
    for rec in registry.list():
        name = rec["name"].lower()
        assert "prob" not in name, f"factor name implies probability: {name}"
        assert name == rec["name"].strip()


def test_registry_register_get_list(registry):
    names = [r["name"] for r in registry.list()]
    assert len(names) == len(set(names))
    f = registry.get("mom_20d")
    assert isinstance(f, Factor)
    assert registry.availability("mom_20d") == "ready"
    with pytest.raises(KeyError):
        registry.get("no_such_factor")
    with pytest.raises(ValueError):
        registry.register(registry.get("mom_20d"))  # 重复注册


def test_registry_code_hash_stable_and_discriminating():
    a = code_hash(_IdentityFactor())
    b = code_hash(_IdentityFactor())
    assert a == b and len(a) == 64
    assert a != code_hash(_UnavailableFactor())


def test_registry_degraded_availability(registry):
    assert registry.availability("earnings_quality") == "degraded_no_source"
    assert registry.availability("value_pe") == "degraded_no_source"


def test_default_registry_covers_required_factor_names():
    names = {r["name"] for r in build_default_registry().list()}
    required = {
        "mom_5d", "mom_10d", "mom_20d", "mom_60d", "rps_20d", "ma20_slope",
        "breakout_20d", "high_52w_proximity", "vol_adj_mom_20d", "trend_persistence_20d",
        "rev_1d", "rev_3d", "rev_5d", "rsi_14", "index_adj_rev_5d",
        "realized_vol_20d", "downside_vol_20d",
        "turnover_20d_avg", "amihud_20d", "volume_ratio_5_20", "amount_accel_5d",
        "volume_price_div_20d", "turnover_persistence_10d",
        "limit_up_count_20d", "days_since_limit_up", "consec_limit_up_days",
        "near_limit_rate_5d",
        "mkt_breadth_5d", "mkt_vol_20d",
        "earnings_quality", "value_pe",
    }
    missing = required - names
    assert not missing, f"missing factors: {missing}"


# ---------------------------------------------------------------------------
# PITResearchData：pit_store 未就绪时给出清晰指引
# ---------------------------------------------------------------------------

def test_pit_research_data_lazy_import_behavior():
    try:
        import engine.data.pit_store  # noqa: F401
        has_store = True
    except Exception:
        has_store = False

    if has_store:
        pit = PITResearchData()
        assert callable(pit.daily_panel)
    else:
        with pytest.raises(ImportError) as ei:
            PITResearchData()
        assert "pit_store" in str(ei.value)
        assert "SyntheticResearchData" in str(ei.value)
