"""统计验证测试：DSR、块自助、PBO、BH 校正、日收益统计、泄漏审计、稳健性。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.validation.leakage import assert_label_alignment, audit_module
from engine.validation.robustness import (
    DelayedStrategy,
    cost_stress,
    delay_stress,
    param_neighborhood,
    subperiod_metrics,
)
from engine.validation.statistics import (
    bootstrap_ci,
    daily_return_stats,
    deflated_sharpe,
    multiple_hypothesis_report,
    pbo_cscv,
)


class TestDeflatedSharpe:
    def test_noise_is_small_and_penalized_by_trials(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.0, 0.01, size=800)
        dsr1 = deflated_sharpe(r, n_trials=1)
        dsr_k = deflated_sharpe(r, n_trials=1000)
        assert 0.0 < dsr1 < 0.99
        assert dsr_k < dsr1, "more trials must deflate the Sharpe"
        assert dsr_k < 0.5

    def test_real_signal_survives(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0.003, 0.01, size=800)   # 日 SR ≈ 0.3
        assert deflated_sharpe(r, n_trials=1) > 0.95

    def test_provided_variance(self):
        rng = np.random.default_rng(5)
        r = rng.normal(0.0, 0.01, size=300)
        a = deflated_sharpe(r, n_trials=50, sr_variance=1e-4)
        b = deflated_sharpe(r, n_trials=50, sr_variance=1e-2)
        assert a > b   # 方差越大 → 期望最大 SR 越大 → DSR 越小

    def test_degenerate_input(self):
        assert deflated_sharpe([0.0, 0.0, 0.0]) == 0.0
        assert deflated_sharpe([0.01]) == 0.0


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.01, size=500)
        ci = bootstrap_ci(r, np.mean, n=500, seed=1)
        assert ci["lo"] <= ci["point"] <= ci["hi"]
        assert ci["lo"] < ci["hi"]
        assert ci["point"] == pytest.approx(float(np.mean(r)), abs=1e-12)

    def test_custom_stat_and_block(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.0, 0.02, size=300)
        ci = bootstrap_ci(r, lambda x: np.mean(x) / np.std(x) * np.sqrt(252),
                          n=300, seed=7, block=10)
        assert ci["block"] == 10
        assert ci["lo"] <= ci["point"] <= ci["hi"]


class TestPBO:
    def test_noise_matrix_near_half(self):
        rng = np.random.default_rng(11)
        m = rng.normal(0.0, 0.01, size=(240, 8))
        out = pbo_cscv(m, S=6)
        assert out["n_combinations"] == 20   # C(6,3)
        assert abs(out["pbo"] - 0.5) <= 0.25

    def test_real_signal_low_pbo(self):
        rng = np.random.default_rng(12)
        m = rng.normal(0.0, 0.01, size=(240, 8))
        m[:, 0] += 0.01                      # 第一列稳定占优
        out = pbo_cscv(m, S=6)
        assert out["pbo"] < 0.2

    def test_needs_two_columns(self):
        assert pbo_cscv(np.ones((10, 1))) == {"pbo": 0.0, "n_combinations": 0}
        with pytest.raises(ValueError):
            pbo_cscv(np.ones((3, 4)), S=6)


class TestMultipleHypothesis:
    def test_bh_qvalues_and_counts(self):
        ps = [0.001, 0.008, 0.039, 0.041, 0.2, 0.45, 0.5, 0.6, 0.8, 0.9]
        exps = [{"experiment_id": f"e{i}", "p_value": p} for i, p in enumerate(ps)]
        rep = multiple_hypothesis_report(exps)
        assert rep["n_experiments"] == 10
        assert rep["n_with_p"] == 10
        assert rep["n_significant_raw"] == 4
        qs = [e["q_value_bh"] for e in rep["experiments"]]
        assert qs[0] == pytest.approx(0.01)          # 最小 p × m / 1
        # BH 单调性（按 p 升序 q 不降）
        order = np.argsort(ps)
        q_sorted = [qs[i] for i in order]
        assert all(a <= b + 1e-12 for a, b in zip(q_sorted, q_sorted[1:]))
        assert rep["n_significant_bh"] == sum(1 for q in qs if q < 0.05)

    def test_experiments_without_p(self):
        rep = multiple_hypothesis_report([{"experiment_id": "x"},
                                          {"experiment_id": "y", "p_value": 0.01}])
        assert rep["n_with_p"] == 1
        assert rep["experiments"][0]["q_value_bh"] is None


class TestDailyReturnStats:
    def test_monotonic_up(self):
        idx = [d.strftime("%Y-%m-%d")
               for d in pd.bdate_range("2025-01-06", periods=100)]
        s = pd.Series(1.0 * (1.001 ** np.arange(100)), index=idx)
        out = daily_return_stats(s)
        assert out["n_days"] == 99
        assert out["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
        assert out["positive_day_ratio"] == pytest.approx(1.0)
        assert out["sharpe"] > 0
        assert out["mdd_start"] == idx[0] and out["mdd_end"] == idx[0]

    def test_drawdown_dates(self):
        idx = ["d0", "d1", "d2", "d3"]
        s = pd.Series([100.0, 120.0, 60.0, 90.0], index=idx)
        out = daily_return_stats(s)
        assert out["max_drawdown"] == pytest.approx(0.5)
        assert out["mdd_start"] == "d1" and out["mdd_end"] == "d2"
        assert out["worst_day"] == pytest.approx(-0.5)

    def test_accepts_dataframe(self):
        df = pd.DataFrame({"equity": [100.0, 110.0, 105.0]})
        assert daily_return_stats(df)["n_days"] == 2


class TestLeakageAudit:
    def test_flags_bad_patterns(self, tmp_path):
        p = tmp_path / "bad.py"
        p.write_text(
            "a = df.shift(-1)\n"
            "b = df.merge(other, direction='forward')\n"
            "c = df.bfill()\n"
            "d = df['x'].cummax()\n"
            "e = df.iloc[::-1]\n"
            "f = features + df['label']\n",
            encoding="utf-8")
        findings = audit_module(p)
        reasons = {f["reason"] for f in findings}
        assert len(findings) >= 6
        assert any("负向 shift" in r for r in reasons)
        assert any("向后填充" in r for r in reasons)
        assert any("cummax" in r for r in reasons)
        assert any("label" in r for r in reasons)

    def test_clean_module_no_findings(self, tmp_path):
        p = tmp_path / "clean.py"
        p.write_text("x = df.shift(1)\ny = df.ffill()\n", encoding="utf-8")
        assert audit_module(p) == []

    def test_missing_file(self, tmp_path):
        assert audit_module(tmp_path / "nope.py") == []

    def test_label_alignment(self):
        assert_label_alignment(["2025-01-01", "2025-01-02"],
                               ["2025-01-02", "2025-01-03"])
        with pytest.raises(Exception):   # LookaheadError
            assert_label_alignment(["2025-01-02"], ["2025-01-02"])
        with pytest.raises(Exception):
            assert_label_alignment(["2025-01-03"], ["2025-01-02"])
        with pytest.raises(Exception):
            assert_label_alignment(["2025-01-01"], ["2025-01-02", "2025-01-03"])


class TestRobustness:
    def test_param_neighborhood_fragile_flag(self):
        # 仅 base 参数盈利 → fragile
        calls = {}

        def run_fn(**params):
            calls[len(calls)] = params
            return {"summary": {"total_return":
                                0.1 if params == {"x": 1.0, "y": 2.0} else -0.05}}

        out = param_neighborhood(run_fn, {"x": 1.0, "y": 2.0}, ["x", "y"])
        assert out["base"] == pytest.approx(0.1)
        assert out["fragile"] is True
        assert set(out["results"]["x"]) == {"0.8", "0.9", "1.1", "1.2"}

    def test_param_neighborhood_robust(self):
        def run_fn(**params):
            return {"summary": {"total_return": 0.1 * params.get("x", 1.0)}}

        out = param_neighborhood(run_fn, {"x": 1.0}, ["x"])
        assert out["fragile"] is False

    def test_cost_stress(self):
        out = cost_stress(lambda m: {"summary": {"total_return": 0.1 / m - 0.02}})
        assert out["results"]["1"] == pytest.approx(0.08)
        assert out["results"]["3"] == pytest.approx(0.1 / 3 - 0.02)
        assert out["survives_2x"] is True

    def test_delay_stress(self):
        out = delay_stress(lambda d: {"total_return": 0.1 if d == 0 else -0.01})
        assert out["results"]["0"] == pytest.approx(0.1)
        assert out["results"]["1"] == pytest.approx(-0.01)

    def test_delayed_strategy_shifts_decisions(self):
        days = [d.strftime("%Y-%m-%d")
                for d in pd.bdate_range("2025-01-06", periods=6)]
        seen = []

        class S:
            def rebalance(self, t, view):
                seen.append((t, view.asof))
                return []

        s = DelayedStrategy(S(), days, delay=2)
        s.rebalance(days[3], _FakeView(days[3]))
        assert seen == [(days[1], days[1])]
        # 延迟期内（前 delay 天）不产生决策
        seen.clear()
        s.rebalance(days[1], _FakeView(days[1]))
        assert seen == []

    def test_subperiod_metrics(self):
        idx = [d.strftime("%Y-%m-%d")
               for d in pd.bdate_range("2024-11-01", periods=60)]
        s = pd.Series(100.0 + np.arange(60), index=idx)
        out = subperiod_metrics(s, regime_fn=lambda d: "up" if d < idx[30] else "down")
        assert set(out["by_year"]) == {"2024", "2025"}
        assert set(out["by_regime"]) == {"up", "down"}
        for grp in list(out["by_year"].values()) + list(out["by_regime"].values()):
            assert grp["total_return"] > 0
            assert grp["n_days"] >= 0


class _FakeView:
    def __init__(self, asof):
        self.asof = asof

    def asof_view(self, date):
        return _FakeView(date)
