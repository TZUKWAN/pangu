"""+5% 目标命中概率模块测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.decision.target_prob import (BUY_MIN_N, BUY_MIN_WILSON_LB,
                                         TargetHitTable, outcome_for_bars,
                                         rev_bucket, vol_bucket, wilson_lb)


class TestOutcome:
    def test_hit_when_high_reaches_target(self):
        res, ret = outcome_for_bars(10.0, np.array([10.4, 10.6]), np.array([9.9, 10.0]),
                                    0.05, 0.05)
        assert res == "hit" and ret == 0.05

    def test_stopped_first_when_both_in_same_bar(self):
        res, ret = outcome_for_bars(10.0, np.array([10.6]), np.array([9.4]),
                                    0.05, 0.05)
        assert res == "stopped" and ret == -0.05     # 保守：双触计止损

    def test_timeout(self):
        res, _ = outcome_for_bars(10.0, np.array([10.1, 10.2]), np.array([9.8, 9.9]),
                                  0.05, 0.05)
        assert res == "timeout"

    def test_invalid_entry(self):
        assert outcome_for_bars(float("nan"), np.array([1]), np.array([1]),
                                0.05, 0.05)[0] == "invalid"


class TestWilson:
    def test_known_values(self):
        assert wilson_lb(0, 0) == 0.0
        lb = wilson_lb(30, 100)
        assert 0.21 < lb < 0.23                      # 30% 命中 n=100 → 下界 ≈ 0.22
        assert wilson_lb(100, 100) > 0.95

    def test_small_n_penalized(self):
        assert wilson_lb(3, 10) < wilson_lb(30, 100)  # 同比率下小样本下界更低


class TestBuckets:
    def test_rev_buckets(self):
        assert rev_bucket(-2.0) == "rev_q1"
        assert rev_bucket(0.5) == "rev_q3"
        assert rev_bucket(None) == "rev_na"
        assert rev_bucket(float("nan")) == "rev_na"

    def test_vol_buckets(self):
        assert vol_bucket(0.01) == "vol_low"
        assert vol_bucket(0.03) == "vol_mid"
        assert vol_bucket(0.06) == "vol_high"


def _panel_with_outcome():
    """3 只票 × 12 天：人为构造 票A(hit) 票B(stopped) 票C(timeout)。"""
    days = pd.bdate_range("2026-08-03", periods=12).strftime("%Y-%m-%d")
    codes = ["A", "B", "C"]
    close = pd.DataFrame(100.0, index=days, columns=codes)
    high = pd.DataFrame(100.5, index=days, columns=codes)
    low = pd.DataFrame(99.5, index=days, columns=codes)
    open_ = pd.DataFrame(100.0, index=days, columns=codes)
    # 决策日 = 第 5 天（index 4），窗口 = 后 5 天（index 5..9）
    close.iloc[7, 0] = 106.0; high.iloc[7, 0] = 106.5     # A: +6% → hit
    low.iloc[7, 1] = 93.0; high.iloc[7, 1] = 103.0        # B: 触止损（同bar双触→stopped）
    # C: 全程 ±0.5% → timeout
    panel = pd.concat([
        close.stack().rename("close"), open_.stack().rename("open"),
        high.stack().rename("high"), low.stack().rename("low")], axis=1)
    panel.index.names = ["date", "code"]
    return panel


class TestTableBuildAndLookup:
    def test_build_counts_match_hand_computed(self):
        panel = _panel_with_outcome()
        days = sorted(panel.index.get_level_values("date").unique())
        rz = pd.Series(0.0, index=pd.MultiIndex.from_product(
            [days, ["A", "B", "C"]], names=["date", "code"]))
        table = TargetHitTable(target_pct=0.05, stop_pct=0.05, horizon=5)
        regime_of = {d: "neutral" for d in days}
        table.build(panel, rz, regime_of, [days[4]], entry_style="tail_close")
        est = table.lookup("neutral", 0.0)
        # 3 样本 1 hit；n<min_n → fallback=True 且 wilson_lb=0（不可 BUY）
        assert est.n == 3 and est.hits == 1
        assert est.rate == pytest.approx(1 / 3, abs=0.01)
        assert est.fallback is True and est.wilson_lb == 0.0
        assert est.buy_eligible is False

    def test_parent_fallback_and_buy_eligibility(self):
        # 一个大样本高命中格 + 一个空格查询
        table = TargetHitTable(min_n=30)
        table.cells["risk_off|rev_q5"] = [50, 100]        # 50% 命中 n=100
        table.merge_parent_cells()
        est = table.lookup("risk_off", 2.5)               # 精确格存在
        assert est.n == 100 and est.wilson_lb > BUY_MIN_WILSON_LB
        assert est.buy_eligible is True
        est2 = table.lookup("neutral", 2.5)               # 精确/父格样本不足 → all 父格
        assert est2.cell == "all|rev_all" and est2.fallback is True
        # 父格样本不足 → 再回退 all
        small = TargetHitTable(min_n=30)
        small.cells["risk_off|rev_q5"] = [5, 10]
        small.merge_parent_cells()
        est3 = small.lookup("risk_off", 2.5)
        assert est3.fallback is True and est3.buy_eligible is False

    def test_save_load_roundtrip(self, tmp_path):
        t = TargetHitTable(min_n=30, target_pct=0.05, horizon=5,
                           built_range="2022-2025")
        t.cells["risk_off|rev_q5"] = [50, 100]
        p = t.save(tmp_path / "t.json")
        t2 = TargetHitTable.load(p)
        assert t2.cells == t.cells and t2.horizon == 5
        assert t2.lookup("risk_off", 2.5).n == 100

    def test_load_missing_returns_empty(self, tmp_path):
        t = TargetHitTable.load(tmp_path / "nope.json")
        assert t.cells == {}

    def test_next_open_entry_semantics(self):
        panel = _panel_with_outcome()
        days = sorted(panel.index.get_level_values("date").unique())
        rz = pd.Series(0.0, index=pd.MultiIndex.from_product(
            [days, ["A", "B", "C"]], names=["date", "code"]))
        table = TargetHitTable(target_pct=0.05, stop_pct=0.05, horizon=5)
        # next_open：以决策日次日开盘为 entry —— 票A次日开盘仍 100 → 第7日 +6% 命中
        table.build(panel, rz, {d: "neutral" for d in days}, [days[4]],
                    entry_style="next_open")
        assert table.lookup("neutral", 0.0).n == 3
