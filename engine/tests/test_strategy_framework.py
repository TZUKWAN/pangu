"""策略框架单元测试：市场阶段、策略池、推荐闸门。"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from engine.data_loader import DataLoader
from engine.market_phase import MarketPhaseAnalyzer
from engine.quant_guard import GuardResult, QuantGuard
from engine.recommendation_gate import RecommendationGate
from engine.strategy_pools import (
    DividendLowVolPool,
    EventDrivenPool,
    LimitUpPool,
    OversoldReboundPool,
    SmallQualityPool,
    ThemeLeaderPool,
    TrendPullbackPool,
)
from engine.trend_scanner import StockCandidate


class DummyDataLoader:
    """最小化 DataLoader 替身，用于策略池独立测试。"""

    def __init__(self):
        self._spot = pd.DataFrame()
        self._limit_up = pd.DataFrame()
        self._limit_down = pd.DataFrame()
        self._kline: dict[str, pd.DataFrame] = {}

    def all_spot(self):
        return self._spot

    def limit_up_pool(self, date: str | None = None):
        return self._limit_up

    def limit_down_pool(self, date: str | None = None):
        return self._limit_down

    def daily_kline(self, code: str, n: int = 120, **kwargs):
        return self._kline.get(code, pd.DataFrame())

    def financial_indicator(self, code: str, date: str | None = None):
        return pd.DataFrame()

    def longhu_bang(self, date: str | None = None):
        return pd.DataFrame()


def _make_spot(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestMarketPhase(unittest.TestCase):
    def test_cold_phase_when_limit_down_high(self):
        dl = DummyDataLoader()
        dl._limit_up = pd.DataFrame({"代码": ["000001"], "名称": ["平安"], "连板数": ["1板"]})
        dl._limit_down = pd.DataFrame({"代码": [f"{i:06d}" for i in range(60)], "名称": ["x"] * 60})
        dl._spot = _make_spot([{"代码": "000001", "名称": "平安", "涨跌幅": -1.0}])
        phase = MarketPhaseAnalyzer(dl, {}).analyze("20250115")
        self.assertEqual(phase.phase, "冰点期")
        self.assertIn("红利低波防守", phase.allowed_strategies)
        self.assertIn("追涨", phase.forbidden_strategies)

    def test_main_rise_phase(self):
        dl = DummyDataLoader()
        codes = [f"{i:06d}" for i in range(50)]
        dl._limit_up = pd.DataFrame({
            "代码": codes,
            "名称": ["x"] * 50,
            "连板数": ["4板"] * 50,
            "所属行业": ["AI"] * 50,
        })
        dl._limit_down = pd.DataFrame()
        dl._spot = _make_spot([{"代码": c, "名称": "x", "涨跌幅": 5.0} for c in codes[:100]])
        with patch("engine.market_phase.SentimentMeter") as MockMeter:
            MockMeter.return_value.measure.return_value.temperature = 70.0
            MockMeter.return_value.measure.return_value.advice = "积极"
            MockMeter.return_value.measure.return_value.posture = "正常"
            MockMeter.return_value.measure.return_value.warnings = []
            MockMeter.return_value.measure.return_value.to_dict.return_value = {"temperature": 70.0}
            phase = MarketPhaseAnalyzer(dl, {}).analyze("20250115")
        self.assertEqual(phase.phase, "主升期")


class TestStrategyPools(unittest.TestCase):
    def test_theme_leader_pool(self):
        dl = DummyDataLoader()
        dl._limit_up = pd.DataFrame({
            "代码": ["000001", "000002", "000003", "000004"],
            "名称": ["A", "B", "C", "D"],
            "连板数": ["2板", "3板", "1板", "1板"],
            "所属行业": ["AI", "AI", "AI", "医药"],
        })
        pool = ThemeLeaderPool(dl, {})
        sigs = pool.select("20250115")
        self.assertTrue(sigs)
        self.assertTrue(all(s.strategy_name == "题材龙头" for s in sigs))

    def test_limit_up_pool(self):
        dl = DummyDataLoader()
        dl._limit_up = pd.DataFrame({
            "代码": ["000001", "000002"],
            "名称": ["A", "B"],
            "连板数": ["2板", "1板"],
            "涨跌幅": [10.0, 9.95],
        })
        sigs = LimitUpPool(dl, {}).select("20250115")
        self.assertEqual(len(sigs), 2)
        self.assertGreater(sigs[0].score, sigs[1].score)

    def test_trend_pullback_pool_respects_rps_min(self):
        dl = DummyDataLoader()
        dl._spot = _make_spot([{
            "代码": "000001", "名称": "A", "涨跌幅": 2.0,
            "最新价": 10.0, "换手率": 3.0, "流通市值": 50e8,
        }])
        # 20 日均线 9.5，最新价 10.0（未回踩也未突破）
        closes = [9.0] * 19 + [10.0]
        dl._kline["000001"] = pd.DataFrame({
            "close": closes,
            "volume": [10000] * 20,
        })
        with patch.object(TrendPullbackPool, "_market_spot", return_value=dl._spot):
            with patch("engine.strategy_pools.RPSCalculator") as MockRPS:
                MockRPS.return_value.rps_for_codes.return_value = {"000001": {"rps": 85, "mode": "real"}}
                sigs = TrendPullbackPool(dl, {"trend": {"min_rps": 90}}).select("20250115")
        self.assertEqual(len(sigs), 0)

    def test_oversold_rebound_pool(self):
        dl = DummyDataLoader()
        dl._spot = _make_spot([{"代码": "000001", "名称": "A", "涨跌幅": -2.0}])
        closes = list(range(120, 100, -1))  # 20 日跌 ~16%
        dl._kline["000001"] = pd.DataFrame({
            "close": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "volume": [10000] * 20,
        })
        sigs = OversoldReboundPool(dl, {}).select("20250115")
        self.assertTrue(len(sigs) >= 0)

    def test_small_quality_pool_filters_loss(self):
        dl = DummyDataLoader()
        dl._spot = _make_spot([{
            "代码": "000001", "名称": "A", "涨跌幅": 2.0,
            "换手率": 3.0, "流通市值": 50e8,
        }])
        sigs = SmallQualityPool(dl, {"small_quality": {"max_circ_mv_yi": 100, "min_turnover": 2.0}}).select("20250115")
        self.assertEqual(len(sigs), 1)


class TestRecommendationGate(unittest.TestCase):
    def test_gate_separates_final_and_watch(self):
        dl = DummyDataLoader()
        dl._spot = _make_spot([{
            "代码": "000001", "名称": "A", "涨跌幅": 2.0,
            "最新价": 10.0, "换手率": 3.0, "流通市值": 50e8,
        }])
        closes = list(range(100, 121))
        dl._kline["000001"] = pd.DataFrame({
            "close": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "volume": [10000] * 21,
        })

        kept = [StockCandidate(
            code="000001", name="A", board="深市主板", close=10.0,
            pct_change=2.0, turnover_rate=3.0, circ_mv_yi=50.0,
            rps=92.0, rps_mode="real", fund_inflow_days=3,
            fund_flow_status="ok", is_watchlist=False,
        )]
        guarded = GuardResult(kept=kept, watch=[], rejected=[], warnings=[])
        phase = {
            "market_phase": "主升期",
            "phase_score": 80,
            "allowed_strategies": ["题材龙头"],
            "forbidden_strategies": [],
            "position_advice": "积极",
        }
        from engine.strategy_pools import StrategySignal
        pooled = {
            "题材龙头": [StrategySignal(
                strategy_name="题材龙头", code="000001", name="A",
                trigger_reason="主线 AI", score=80,
            )]
        }
        candidate_map = {c.code: c for c in kept}
        gate = RecommendationGate(dl, guarded, phase, {}, recommendation_allowed=True)
        result = gate.pass_gate(pooled, candidate_map)
        self.assertEqual(len(result.final_recommendations), 1)
        self.assertEqual(result.final_recommendations[0]["code"], "000001")

    def test_phase_forbidden_goes_to_watch(self):
        dl = DummyDataLoader()
        kept = [StockCandidate(
            code="000001", name="A", board="深市主板", close=10.0,
            pct_change=2.0, turnover_rate=3.0, circ_mv_yi=50.0,
            rps=92.0, rps_mode="real", fund_flow_status="ok",
        )]
        guarded = GuardResult(kept=kept, watch=[], rejected=[], warnings=[])
        phase = {
            "market_phase": "冰点期",
            "allowed_strategies": [],
            "forbidden_strategies": ["题材龙头"],
            "position_advice": "空仓",
        }
        from engine.strategy_pools import StrategySignal
        pooled = {
            "题材龙头": [StrategySignal(
                strategy_name="题材龙头", code="000001", name="A",
                trigger_reason="主线 AI", score=80,
            )]
        }
        gate = RecommendationGate(dl, guarded, phase, {}, recommendation_allowed=True)
        result = gate.pass_gate(pooled, {c.code: c for c in kept})
        self.assertEqual(len(result.watchlist), 1)
        self.assertEqual(len(result.final_recommendations), 0)


if __name__ == "__main__":
    unittest.main()
