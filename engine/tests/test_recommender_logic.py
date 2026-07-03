"""Recommender 真实特征分化测试：确保 score/target_pct/rrr 不再全同。"""

from __future__ import annotations

import statistics

import pytest

from engine.recommender import Recommender


def _candidate(code: str, rps: float, pct_change: float, ma_bull: bool, macd: bool, turnover: float, fund: float):
    return {
        "code": code,
        "name": f"票{code}",
        "board": "AI算力",
        "close": 10.0,
        "pct_change": pct_change,
        "turnover_rate": turnover,
        "fund_inflow_days": fund,
        "rps": rps,
        "reasons": ["RPS靠前"],
        "technical": {
            "kline": [
                {"open": 9.5, "high": 10.2, "low": 9.4, "close": 10.0, "volume": 10000},
                {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "volume": 12000},
                {"open": 10.2, "high": 10.8, "low": 10.0, "close": 10.5, "volume": 15000},
            ],
            "ma": {"ma5": 10.3 if ma_bull else 9.8, "ma10": 10.1 if ma_bull else 10.2, "ma30": 9.9 if ma_bull else 10.3},
            "macd": {"golden_cross": macd},
            "volume": {"volume_ratio": turnover / 5.0},
        },
        "entry_exit": {
            "buy_points": [{"price": 10.0, "type": "MA5", "is_primary": True}],
            "stop_loss": {"price": 9.5},
            "take_profit": [{"price": 11.0}],
            "risk_reward_ratio": 2.0,
        },
    }


def test_target_pct_and_score_differentiated():
    rec = Recommender()
    candidates = [
        _candidate("000001", rps=90, pct_change=5.0, ma_bull=True, macd=True, turnover=12.0, fund=5.0),
        _candidate("000002", rps=60, pct_change=-1.0, ma_bull=False, macd=False, turnover=4.0, fund=0.0),
        _candidate("000003", rps=75, pct_change=2.0, ma_bull=True, macd=False, turnover=8.0, fund=2.0),
        _candidate("000004", rps=55, pct_change=0.5, ma_bull=False, macd=True, turnover=6.0, fund=1.0),
    ]
    results = rec.rank(candidates)
    scores = [r.recommend_score for r in results]
    tps = [r.target_pct for r in results]
    print("scores:", scores)
    print("target_pcts:", tps)
    assert len(set(round(s, 1) for s in scores)) >= 3
    assert len(set((round(tp[0], 1), round(tp[1], 1)) for tp in tps)) >= 3
    assert statistics.pstdev(scores) > 1.0
