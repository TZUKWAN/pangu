"""Strict causal replay tests for the short-term recommendation agent."""

from __future__ import annotations

import pandas as pd

from engine.short_term_replay import (
    ReplayOutcome,
    ShortTermReplayConfig,
    ShortTermReplayEngine,
)


def _plan() -> dict:
    return {
        "entry_price": 10.0,
        "initial_stop": 9.0,
        "first_target": 11.0,
        "final_target": 12.0,
        "trailing_reference": 10.5,
        "max_holding_days": 3,
        "sentiment_exit_drop": 15.0,
        "conservative_same_day_order": "stop_first",
        "rules": [],
    }


def _recommendation(style: str = "ma_pullback") -> dict:
    plan = _plan()
    return {
        "code": "000001",
        "entry_plan": {
            "entry_style": style,
            "trigger_price": 10.0,
            "ideal_entry_zone": [9.9, 10.1],
        },
        "entry_exit": {"exit_plan": plan},
    }


def _context() -> dict:
    return {
        "news_evidence": {"sentiment_label": "bullish", "risk_events": []},
        "market_phase": "主升期",
        "current_temperature": 70,
        "entry_temperature": 70,
        "theme_invalidated": False,
        "news_context_exact": True,
        "market_context_exact": True,
        "theme_status_known": True,
    }


def _kline(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"日期": date, "开盘": open_, "最高": high, "最低": low, "收盘": close, "成交量": 100000}
        for date, open_, high, low, close in rows
    ])


def test_signal_cannot_enter_on_same_day() -> None:
    rows = _kline([
        ("20260701", 10.0, 12.0, 9.0, 11.0),
        ("20260702", 10.5, 10.8, 10.4, 10.7),
    ])
    result = ShortTermReplayEngine().replay(
        _recommendation(), rows, signal_date="20260701",
        daily_context={"20260702": _context()}, causal_signal_evidence=True,
    )
    assert result.status == "no_entry"
    assert result.fills == []


def test_no_trigger_is_no_entry_not_a_win() -> None:
    rows = _kline([
        ("20260701", 9.8, 10.0, 9.7, 9.9),
        ("20260702", 10.3, 10.5, 10.2, 10.4),
    ])
    result = ShortTermReplayEngine().replay(_recommendation(), rows, signal_date="20260701")
    assert result.status == "no_entry"
    assert result.win is None


def test_same_day_stop_wins_over_both_targets() -> None:
    rows = _kline([
        ("20260701", 9.9, 10.1, 9.8, 10.0),
        ("20260702", 10.0, 12.5, 8.8, 11.8),
    ])
    result = ShortTermReplayEngine().replay(
        _recommendation(), rows, signal_date="20260701",
        daily_context={"20260702": _context()}, causal_signal_evidence=True,
    )
    assert result.status == "closed"
    assert result.reason == "hard_stop"
    assert result.net_pnl < 0
    assert result.fills[-1].raw_price == 9.0


def test_two_targets_are_filled_half_then_final_with_costs() -> None:
    rows = _kline([
        ("20260701", 9.9, 10.1, 9.8, 10.0),
        ("20260702", 10.0, 12.2, 9.5, 12.0),
    ])
    result = ShortTermReplayEngine().replay(
        _recommendation(), rows, signal_date="20260701",
        daily_context={"20260702": _context()}, causal_signal_evidence=True,
    )
    assert result.status == "closed"
    assert result.reason == "final_target"
    assert [fill.action for fill in result.fills] == ["buy", "sell", "sell"]
    assert result.fills[1].raw_price == 11.0
    assert result.fills[2].raw_price == 12.0
    assert result.net_return < result.gross_return
    assert result.causal_context_complete is True


def test_time_stop_after_three_trading_days() -> None:
    rows = _kline([
        ("20260701", 9.9, 10.1, 9.8, 10.0),
        ("20260702", 10.0, 10.8, 9.6, 10.6),
        ("20260703", 10.6, 10.8, 10.4, 10.6),
        ("20260706", 10.6, 10.8, 10.4, 10.6),
    ])
    contexts = {date: _context() for date in ("20260702", "20260703", "20260706")}
    result = ShortTermReplayEngine().replay(
        _recommendation(), rows, signal_date="20260701",
        daily_context=contexts, causal_signal_evidence=True,
    )
    assert result.status == "closed"
    assert result.reason == "time_stop"
    assert result.exit_date == "20260706"


def test_gap_above_entry_zone_is_not_chased() -> None:
    rows = _kline([
        ("20260701", 9.9, 10.1, 9.8, 10.0),
        ("20260702", 10.3, 11.0, 10.2, 10.8),
    ])
    result = ShortTermReplayEngine().replay(
        _recommendation("breakout_confirm"), rows, signal_date="20260701"
    )
    assert result.status == "no_entry"


def test_85_percent_claim_fails_on_tiny_sample() -> None:
    cfg = ShortTermReplayConfig(min_trades=100, min_trade_dates=30, require_causal_context=False)
    engine = ShortTermReplayEngine(cfg)
    outcomes = [
        ReplayOutcome("20260701", f"{i:06d}", "closed", "target", net_return=0.02, net_pnl=100)
        for i in range(10)
    ]
    report = engine.acceptance(outcomes)
    assert report["observed_win_rate"] == 1.0
    assert report["verified_success_rate_85"] is False
    assert report["verification_status"] == "insufficient_sample"


def test_85_percent_claim_requires_causal_context() -> None:
    cfg = ShortTermReplayConfig(
        min_trades=2, min_trade_dates=2, min_execution_rate=0,
        target_win_rate=0.5, require_causal_context=True,
    )
    engine = ShortTermReplayEngine(cfg)
    outcomes = [
        ReplayOutcome("20260701", "000001", "closed", "target", net_return=0.02, net_pnl=100),
        ReplayOutcome("20260702", "000002", "closed", "target", net_return=0.03, net_pnl=100),
    ]
    report = engine.acceptance(outcomes)
    assert report["verified_success_rate_85"] is False
    assert report["verification_status"] == "incomplete_causal_context"
    assert any("上下文覆盖率" in blocker for blocker in report["blockers"])
