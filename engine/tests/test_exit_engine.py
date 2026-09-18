"""短期卖出计划与退出决策测试。"""

from __future__ import annotations

import pandas as pd

from engine.entry_exit import EntryExitEngine
from engine.exit_engine import ShortTermExitEngine
from engine.trend_scanner import StockCandidate


def _plan():
    return ShortTermExitEngine({"horizon_days": 3}).build_plan(
        entry_price=10.0,
        stop_price=9.0,
        take_profit_prices=[11.0, 12.0],
        trailing_reference=10.6,
    )


def test_hard_stop_has_priority_over_same_day_target() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {"open": 10.0, "high": 12.5, "low": 8.8, "close": 11.8}
    )
    assert decision.rule_type == "hard_stop"
    assert decision.action == "exit_all"
    assert decision.execution_price == 9.0


def test_gap_down_executes_at_open_not_stop_price() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {"open": 8.6, "high": 9.1, "low": 8.5, "close": 8.9}
    )
    assert decision.rule_type == "hard_stop"
    assert decision.execution_price == 8.6


def test_news_risk_exits_position() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {"close": 10.2, "news_evidence": {"risk_events": ["立案调查"]}}
    )
    assert decision.rule_type == "news_invalidation"
    assert decision.action == "exit_all"


def test_market_retreat_exits_position() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {"close": 10.2, "market_phase": "退潮期"}
    )
    assert decision.rule_type == "market_retreat"


def test_missing_current_temperature_does_not_false_trigger_retreat() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {"close": 10.8, "entry_temperature": 70}
    )
    assert decision.rule_type == "none"
    assert decision.action == "hold"


def test_theme_invalidation_exits_position() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {"close": 10.2, "theme_invalidated": True}
    )
    assert decision.rule_type == "theme_invalidation"


def test_trend_break_exits_position() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {"close": 9.8, "ma10": 9.9, "ma20": 9.5, "volume_ratio": 1.3}
    )
    assert decision.rule_type == "trend_break"


def test_first_target_reduces_half_and_raises_stop() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {"open": 10.2, "high": 11.2, "low": 10.1, "close": 10.9}
    )
    assert decision.rule_type == "first_target"
    assert decision.action == "reduce_half"
    assert decision.execution_price == 11.0
    assert decision.new_stop == 10.0


def test_final_target_exits_all() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {
            "open": 11.0, "high": 12.2, "low": 10.8, "close": 12.0,
            "first_target_taken": True, "active_stop": 10.0,
        }
    )
    assert decision.rule_type == "final_target"
    assert decision.action == "exit_all"
    assert decision.execution_price == 12.0


def test_time_stop_exits_on_third_trading_day() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan(), {"close": 10.8, "days_held": 3}
    )
    assert decision.rule_type == "time_stop"
    assert decision.action == "exit_all"


def test_serialized_plan_keeps_trailing_stop_reference() -> None:
    decision = ShortTermExitEngine().evaluate(
        _plan().to_dict(), {"close": 10.5}
    )
    assert decision.rule_type == "trailing_stop"
    assert decision.action == "exit_all"


class _Loader:
    def daily_kline(self, code: str, days: int = 60, adjust: str = "qfq", date: str | None = None):
        rows = []
        for i in range(80):
            close = 10 + i * 0.05
            rows.append({
                "日期": f"2024{i + 1:04d}", "开盘": close - 0.02, "收盘": close,
                "最高": close + 0.1, "最低": close - 0.1, "成交量": 100000 + i * 100,
            })
        return pd.DataFrame(rows).tail(days).reset_index(drop=True)


def test_entry_exit_output_contains_complete_exit_plan() -> None:
    candidate = StockCandidate(
        code="000001", name="测试股", board="测试", close=13.95,
        pct_change=1.0, turnover_rate=2.0, circ_mv_yi=100.0, rps=90.0,
    )
    data = EntryExitEngine(_Loader(), {"horizon_days": 3}).compute(candidate).to_dict()
    plan = data["exit_plan"]
    assert plan["entry_price"] > plan["initial_stop"] > 0
    assert plan["first_target"] > plan["entry_price"]
    assert plan["final_target"] >= plan["first_target"]
    assert plan["max_holding_days"] == 3
    assert {rule["rule_type"] for rule in plan["rules"]} >= {
        "hard_stop", "news_invalidation", "market_retreat", "theme_invalidation",
        "trend_break", "first_target", "final_target", "trailing_stop", "time_stop",
    }
