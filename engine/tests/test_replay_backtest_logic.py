"""回放回测评估器测试：用合成 K 线验证 win / loss / no_entry 判定。"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.short_term_replay import ShortTermReplayEngine


def _plan(entry: float = 10.0, stop: float = 9.4, first: float = 11.0,
          final: float = 12.0, max_days: int = 3) -> dict:
    rule_types = [
        "hard_stop", "news_invalidation", "market_retreat", "theme_invalidation",
        "trend_break", "first_target", "final_target", "trailing_stop", "time_stop",
    ]
    return {
        "entry_price": entry,
        "initial_stop": stop,
        "first_target": first,
        "final_target": final,
        "trailing_reference": entry * 1.01,
        "max_holding_days": max_days,
        "sentiment_exit_drop": 15.0,
        "conservative_same_day_order": "stop_first",
        "rules": [{"rule_type": n, "action": "exit_all", "condition": n} for n in rule_types],
    }


def _entry_plan(trigger: float, style: str = "ma_pullback", zone=None) -> dict:
    zone = zone or [trigger * 0.99, trigger * 1.01]
    return {
        "entry_style": style,
        "trigger_price": trigger,
        "trigger_condition": "测试",
        "ideal_entry_zone": zone,
        "current_price": trigger * 1.05,
        "is_chasing": False,
        "invalid_condition": "测试失效",
    }


def _kline(closes: list[float], opens: list[float] | None = None,
           highs: list[float] | None = None, lows: list[float] | None = None,
           start: str = "20260601") -> pd.DataFrame:
    n = len(closes)
    opens = opens or closes
    highs = highs or [max(o, c) for o, c in zip(opens, closes)]
    lows = lows or [min(o, c) for o, c in zip(opens, closes)]
    dates = pd.bdate_range(start, periods=n).strftime("%Y%m%d")
    return pd.DataFrame({
        "日期": dates, "开盘": opens, "收盘": closes, "最高": highs, "最低": lows,
        "成交量": [1000] * n, "成交额": [10000.0] * n,
    })


def _engine() -> ShortTermReplayEngine:
    return ShortTermReplayEngine({"short_term_replay": {"min_trades": 10}})


def test_replay_win_on_pullback_entry_then_target():
    """次日开盘 9.95 触发回踩买点（区间 9.90-10.10），第 2 日触及第一目标减半，
    第 3 日时间止损退出剩余 → win。"""
    k = _kline(
        closes=[10.5, 10.1, 11.2, 11.0],
        opens=[10.4, 9.95, 10.2, 10.9],
        highs=[10.6, 10.3, 11.25, 11.05],
        lows=[10.3, 9.9, 10.1, 10.85],
    )
    rec = {"entry_exit": {"entry_plan": _entry_plan(10.0), "exit_plan": _plan()}}
    out = _engine().replay(rec, k, signal_date="20260601")
    assert out.status == "closed"
    assert out.win is True
    assert out.entry_date == "20260602"
    assert out.first_target_taken is True


def test_replay_loss_on_stop():
    """次日入场后跌破止损 → loss。"""
    k = _kline(
        closes=[10.5, 9.8, 9.2, 9.3],
        opens=[10.4, 10.0, 9.2, 9.3],
        highs=[10.6, 10.1, 9.4, 9.5],
        lows=[10.3, 9.6, 9.1, 9.2],
    )
    rec = {"entry_exit": {"entry_plan": _entry_plan(10.0), "exit_plan": _plan()}}
    out = _engine().replay(rec, k, signal_date="20260601")
    assert out.status == "closed"
    assert out.win is False


def test_replay_no_entry_when_open_gaps_above_zone():
    """次日大幅高开、区间未触及 → no_entry（不追价）。"""
    k = _kline(
        closes=[10.5, 11.5, 11.6, 11.7],
        opens=[10.4, 11.4, 11.5, 11.6],
        highs=[10.6, 11.6, 11.7, 11.8],
        lows=[10.3, 11.3, 11.4, 11.5],
    )
    rec = {"entry_exit": {"entry_plan": _entry_plan(10.0, zone=[9.9, 10.1]), "exit_plan": _plan()}}
    out = _engine().replay(rec, k, signal_date="20260601")
    assert out.status == "no_entry"
    assert out.win is None


def test_replay_breakout_confirm_fills_on_strength():
    """突破确认：次日盘中上穿触发价 → 以触发价成交。"""
    k = _kline(
        closes=[10.5, 10.8, 11.3, 11.1],
        opens=[10.4, 10.55, 10.9, 11.0],
        highs=[10.6, 11.05, 11.4, 11.2],
        lows=[10.3, 10.5, 10.85, 10.95],
    )
    rec = {
        "entry_exit": {
            "entry_plan": _entry_plan(10.6, style="breakout_confirm", zone=[10.49, 10.71]),
            "exit_plan": _plan(),
        }
    }
    out = _engine().replay(rec, k, signal_date="20260601")
    assert out.status == "closed"
    assert out.entry_date == "20260602"
    # 突破确认成交价 = 触发价
    buy = out.fills[0]
    assert abs(buy.raw_price - 10.6) < 1e-6


def test_invalid_plan_rejected():
    """缺少规则/止损无效的计划直接 invalid，不算交易。"""
    k = _kline(closes=[10.5, 10.6, 10.7, 10.8])
    rec = {"entry_exit": {"entry_plan": _entry_plan(10.0), "exit_plan": {"initial_stop": 0}}}
    out = _engine().replay(rec, k, signal_date="20260601")
    assert out.status == "invalid"
