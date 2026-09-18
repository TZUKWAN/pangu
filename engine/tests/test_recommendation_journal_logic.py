import pandas as pd

from engine.recommendation_journal import RecommendationJournal
from engine.short_term_replay import ShortTermReplayConfig


class FakeKlineLoader:
    def daily_kline(self, code, days=80, date=None):
        base = 10 if code == "000001" else 20
        return pd.DataFrame(
            [
                {"日期": "2026-07-02", "收盘": base * 1.05, "最高": base * 1.08, "最低": base * 1.02},
                {"日期": "2026-07-03", "收盘": base * 1.10, "最高": base * 1.12, "最低": base * 1.04},
                {"日期": "2026-07-06", "收盘": base * 1.15, "最高": base * 1.18, "最低": base * 1.08},
                {"日期": "2026-07-07", "收盘": base * 1.12, "最高": base * 1.16, "最低": base * 1.06},
                {"日期": "2026-07-08", "收盘": base * 1.20, "最高": base * 1.22, "最低": base * 1.09},
                {"日期": "2026-07-09", "收盘": base * 1.18, "最高": base * 1.24, "最低": base * 1.10},
                {"日期": "2026-07-10", "收盘": base * 1.22, "最高": base * 1.25, "最低": base * 1.12},
                {"日期": "2026-07-13", "收盘": base * 1.25, "最高": base * 1.28, "最低": base * 1.14},
                {"日期": "2026-07-14", "收盘": base * 1.28, "最高": base * 1.30, "最低": base * 1.16},
                {"日期": "2026-07-15", "收盘": base * 1.30, "最高": base * 1.32, "最低": base * 1.18},
            ]
        )


def _candidate(code, name, status, close):
    return {
        "code": code,
        "name": name,
        "board": "AI算力",
        "close": close,
        "recommend": {"recommend_score": 88 if status == "xuanwu" else 62, "grade": "S"},
        "entry_exit": {
            "buy_points": [{"price": close, "is_primary": True}],
            "stop_loss": {"price": close * 0.94},
            "take_profit": [{"price": close * 1.12}],
            "risk_reward_ratio": 2.0,
        },
        "debate": {"verdict": "推荐", "confidence": 82} if status == "xuanwu" else {},
        "xuanwu": {"status": status, "blockers": [] if status == "xuanwu" else ["multi_agent_missing"]},
    }


def test_recommendation_journal_records_and_evaluates_forward_returns():
    journal = RecommendationJournal(":memory:", data_loader=FakeKlineLoader())
    result = journal.record_pipeline_result(
        {
            "date": "20260701",
            "candidates": [
                _candidate("000001", "测试一号", "xuanwu", 10),
                _candidate("000002", "测试二号", "watch", 20),
            ],
        }
    )

    assert result == {"run_date": "20260701", "recorded": 2, "recommended": 1}

    evaluation = journal.evaluate(as_of="20260715")
    assert evaluation["evaluated_metrics"] == 8
    assert evaluation["skipped"] == 0

    summary = journal.summary(days=3650)
    assert summary["total"] == 2
    assert summary["recommended"] == 1
    assert summary["horizons"][1]["evaluated"] == 2
    assert summary["horizons"][1]["avg_return"] == 0.05
    assert summary["horizons"][10]["avg_return"] == 0.3
    assert summary["latest"][0]["code"] == "000001"
    assert summary["latest"][0]["return_1d"] == 5.0


def test_recommendation_journal_only_recommended_filters_watch_rows():
    journal = RecommendationJournal(":memory:", data_loader=FakeKlineLoader())
    journal.record_pipeline_result(
        {
            "date": "20260701",
            "candidates": [
                _candidate("000001", "测试一号", "xuanwu", 10),
                _candidate("000002", "测试二号", "watch", 20),
            ],
        }
    )
    journal.evaluate(as_of="20260715", only_recommended=True)

    summary = journal.summary(days=3650, only_recommended=True)
    assert summary["total"] == 1
    assert summary["recommended"] == 1
    assert summary["horizons"][1]["evaluated"] == 1
    assert summary["latest"][0]["code"] == "000001"


def _strict_exit_plan() -> dict:
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


def _formal_candidate() -> dict:
    plan = _strict_exit_plan()
    return {
        "code": "000001",
        "name": "正式推荐",
        "close": 10.0,
        "entry_plan": {
            "entry_style": "ma_pullback",
            "trigger_price": 10.0,
            "ideal_entry_zone": [9.9, 10.1],
        },
        "entry_exit": {
            "buy_points": [{"price": 10.0, "is_primary": True}],
            "stop_loss": {"price": 9.0},
            "take_profit": [{"price": 11.0}, {"price": 12.0}],
            "exit_plan": plan,
        },
        "xuanwu": {"status": "xuanwu", "blockers": []},
        "gate_status": "final",
    }


def test_formal_gate_contract_overrides_legacy_xuanwu_flag() -> None:
    journal = RecommendationJournal(":memory:", data_loader=FakeKlineLoader())
    candidate = _formal_candidate()
    result = journal.record_pipeline_result({
        "date": "20260701",
        "candidates": [candidate],
        "final_recommendations": [],
        "watchlist": [{**candidate, "gate_status": "watch"}],
        "rejected": [],
    })
    assert result["recorded"] == 1
    assert result["recommended"] == 0
    with journal._connect() as conn:
        row = conn.execute(
            "SELECT is_recommended, decision_status FROM recommendation_journal"
        ).fetchone()
    assert row == (0, "watch")


class StrictReplayLoader:
    def daily_kline(self, code, days=120, date=None):
        return pd.DataFrame([
            {"日期": "20260701", "开盘": 9.9, "最高": 10.1, "最低": 9.8, "收盘": 10.0, "成交量": 100000},
            {"日期": "20260702", "开盘": 10.0, "最高": 10.8, "最低": 9.6, "收盘": 10.6, "成交量": 100000},
            {"日期": "20260703", "开盘": 10.6, "最高": 11.2, "最低": 10.5, "收盘": 11.0, "成交量": 100000},
            {"日期": "20260706", "开盘": 11.0, "最高": 12.2, "最低": 10.8, "收盘": 12.0, "成交量": 100000},
        ])


def test_short_term_replay_uses_recorded_formal_plan_and_exact_news_flag() -> None:
    journal = RecommendationJournal(":memory:", data_loader=StrictReplayLoader())
    candidate = _formal_candidate()
    recorded = journal.record_pipeline_result({
        "date": "20260701",
        "candidates": [candidate],
        "final_recommendations": [candidate],
        "watchlist": [],
        "rejected": [],
        "data_quality": "ok",
        "historical_mode": "live",
        "news": {
            "date": "20260701",
            "source_state": {
                "archive": {"status": "ok", "mode": "persisted_exact_date"},
            },
        },
    })
    assert recorded["recommended"] == 1
    with journal._connect() as conn:
        stored = conn.execute(
            "SELECT exit_plan_json, causal_news_available FROM recommendation_journal"
        ).fetchone()
    assert '"max_holding_days": 3' in stored[0]
    assert stored[1] == 1

    def contexts(run_date, code, kline, evidence):
        return {
            date: {
                "news_evidence": {"sentiment_label": "bullish", "risk_events": []},
                "market_phase": "主升期",
                "current_temperature": 70,
                "entry_temperature": 70,
                "theme_invalidated": False,
                "news_context_exact": True,
                "market_context_exact": True,
                "theme_status_known": True,
            }
            for date in ("20260702", "20260703", "20260706")
        }

    result = journal.evaluate_short_term(
        as_of="20260710",
        cfg=ShortTermReplayConfig(
            min_trades=1,
            min_trade_dates=1,
            min_execution_rate=0,
            target_win_rate=0,
            min_profit_factor=0,
            max_drawdown_limit=1,
            require_causal_context=True,
        ),
        context_provider=contexts,
    )
    assert result["evaluated_records"] == 1
    assert result["outcomes"][0]["status"] == "closed"
    assert result["outcomes"][0]["reason"] == "final_target"
    assert result["outcomes"][0]["causal_context_complete"] is True
    with journal._connect() as conn:
        replay_row = conn.execute(
            "SELECT status, net_return FROM short_term_replay_metrics"
        ).fetchone()
    assert replay_row[0] == "closed"
    assert replay_row[1] > 0
    summary = journal.short_term_summary(
        days=3650,
        cfg=ShortTermReplayConfig(
            min_trades=1, min_trade_dates=1, min_execution_rate=0,
            target_win_rate=0, min_profit_factor=0,
            max_drawdown_limit=1, require_causal_context=True,
        ),
    )
    assert summary["latest"][0]["name"] == "正式推荐"
    assert summary["acceptance"]["executed_count"] == 1
