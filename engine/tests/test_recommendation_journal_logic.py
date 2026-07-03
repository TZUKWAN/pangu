import pandas as pd

from engine.recommendation_journal import RecommendationJournal


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
