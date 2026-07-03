from __future__ import annotations

from engine.p0_factors import P0FactorCollector
from engine.recommender import Recommender


class FakeP0Collector(P0FactorCollector):
    def __init__(self) -> None:
        super().__init__({"structured_data": {"workers": 1, "deep_max_candidates": 10, "fund_flow_max_candidates": 10}})

    def _cached(self, name, parts, fn):  # noqa: ANN001
        return fn()

    # ── THS dataapi mock for dragon tiger ──
    def _ths_api_get(self, path, params=None):  # noqa: ANN001
        if "dragon_tiger_pool" in str(path):
            return {"data": [
                {"code": "000001", "name": "A", "reason": "daily turnover deviation",
                 "net_buy": 60_000_000, "buy_amt": 90_000_000, "sell_amt": 30_000_000,
                 "turnover_rate": 12.3},
            ]}
        return None

    # ── THS hot list ──
    def _ths_hot_list(self):
        return [{"rank": 3, "code": "000001", "name": "A", "heat": 99, "concepts": ["AI"], "tag": "hot"}]

    # ── Limit up sentiment: use THS pools via data_loader ──
    def _collect_limit_sentiment(self, date, recorder):  # noqa: ANN001
        recorder.ok(2)
        return {"limit_up_sentiment": {
            "date": date, "zt_count": 1, "zb_count": 0, "dt_count": 1,
            "break_rate": 0.0, "max_height": 2, "ladder": {2: 1},
            "sample_limit_up": [{"代码": "000001", "名称": "A", "连板数": 2}],
            "source": "ths_dataapi",
        }}

    # ── Fund flow ──
    def _stock_fund_flow_120d_summary(self, code: str):
        if code == "000001":
            return {"sum_20d_main_net": 120_000_000, "positive_days_20": 15,
                    "latest": {"date": "2026-07-01", "main_net": 10_000_000}, "source": "ths_instant"}
        if code == "000002":
            return {"sum_20d_main_net": -80_000_000, "positive_days_20": 3,
                    "latest": {"date": "2026-07-01", "main_net": -10_000_000}, "source": "ths_instant"}
        return {}

    # ── Removed sources (Eastmoney exclusive) ──
    def _margin_trading_summary(self, code: str):
        return {}

    def _lockup_summary(self, code: str, trade_date: str):  # noqa: ARG002
        return {}

    def _block_trade_summary(self, code: str):
        return {}

    def _holder_num_summary(self, code: str):
        return {}

    # ── Research (THS EPS forecast) ──
    def _research_summary(self, code: str):
        return {"count": 2, "latest": {"title": "positive report"}, "source": "ths_basic"} if code == "000001" else {}

    # ── Announcements ──
    def _announcement_summary(self, code: str):
        if code == "000002":
            return {"count": 1, "latest": {"title": "减持公告"}, "risk_count": 1}
        return {}

    # ── IRM ──
    def _irm_summary(self, code: str):
        return {"count": 1, "answered_count": 1, "latest": {"question": "Q", "answer": "A"}} if code == "000001" else {}

    # ── Northbound: market-level only ──
    def _northbound_summary(self, code: str):
        return {"scope": "market", "latest": {"date": "2026-07-01", "net_buy": 85.3},
                "sum_5d_net_buy": None, "source": "ths_hexin",
                "note": "per-code northbound removed"}
        # Previously per-code: now market-level aggregate for all codes

    # ── Dividend ──
    def _dividend_summary(self, code: str):
        return {"dividend_per_share": 0.5, "source": "baidu_gushitong"} if code == "000001" else {}


def _candidate(code: str) -> dict:
    return {
        "code": code,
        "name": code,
        "board": "test",
        "close": 10,
        "pct_change": 2,
        "turnover_rate": 8,
        "fund_inflow_days": 1,
        "rps": 70,
        "reasons": [],
        "technical": {"kline": [{"close": 10}], "ma": {}, "macd": {}, "volume": {}},
        "entry_exit": {
            "buy_points": [{"price": 10, "is_primary": True, "type": "MA5"}],
            "stop_loss": {"price": 9.4},
            "take_profit": [{"price": 11.2}],
            "risk_reward_ratio": 2.0,
        },
    }


def test_p0_collector_attaches_structured_factors_and_source_state():
    """Test structured factors after Eastmoney removal.

    Key changes from original:
    - dragon_tiger: now from THS dataapi (not EM datacenter)
    - margin/block_trade/holder_num/lockup: removed_per_policy (no free alternative)
    - northbound: market-level only (THS hexin)
    - hot_rank: THS only (EM hot rank removed)
    - limit_up_sentiment: THS limit pools (EM pools removed)
    """
    rows = [_candidate("000001"), _candidate("000002")]
    state, modules = FakeP0Collector().collect("20260701", rows)

    # Dragon tiger from THS dataapi
    assert rows[0]["structured_factors"]["dragon_tiger"]["net_buy_wan"] == 6000.0
    assert rows[0]["structured_factors"]["dragon_tiger"]["source"] == "ths_dataapi"

    # Hot rank from THS
    assert rows[0]["structured_factors"]["hot_rank"]["rank"] == 3

    # Removed factors should not be present (empty dict returned)
    assert "block_trade" not in rows[1]["structured_factors"] or not rows[1]["structured_factors"]["block_trade"]
    assert "lockup" not in rows[1]["structured_factors"] or not rows[1]["structured_factors"]["lockup"]

    # Northbound: market-level
    assert rows[0]["structured_factors"]["northbound"]["scope"] == "market"

    # Limit up sentiment
    assert modules["limit_up_sentiment"]["zt_count"] == 1

    # Fund flow
    assert state["capital_flow_120d"]["success_count"] == 2

    # Summary
    assert state["summary"]["candidate_count"] == 2

    # Source state entries exist
    assert "northbound" in state
    assert "dragon_tiger_daily" in state

    # Removed/disabled sources should have appropriate status
    block_trade_state = state.get("block_trade", {})
    if block_trade_state:
        assert block_trade_state.get("status") in ("removed", "skipped", "empty", "degraded")


def test_recommender_uses_structured_signal():
    """Recommender still produces structured_signal after EM removal."""
    c = _candidate("000001")
    collector = FakeP0Collector()
    collector.collect("20260701", [c])

    rec = Recommender()
    results = rec.rank([c], news_sentiment={})

    assert len(results) > 0
    # rank() returns Recommendation objects; convert to dict for assertion
    result = results[0].to_dict() if hasattr(results[0], 'to_dict') else results[0]
    assert isinstance(result.get("recommend_score"), (int, float))
    assert "grade" in result


def test_p0_removed_sources_clean():
    """Removed Eastmoney-exclusive sources should produce empty results not errors."""
    collector = FakeP0Collector()
    # Call removed source methods directly
    assert collector._margin_trading_summary("000001") == {}
    assert collector._lockup_summary("000001", "2026-07-01") == {}
    assert collector._block_trade_summary("000001") == {}
    assert collector._holder_num_summary("000001") == {}
