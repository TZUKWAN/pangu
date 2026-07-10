"""数据质量判定单元测试。"""

from __future__ import annotations

from engine.pipeline import Pipeline


def _candidates(close=10.0, pct_change=1.0, turnover_rate=5.0, volume=1000.0, amount=10000.0):
    return [{
        "code": f"00000{i}",
        "close": close,
        "pct_change": pct_change,
        "turnover_rate": turnover_rate,
        "volume": volume,
        "amount": amount,
    } for i in range(10)]


def _source_status(all_spot="ok", daily_kline="ok"):
    return {
        "all_spot": {"status": all_spot},
        "daily_kline": {"status": daily_kline},
        "rps": {"status": "ok"},
        "fund_flow": {"status": "ok"},
        "entry_exit": {"status": "ok"},
        "quant_guard": {"status": "ok"},
        "volume_audit": {"status": "ok"},
    }


def _sentiment(limit_up_count=10):
    return {"components": {"limit_up_count": limit_up_count}}


def test_close_missing_rate_failed():
    pipe = Pipeline()
    cands = _candidates(close=None)
    quality, reasons = pipe._compute_data_quality(
        _source_status(), _sentiment(), True, cands
    )
    assert quality == "failed"
    assert any("close" in r for r in reasons)


def test_pct_change_missing_rate_failed():
    pipe = Pipeline()
    cands = _candidates(pct_change=None)
    quality, reasons = pipe._compute_data_quality(
        _source_status(), _sentiment(), True, cands
    )
    assert quality == "failed"
    assert any("pct_change" in r for r in reasons)


def test_turnover_rate_missing_rate_degraded():
    """换手率缺失不应判 failed，只判 degraded。"""
    pipe = Pipeline()
    cands = _candidates(turnover_rate=None)
    quality, reasons = pipe._compute_data_quality(
        _source_status(), _sentiment(), True, cands
    )
    assert quality == "degraded"
    assert any("换手" in r or "turnover" in r for r in reasons)


def test_volume_amount_absent_from_candidate_but_audit_ok_is_not_degraded():
    """量能由 VolumeAudit 统一判定，候选顶层不再重复要求 volume/amount。"""
    pipe = Pipeline()
    cands = _candidates(volume=None, amount=None)
    quality, reasons = pipe._compute_data_quality(
        _source_status(), _sentiment(), True, cands
    )
    assert quality == "ok"
    assert reasons == []


def test_volume_audit_degraded_marks_report_degraded():
    status = _source_status()
    status["volume_audit"] = {"status": "degraded", "reason": "部分量能缺失"}
    quality, reasons = Pipeline()._compute_data_quality(
        status, _sentiment(), True, _candidates(volume=None, amount=None)
    )
    assert quality == "degraded"
    assert any("量能" in r for r in reasons)


def test_all_fields_ok():
    pipe = Pipeline()
    quality, reasons = pipe._compute_data_quality(
        _source_status(), _sentiment(), True, _candidates()
    )
    assert quality == "ok"
    assert reasons == []


def test_quant_guard_rejections_are_not_data_degradation():
    status = _source_status()
    status["quant_guard"] = {"status": "ok", "rejected_count": 353, "kept_count": 0}
    quality, reasons = Pipeline()._compute_data_quality(
        status, _sentiment(), True, []
    )
    assert quality == "ok"
    assert reasons == []


def test_fund_flow_unavailable_is_not_global_data_degradation():
    status = _source_status()
    status["fund_flow"] = {"status": "failed", "reason": "fund_flow_unavailable"}
    quality, reasons = Pipeline()._compute_data_quality(
        status, _sentiment(), True, _candidates()
    )
    assert quality == "ok"
    assert reasons == []


def test_daily_kline_minority_degradation_does_not_degrade_whole_report():
    quality_map = {
        **{
            f"daily_kline:{i:06d}": {
                "status": "ok", "ok": True, "source": "sina", "row_count": 60,
            }
            for i in range(9)
        },
        "daily_kline:999999": {
            "status": "degraded", "ok": True, "source": "fallback", "row_count": 60,
        },
    }

    class Loader:
        def get_source_quality(self):
            return quality_map

    pipe = Pipeline.__new__(Pipeline)
    pipe.dl = Loader()
    status = {}
    pipe._merge_loader_source_quality(status)
    assert status["daily_kline"]["status"] == "ok"
    assert status["daily_kline"]["status_counts"] == {"ok": 9, "degraded": 1, "failed": 0}


def test_decision_buckets_are_deduplicated_and_mutually_exclusive():
    final, watch, rejected = Pipeline._exclusive_decision_buckets(
        [{"code": "000001"}, {"code": "000002"}, {"code": "000002"}],
        [{"code": "000002"}, {"code": "000003"}, {"code": "000003"}],
        [{"code": "000003"}, {"code": "000004"}, {"code": "000004"}],
    )

    assert [item["code"] for item in final] == ["000001"]
    assert [item["code"] for item in watch] == ["000002"]
    assert [item["code"] for item in rejected] == ["000003", "000004"]


def test_rejected_precedence_prevents_blocked_stock_from_returning_to_final():
    final, watch, rejected = Pipeline._exclusive_decision_buckets(
        [{"code": "000001"}],
        [{"code": "000001"}],
        [{"code": "000001", "reject_reason": "重大风险新闻"}],
    )

    assert final == []
    assert watch == []
    assert [item["code"] for item in rejected] == ["000001"]
