from __future__ import annotations

import pandas as pd

from engine.volume_audit import VolumeAudit


def _rows(closes, volumes):
    return [
        {"日期": f"202607{i+1:02d}", "收盘": c, "成交量": v, "成交额": v * c}
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


def test_volume_missing_cannot_final():
    audit = VolumeAudit().audit({"code": "000001", "technical": {"kline": []}, "turnover_missing": True})

    assert audit["status"] == "missing"
    assert audit["price_volume_pattern"] == "missing"
    assert audit["turnover_status"] == "missing"


def test_breakout_without_volume_is_watch():
    closes = [10] * 20 + [10.2, 10.4]
    volumes = [1000] * 21 + [900]
    audit = VolumeAudit().audit(
        {"code": "000001", "technical": {"kline": _rows(closes, volumes)}, "turnover_rate": 2.0}
    )

    assert audit["status"] == "watch"
    assert audit["price_volume_pattern"] == "breakout_without_volume"


def test_pullback_shrink_can_support_ok():
    closes = [10, 10.2, 10.5, 10.8, 11, 10.8, 10.6, 10.4, 10.3, 10.2, 10.15, 10.1]
    volumes = [1000, 1100, 1200, 1300, 1400, 900, 850, 800, 760, 720, 700, 650]
    audit = VolumeAudit().audit(
        {"code": "000001", "technical": {"kline": _rows(closes, volumes)}, "turnover_rate": 1.5}
    )

    assert audit["status"] == "ok"
    assert audit["price_volume_pattern"] == "pullback_shrink"


def test_distribution_volume_is_watch():
    closes = [10] * 18 + [11.0, 11.2, 11.25, 11.26]
    volumes = [1000] * 21 + [2500]
    audit = VolumeAudit().audit(
        {"code": "000001", "technical": {"kline": _rows(closes, volumes)}, "turnover_rate": 5.0}
    )

    assert audit["status"] == "watch"
    assert audit["price_volume_pattern"] == "distribution_volume"
