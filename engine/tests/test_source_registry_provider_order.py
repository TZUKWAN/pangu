from __future__ import annotations

from engine.sources.registry import build_default_registry


def test_daily_kline_chain_includes_baostock_before_eastmoney_placeholder() -> None:
    names = [p.name for p in build_default_registry().providers("daily_kline")]

    assert "baostock_daily_kline" in names
    assert names.index("baostock_daily_kline") < names.index("eastmoney_fflow")
