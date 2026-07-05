from __future__ import annotations

from types import SimpleNamespace

from engine.sources.registry import build_default_registry
from engine.sources.base import SourceContext
from engine.sources.providers import core as provider_core


def test_daily_kline_chain_includes_baostock_before_eastmoney_placeholder() -> None:
    names = [p.name for p in build_default_registry().providers("daily_kline")]

    assert "baostock_daily_kline" in names
    assert "eastmoney_push2his_daily" in names
    assert names.index("baostock_daily_kline") < names.index("eastmoney_push2his_daily")


def test_all_spot_chain_includes_requested_middle_sources() -> None:
    names = [p.name for p in build_default_registry().providers("all_spot")]

    assert names[:7] == [
        "snapshot_exact",
        "snapshot_exact",
        "stale_cache",
        "ths_all_spot",
        "tencent_qt_all_spot",
        "sina_hq_all_spot",
        "baidu_gushitong_all_spot",
    ]
    assert "adata_all_spot" in names
    assert "efinance_all_spot" in names


def test_eastmoney_daily_kline_parses_push2his(monkeypatch) -> None:
    def fake_json(url, params, *, timeout=10.0):
        assert "stock/kline/get" in url
        assert params["secid"] == "0.000001"
        return {"data": {"klines": ["2026-07-03,10,10.5,10.8,9.9,1000,1000000,9,2.1,0.2,1.5"]}}

    monkeypatch.setattr(provider_core, "_http_json", fake_json)
    provider = provider_core.EastmoneyDailyKlineProvider()
    result = provider.fetch(SourceContext(loader=SimpleNamespace(), symbol="000001", date="20260703"))

    assert result.quality.ok is True
    assert result.quality.source == "eastmoney_push2his_daily"
    assert result.data.iloc[0]["股票代码"] == "000001"
    assert result.data.iloc[0]["收盘"] == 10.5


def test_eastmoney_fund_flow_parses_daykline(monkeypatch) -> None:
    def fake_json(url, params, *, timeout=10.0):
        assert "fflow/daykline/get" in url
        assert params["secid"] == "1.600519"
        return {"data": {"klines": ["2026-07-03,100,10,20,30,40"]}}

    monkeypatch.setattr(provider_core, "_http_json", fake_json)
    provider = provider_core.EastmoneyFundFlowProvider()
    result = provider.fetch(SourceContext(loader=SimpleNamespace(), symbol="600519", date="20260703"))

    assert result.quality.ok is True
    assert result.quality.source == "eastmoney_fflow"
    assert result.data.iloc[0]["股票代码"] == "600519"
    assert result.data.iloc[0]["主力净流入-净额"] == 100.0


def test_tushare_moneyflow_requires_token(monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("PANGU_TUSHARE_TOKEN", raising=False)

    provider = provider_core.TushareMoneyFlowProvider()
    result = provider.fetch(SourceContext(loader=SimpleNamespace(), symbol="000001", date="20260703"))

    assert result.quality.ok is False
    assert result.quality.source == "tushare_moneyflow"
    assert "token_missing" in result.quality.warnings
