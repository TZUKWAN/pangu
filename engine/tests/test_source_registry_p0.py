"""P0 source registry and quality contract tests."""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from engine import data_loader as dl_mod
from engine.data_loader import MultiSourceDataLoader
from engine.source_quality import FIELD_MISSING, assess_dataframe, failed_result
from engine.sources.providers import core as provider_core


@pytest.fixture
def loader_env(monkeypatch, tmp_path):
    monkeypatch.setattr(dl_mod, "ak", None)
    monkeypatch.setenv("PANGU_TDX_FALLBACK", "1")
    monkeypatch.delenv("PANGU_DATA_MODE", raising=False)
    monkeypatch.delenv("PANGU_DATA_DATE", raising=False)
    return {
        "cache_dir": tmp_path / "cache",
        "snapshot_dir": tmp_path / "snapshots",
        "retry_times": 1,
        "backoff_seconds": 0.0,
    }


def test_all_spot_registry_falls_back_to_tencent(monkeypatch, loader_env):
    def ths_empty():
        return pd.DataFrame()

    def tencent_ok():
        return pd.DataFrame(
            {
                "代码": ["000001"],
                "名称": ["平安银行"],
                "最新价": [10.0],
                "涨跌幅": [1.2],
                "成交量": [1000],
                "成交额": [1000000],
                "换手率": [2.5],
            }
        )

    monkeypatch.setattr("engine.tdx_source.ths_all_spot", ths_empty)
    monkeypatch.setattr("engine.tdx_source.tencent_all_spot", tencent_ok)

    dl = MultiSourceDataLoader(**loader_env)
    df = dl.all_spot()

    assert len(df) == 1
    assert df.attrs["source_quality"]["source"] == "tencent_qt_all_spot"
    assert dl.get_source_quality("all_spot")["source"] == "tencent_qt_all_spot"
    assert [step["source"] for step in df.attrs["source_chain"][:2]] == ["ths_all_spot", "tencent_qt_all_spot"]


def test_daily_kline_registry_falls_back_to_tencent(monkeypatch, loader_env):
    def sina_fail(self, context):
        return failed_result(source=self.name, kind=self.kind, warning="forced_empty", data_mode=context.mode)

    def tencent_ok(self, context):
        df = pd.DataFrame(
            {
                "日期": ["20260703"],
                "股票代码": ["000001"],
                "开盘": [10.0],
                "收盘": [10.5],
                "最高": [10.8],
                "最低": [9.9],
                "成交量": [1000],
                "成交额": [1000000],
                "换手率": [1.5],
            }
        )
        return assess_dataframe(df, source=self.name, kind=self.kind, data_mode=context.mode, expected_date=context.effective_date)

    monkeypatch.setattr(provider_core.SinaDailyKlineProvider, "fetch", sina_fail)
    monkeypatch.setattr(provider_core.TencentDailyKlineProvider, "fetch", tencent_ok)

    dl = MultiSourceDataLoader(**loader_env)
    df = dl.daily_kline("000001", date="20260703")

    assert len(df) == 1
    assert df.attrs["source_quality"]["source"] == "tencent_kline_qfq"
    assert df.attrs["source_quality"]["date_matched_or_not"] is True


def test_fund_flow_unavailable_is_explicit_not_exception(monkeypatch, loader_env):
    def fail(self, context):
        return failed_result(source=self.name, kind=self.kind, warning="forced_unavailable", data_mode=context.mode)

    monkeypatch.setattr(provider_core.ThsFundFlowProvider, "fetch", fail)
    monkeypatch.setattr(provider_core.AdataFundFlowProvider, "fetch", fail)
    monkeypatch.setattr(provider_core.TushareMoneyFlowProvider, "fetch", fail)

    dl = MultiSourceDataLoader(**loader_env)
    df = dl.all_fund_flow_snapshot()

    assert df.empty
    quality = dl.get_source_quality("fund_flow")
    assert quality["status"] == "failed"
    assert quality["source"] == "unavailable"
    assert any(step["source"] == "unavailable" for step in dl.get_source_quality("fund_flow_chain"))


def test_snapshot_mode_reads_only_exact_date(monkeypatch, tmp_path):
    monkeypatch.setattr(dl_mod, "ak", None)
    monkeypatch.setenv("PANGU_DATA_MODE", "snapshot")
    monkeypatch.setenv("PANGU_DATA_DATE", "20260703")
    monkeypatch.setenv("PANGU_TDX_FALLBACK", "0")

    snap_older = tmp_path / "snapshots" / "2026-07-02"
    snap_older.mkdir(parents=True)
    pd.DataFrame({"代码": ["000001"], "名称": ["旧快照"], "最新价": [9.0]}).to_parquet(snap_older / "all_spot.parquet", index=False)

    dl = MultiSourceDataLoader(cache_dir=tmp_path / "cache", snapshot_dir=tmp_path / "snapshots")
    assert dl.all_spot().empty
    assert dl.get_source_quality("all_spot")["status"] == "failed"

    snap_exact = tmp_path / "snapshots" / "2026-07-03"
    snap_exact.mkdir(parents=True)
    pd.DataFrame({"代码": ["000002"], "名称": ["当日快照"], "最新价": [10.0]}).to_parquet(snap_exact / "all_spot.parquet", index=False)

    dl2 = MultiSourceDataLoader(cache_dir=tmp_path / "cache2", snapshot_dir=tmp_path / "snapshots")
    df = dl2.all_spot()
    assert len(df) == 1
    assert df.iloc[0]["代码"] == "000002"
    assert df.attrs["source_quality"]["source"] == "snapshot_exact"


def test_diagnostic_mode_allows_stale_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(dl_mod, "ak", None)
    monkeypatch.setenv("PANGU_DATA_MODE", "diagnostic")
    monkeypatch.setenv("PANGU_TDX_FALLBACK", "0")

    dl = MultiSourceDataLoader(cache_dir=tmp_path / "cache", snapshot_dir=tmp_path / "snapshots", cache_ttl_minutes=0)
    cached = pd.DataFrame({"代码": ["000003"], "名称": ["旧缓存"], "最新价": [11.0], "换手率": [pd.NA]})
    dl._cache_put("all_spot", cached)
    cache_file = next((tmp_path / "cache").glob("*.parquet"))
    old = time.time() - 86400
    os.utime(cache_file, (old, old))

    df = dl.all_spot()
    assert len(df) == 1
    assert df.attrs["source_quality"]["source"] == "stale_cache"
    assert df.attrs["source_quality"]["stale_or_not"] is True


def test_turnover_missing_is_not_treated_as_zero():
    result = assess_dataframe(
        pd.DataFrame({"代码": ["000001"], "名称": ["无换手"], "最新价": [10.0]}),
        source="unit",
        kind="all_spot",
    )
    df = result.data

    assert pd.isna(df.iloc[0]["turnover_rate"])
    assert bool(df.iloc[0]["turnover_missing"]) is True
    assert df.iloc[0]["turnover_status"] == FIELD_MISSING
    assert result.quality.field_quality["turnover_rate"] == FIELD_MISSING
