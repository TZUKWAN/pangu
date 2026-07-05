"""snapshot 与多源 fallback 纯逻辑测试（mock 数据，不依赖网络）。"""

import hashlib
from datetime import datetime, timedelta

import pandas as pd
import pytest

from engine import data_loader as dl_mod
from engine.data_loader import MultiSourceDataLoader
from engine.snapshot import SnapshotBuilder


class FakeDataLoader:
    """内存版 DataLoader，返回预设 DataFrame。"""

    def __init__(self):
        self._spot = pd.DataFrame({"代码": ["000001"], "名称": ["测试"], "最新价": [10.0]})
        self._boards = pd.DataFrame({"板块名称": ["芯片"], "涨跌幅": [3.0]})
        self._zt = pd.DataFrame({"代码": ["000001"], "名称": ["测试"]})
        self._broke = pd.DataFrame({"代码": ["000002"], "名称": ["炸板"]})
        self._dt = pd.DataFrame({"代码": ["000003"], "名称": ["跌停"]})
        self._sector = pd.DataFrame({"名称": ["芯片"], "今日涨跌幅": [3.0]})

    def all_spot(self):
        return self._spot

    def concept_boards(self):
        return self._boards

    def limit_up_pool(self, date=None):
        return self._zt

    def broke_pool(self, date=None):
        return self._broke

    def limit_down_pool(self, date=None):
        return self._dt

    def sector_fund_flow_rank(self, indicator="今日"):
        return self._sector


@pytest.fixture
def no_akshare(monkeypatch):
    """把 akshare 置空，并关闭 tdx 联网兜底，避免测试触发网络请求。

    PANGU_TDX_FALLBACK=0 让 data_loader 跳过腾讯/mootdx 联网兜底
    （mootdx TCP 探测 + adata 全市场联网会卡住测试进程）。
    """
    original = dl_mod.ak
    monkeypatch.setattr(dl_mod, "ak", None)
    monkeypatch.setenv("PANGU_TDX_FALLBACK", "0")
    yield
    monkeypatch.setattr(dl_mod, "ak", original)
    monkeypatch.delenv("PANGU_TDX_FALLBACK", raising=False)


# ---------------------------------------------------------------------- #
def test_snapshot_builder_saves_parquet(tmp_path):
    dl = FakeDataLoader()
    builder = SnapshotBuilder(dl, snapshot_dir=tmp_path)
    res = builder.build("2025-01-01")

    assert res.date_dir == "2025-01-01"
    expected_tables = {
        "all_spot",
        "concept_boards",
        "limit_up_pool",
        "broke_pool",
        "limit_down_pool",
        "sector_fund_flow_rank",
    }
    assert set(res.rows.keys()) == expected_tables
    assert (tmp_path / "2025-01-01" / "all_spot.parquet").exists()

    df = pd.read_parquet(tmp_path / "2025-01-01" / "all_spot.parquet")
    assert df.iloc[0]["代码"] == "000001"


def test_snapshot_builder_YYYYMMDD_date(tmp_path):
    dl = FakeDataLoader()
    builder = SnapshotBuilder(dl, snapshot_dir=tmp_path)
    res = builder.build("20250115")
    assert res.date_dir == "2025-01-15"


def test_snapshot_builder_isolates_single_failure(tmp_path):
    """单表失败不应阻断其余表的保存。"""

    class BadDataLoader(FakeDataLoader):
        def all_spot(self):
            raise RuntimeError("boom")

    dl = BadDataLoader()
    builder = SnapshotBuilder(dl, snapshot_dir=tmp_path)
    res = builder.build("2025-01-01")

    failed = [e.split(":")[0] for e in res.errors]
    assert "all_spot" in failed
    assert (tmp_path / "2025-01-01" / "concept_boards.parquet").exists()


# ---------------------------------------------------------------------- #
def test_multisource_all_spot_fallback_to_snapshot(no_akshare, tmp_path):
    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snapshots"
    snap_dir = snapshot_dir / "2025-01-01"
    snap_dir.mkdir(parents=True)
    df = pd.DataFrame({"代码": ["000002"], "名称": ["快照股"], "最新价": [20.0]})
    df.to_parquet(snap_dir / "all_spot.parquet", index=False)

    dl = MultiSourceDataLoader(
        cache_dir=cache_dir,
        snapshot_dir=snapshot_dir,
        retry_times=1,
        backoff_seconds=0.0,
    )
    result = dl.all_spot()

    assert len(result) == 1
    assert result.iloc[0]["代码"] == "000002"


def test_multisource_daily_kline_fallback_to_stale_cache(no_akshare, tmp_path, monkeypatch):
    """当所有真实 daily_kline provider 失败时，fallback 到 stale cache。"""
    from engine.source_quality import failed_result
    from engine.sources.providers.core import (
        SinaDailyKlineProvider, TencentDailyKlineProvider, MootdxDailyKlineProvider,
        BaiduDailyKlineProvider, AdataDailyKlineProvider, BaostockDailyKlineProvider,
        EastmoneyDailyKlineProvider,
    )

    def fail(self, context):
        return failed_result(source=self.name, kind=self.kind, warning="forced_fail", data_mode=context.mode)

    for cls in (SinaDailyKlineProvider, TencentDailyKlineProvider, MootdxDailyKlineProvider,
                BaiduDailyKlineProvider, AdataDailyKlineProvider, BaostockDailyKlineProvider,
                EastmoneyDailyKlineProvider):
        monkeypatch.setattr(cls, "fetch", fail)

    cache_dir = tmp_path / "cache"
    snapshot_dir = tmp_path / "snapshots"
    cache_dir.mkdir(parents=True)

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=120)).strftime("%Y%m%d")
    cache_key = f"kline:000001:qfq:{start}:{end}"
    h = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:16]
    df = pd.DataFrame({"日期": ["20250101"], "收盘": [10.0]})
    df.to_parquet(cache_dir / f"{h}.parquet", index=False)

    dl = MultiSourceDataLoader(
        cache_dir=cache_dir,
        snapshot_dir=snapshot_dir,
        retry_times=1,
        backoff_seconds=0.0,
    )
    result = dl.daily_kline("000001")

    assert len(result) == 1
    assert result.iloc[0]["日期"] == "20250101"


def test_multisource_empty_when_all_sources_fail(no_akshare, tmp_path, monkeypatch):
    """全部数据源不可用时返回空 DataFrame，不抛异常。"""
    from engine.source_quality import failed_result
    from engine.sources.providers.core import (
        SinaDailyKlineProvider, TencentDailyKlineProvider, MootdxDailyKlineProvider,
        BaiduDailyKlineProvider, AdataDailyKlineProvider, BaostockDailyKlineProvider,
        EastmoneyDailyKlineProvider, ThsSpotProvider, TencentSpotProvider, SinaSpotProvider,
        BaiduSpotProvider, AdataSpotProvider, EfinanceSpotProvider,
    )

    def fail(self, context):
        return failed_result(source=self.name, kind=self.kind, warning="forced_fail", data_mode=context.mode)

    for cls in (SinaDailyKlineProvider, TencentDailyKlineProvider, MootdxDailyKlineProvider,
                BaiduDailyKlineProvider, AdataDailyKlineProvider, BaostockDailyKlineProvider,
                EastmoneyDailyKlineProvider, ThsSpotProvider, TencentSpotProvider, SinaSpotProvider,
                BaiduSpotProvider, AdataSpotProvider, EfinanceSpotProvider):
        monkeypatch.setattr(cls, "fetch", fail)

    dl = MultiSourceDataLoader(
        cache_dir=tmp_path / "cache",
        snapshot_dir=tmp_path / "snapshots",
        retry_times=1,
        backoff_seconds=0.0,
    )
    assert dl.all_spot().empty
    assert dl.daily_kline("000001").empty
