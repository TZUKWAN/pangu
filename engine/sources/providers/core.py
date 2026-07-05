"""Built-in provider implementations for core market data."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from engine.source_quality import SourceResult, assess_dataframe, failed_result
from engine.sources.base import SourceContext, SourceProvider, mode_tuple


def _quality(
    df: pd.DataFrame,
    *,
    source: str,
    kind: str,
    t0: float,
    ctx: SourceContext,
    stale: bool = False,
    warnings: list[str] | None = None,
) -> SourceResult:
    return assess_dataframe(
        df,
        source=source,
        kind=kind,
        latency=time.monotonic() - t0,
        warnings=warnings,
        stale=stale,
        expected_date=ctx.effective_date,
        data_mode=ctx.mode,
    )


class ExactSnapshotProvider(SourceProvider):
    name = "snapshot_exact"

    def __init__(self, kind: str, modes: tuple[str, ...] | None = None) -> None:
        self.kind = kind
        self.modes = mode_tuple(modes)

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        loader = context.loader
        df = None
        if hasattr(loader, "_load_snapshot_exact"):
            name = "all_spot" if self.kind == "all_spot" else self.kind
            df = loader._load_snapshot_exact(name)  # noqa: SLF001
        if df is None:
            return failed_result(
                source=self.name,
                kind=self.kind,
                latency=time.monotonic() - t0,
                warning=f"snapshot_missing:{context.data_date or ''}",
                data_mode=context.mode,
            )
        return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context)


class LocalSnapshotProvider(SourceProvider):
    name = "local_snapshot"

    def __init__(self, kind: str, modes: tuple[str, ...] | None = None) -> None:
        self.kind = kind
        self.modes = mode_tuple(modes)

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        loader = context.loader
        df = None
        if hasattr(loader, "_load_snapshot"):
            name = "all_spot" if self.kind == "all_spot" else self.kind
            df = loader._load_snapshot(name, as_of_date=context.effective_date)  # noqa: SLF001
        if df is None:
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, data_mode=context.mode)
        return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context, stale=True)


class StaleCacheProvider(SourceProvider):
    name = "stale_cache"

    def __init__(self, kind: str, modes: tuple[str, ...] | None = None) -> None:
        self.kind = kind
        self.modes = mode_tuple(modes)

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        key = context.cache_key or self.kind
        df = context.loader._cache_get_stale(key) if hasattr(context.loader, "_cache_get_stale") else None  # noqa: SLF001
        if df is None:
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, data_mode=context.mode)
        return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context, stale=True)


class ThsSpotProvider(SourceProvider):
    name = "ths_all_spot"
    kind = "all_spot"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        if os.environ.get("PANGU_TDX_FALLBACK", "1") == "0":
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="disabled_by_PANGU_TDX_FALLBACK", data_mode=context.mode)
        from engine import tdx_source

        return _quality(tdx_source.ths_all_spot(), source=self.name, kind=self.kind, t0=t0, ctx=context)


class TencentSpotProvider(SourceProvider):
    name = "tencent_qt_all_spot"
    kind = "all_spot"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        if os.environ.get("PANGU_TDX_FALLBACK", "1") == "0":
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="disabled_by_PANGU_TDX_FALLBACK", data_mode=context.mode)
        from engine import tdx_source

        return _quality(tdx_source.tencent_all_spot(), source=self.name, kind=self.kind, t0=t0, ctx=context)


class AdataSpotProvider(SourceProvider):
    name = "adata_all_spot"
    kind = "all_spot"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        if os.environ.get("PANGU_TDX_FALLBACK", "1") == "0":
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="disabled_by_PANGU_TDX_FALLBACK", data_mode=context.mode)
        from engine import adata_source

        return _quality(adata_source.all_spot(timeout_sec=20.0), source=self.name, kind=self.kind, t0=t0, ctx=context)


class SinaDailyKlineProvider(SourceProvider):
    name = "sina_stock_zh_a_daily"
    kind = "daily_kline"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        symbol = str(context.symbol or "").zfill(6)
        if not symbol:
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="symbol_missing", data_mode=context.mode)
        loader = context.loader
        from engine.data_loader import _guess_sina_symbol, _normalize_sina_kline

        end_dt = datetime.strptime(context.effective_date, "%Y%m%d") if context.effective_date else datetime.now()
        start = (end_dt - timedelta(days=context.days * 2)).strftime("%Y%m%d")
        end = end_dt.strftime("%Y%m%d")
        cache_key = (context.cache_key or f"kline:{symbol}:{context.adjust}:{start}:{end}") + ":sina"

        def _fetch() -> pd.DataFrame:
            old_retry = getattr(loader, "retry_times", 1)
            loader.retry_times = 1
            try:
                return loader._call(  # noqa: SLF001
                    "stock_zh_a_daily",
                    cache_key,
                    symbol=_guess_sina_symbol(symbol),
                    start_date=start,
                    end_date=end,
                    adjust=context.adjust,
                )
            finally:
                loader.retry_times = old_retry

        df = loader._call_with_timeout(_fetch, timeout_seconds=6.0) if hasattr(loader, "_call_with_timeout") else _fetch()  # noqa: SLF001
        if df is None or df.empty:
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, data_mode=context.mode)
        df = _normalize_sina_kline(df, symbol)
        if len(df) > context.days:
            df = df.tail(context.days).reset_index(drop=True)
        return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context)


class TencentDailyKlineProvider(SourceProvider):
    name = "tencent_kline_qfq"
    kind = "daily_kline"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        if os.environ.get("PANGU_TDX_FALLBACK", "1") == "0":
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="disabled_by_PANGU_TDX_FALLBACK", data_mode=context.mode)
        from engine import tdx_source

        df = tdx_source.tencent_kline_qfq(str(context.symbol or "").zfill(6), days=context.days, adjust=context.adjust)
        return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context)


class AdataDailyKlineProvider(SourceProvider):
    name = "adata_daily_kline"
    kind = "daily_kline"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        if os.environ.get("PANGU_TDX_FALLBACK", "1") == "0":
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="disabled_by_PANGU_TDX_FALLBACK", data_mode=context.mode)
        from engine import adata_source

        df = adata_source.daily_kline(
            str(context.symbol or "").zfill(6),
            days=context.days,
            adjust=context.adjust,
            date=context.effective_date,
        )
        return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context)


class BaostockDailyKlineProvider(SourceProvider):
    name = "baostock_daily_kline"
    kind = "daily_kline"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        symbol = str(context.symbol or "").zfill(6)
        if not symbol:
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="symbol_missing", data_mode=context.mode)
        if os.environ.get("PANGU_TDX_FALLBACK", "1") == "0":
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="disabled_by_PANGU_TDX_FALLBACK", data_mode=context.mode)
        try:
            import baostock as bs  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning=f"import_failed:{exc}", data_mode=context.mode)

        market_code = f"sh.{symbol}" if symbol.startswith(("6", "9")) else f"sz.{symbol}"
        end_dt = datetime.strptime(context.effective_date, "%Y%m%d") if context.effective_date else datetime.now()
        start_dt = end_dt - timedelta(days=context.days * 2)
        adjustflag = {"qfq": "2", "hfq": "1", "": "3", None: "3"}.get(context.adjust, "2")
        lg = None
        try:
            lg = bs.login()
            if getattr(lg, "error_code", "0") != "0":
                return failed_result(
                    source=self.name,
                    kind=self.kind,
                    latency=time.monotonic() - t0,
                    warning=f"login_failed:{getattr(lg, 'error_msg', '')}",
                    data_mode=context.mode,
                )
            rs = bs.query_history_k_data_plus(
                market_code,
                "date,code,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag=adjustflag,
            )
            if getattr(rs, "error_code", "0") != "0":
                return failed_result(
                    source=self.name,
                    kind=self.kind,
                    latency=time.monotonic() - t0,
                    warning=f"query_failed:{getattr(rs, 'error_msg', '')}",
                    data_mode=context.mode,
                )
            rows: list[list[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning="empty", data_mode=context.mode)
            df = pd.DataFrame(rows, columns=rs.fields)
            df = df.rename(columns={
                "date": "日期",
                "code": "股票代码",
                "open": "开盘",
                "close": "收盘",
                "high": "最高",
                "low": "最低",
                "volume": "成交量",
                "amount": "成交额",
                "turn": "换手率",
                "pctChg": "涨跌幅",
            })
            if "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"], errors="coerce").dt.strftime("%Y%m%d")
            df["股票代码"] = symbol
            for col in ("开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率", "涨跌幅"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if len(df) > context.days:
                df = df.tail(context.days).reset_index(drop=True)
            return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context)
        except Exception as exc:  # noqa: BLE001
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, error=str(exc), data_mode=context.mode)
        finally:
            if lg is not None:
                try:
                    bs.logout()
                except Exception:
                    pass


class MootdxDailyKlineProvider(SourceProvider):
    name = "mootdx_daily_kline"
    kind = "daily_kline"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        if os.environ.get("PANGU_TDX_FALLBACK", "1") == "0":
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="disabled_by_PANGU_TDX_FALLBACK", data_mode=context.mode)
        from engine import tdx_source

        return _quality(tdx_source.tdx_daily_kline(str(context.symbol or "").zfill(6), days=context.days), source=self.name, kind=self.kind, t0=t0, ctx=context)


class ThsFundFlowProvider(SourceProvider):
    name = "ths_fund_flow"
    kind = "fund_flow"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        loader = context.loader
        if context.symbol:
            df = super(type(loader), loader).individual_fund_flow(context.symbol) if False else pd.DataFrame()
            # Use loader._call directly to avoid recursing into MultiSourceDataLoader.
            try:
                df = loader._call("stock_fund_flow_individual", f"fund_ths:{context.symbol}", symbol=str(context.symbol).zfill(6))  # noqa: SLF001
            except Exception:
                df = pd.DataFrame()
        else:
            try:
                df = loader._call("stock_fund_flow_individual", "fund_ths:all_spot", symbol="即时")  # noqa: SLF001
            except Exception:
                df = pd.DataFrame()
        return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context)


class EastmoneyFundFlowProvider(SourceProvider):
    """Optional Eastmoney-style provider placeholder.

    The repository has intentionally removed Eastmoney from parts of the main
    chain because it was unstable. We keep this provider explicit so the chain
    can record that the source is unavailable instead of silently pretending it
    was tried.
    """

    name = "eastmoney_fflow"
    modes = ("live", "diagnostic")

    def __init__(self, kind: str = "fund_flow") -> None:
        self.kind = kind

    def fetch(self, context: SourceContext) -> SourceResult:
        return failed_result(
            source=self.name,
            kind=self.kind,
            warning="eastmoney_provider_not_enabled",
            data_mode=context.mode,
        )


class AdataFundFlowProvider(SourceProvider):
    name = "adata_fund_flow"
    kind = "fund_flow"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        if os.environ.get("PANGU_TDX_FALLBACK", "1") == "0":
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="disabled_by_PANGU_TDX_FALLBACK", data_mode=context.mode)
        from engine import adata_source

        if context.symbol:
            df = adata_source.individual_fund_flow(str(context.symbol).zfill(6), days=context.days)
        else:
            df = adata_source.concept_fund_flow()
        return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context)


class TushareMoneyFlowProvider(SourceProvider):
    name = "tushare_moneyflow"
    kind = "fund_flow"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("PANGU_TUSHARE_TOKEN")
        if not token:
            return failed_result(source=self.name, kind=self.kind, warning="token_missing", data_mode=context.mode)
        return failed_result(source=self.name, kind=self.kind, warning="not_implemented_in_repo", data_mode=context.mode)


class UnavailableProvider(SourceProvider):
    name = "unavailable"

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.modes = ("live", "snapshot", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        return failed_result(source=self.name, kind=self.kind, warning=f"{self.kind}_unavailable", data_mode=context.mode)
