"""Built-in provider implementations for core market data."""

from __future__ import annotations

import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request

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


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, "", "-"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_prefix(code: str) -> str:
    code = str(code).strip().zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith("8"):
        return f"bj{code}"
    return f"sz{code}"


def _eastmoney_market_id(code: str) -> int:
    code = str(code).strip().zfill(6)
    return 1 if code.startswith(("6", "9")) else 0


def _urlopen_text(url: str, *, timeout: float = 10.0, headers: dict[str, str] | None = None) -> str:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_json(url: str, params: dict[str, Any], *, timeout: float = 10.0) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    text = _urlopen_text(
        full_url,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://quote.eastmoney.com/",
            "Origin": "https://quote.eastmoney.com",
        },
    )
    return json.loads(text)


def _codes_from_local_snapshot(date: str | None = None) -> list[str]:
    """从本地 all_spot 快照读取全市场代码列表，绕开东财/adata 网络依赖。

    优先读 data/snapshots/<date>/all_spot.parquet；date 为空时取最新快照目录。
    """
    try:
        snapshot_root = Path("data/snapshots")
        if not snapshot_root.exists():
            return []
        if date:
            target_dir = snapshot_root / date
        else:
            # 取最新日期目录
            dirs = sorted([d for d in snapshot_root.iterdir() if d.is_dir()], reverse=True)
            target_dir = dirs[0] if dirs else None
        if not target_dir or not target_dir.exists():
            return []
        spot_file = target_dir / "all_spot.parquet"
        if not spot_file.exists():
            return []
        df = pd.read_parquet(spot_file)
        if df is None or df.empty:
            return []
        # 找代码列
        code_col = None
        for col in df.columns:
            if "代码" in str(col) or "code" in str(col).lower():
                code_col = col
                break
        if code_col is None:
            code_col = df.columns[0]
        return df[code_col].astype(str).str.extract(r"(\d{6})", expand=False).dropna().tolist()
    except Exception:
        return []


def _all_a_share_codes(timeout_sec: float = 8.0, snapshot_date: str | None = None) -> list[str]:
    """获取全市场 A 股代码列表：本地快照优先，adata 兜底。

    本地快照绕开东财/adata 的网络依赖（东财 push2his 在部分网络环境不可达），
    adata 作为 fallback 保留多源能力。
    """
    # 1. 本地快照优先
    codes = _codes_from_local_snapshot(snapshot_date)
    if codes:
        return codes

    # 2. adata 兜底
    try:
        import adata  # type: ignore
    except Exception:
        return []

    import threading
    box: list[list[str]] = [[]]

    def _load() -> None:
        try:
            codes_df = adata.stock.info.all_code()
            if codes_df is None or codes_df.empty:
                return
            code_col = "stock_code" if "stock_code" in codes_df.columns else codes_df.columns[0]
            box[0] = codes_df[code_col].astype(str).str.extract(r"(\d{6})", expand=False).dropna().tolist()
        except Exception:
            box[0] = []

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    return box[0]


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


class SinaSpotProvider(SourceProvider):
    name = "sina_hq_all_spot"
    kind = "all_spot"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        if os.environ.get("PANGU_TDX_FALLBACK", "1") == "0":
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="disabled_by_PANGU_TDX_FALLBACK", data_mode=context.mode)
        # 本地快照日期目录用 YYYY-MM-DD；effective_date 是 YYYYMMDD
        snap_date = None
        if context.effective_date:
            d = str(context.effective_date)
            if len(d) == 8 and d.isdigit():
                snap_date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            else:
                snap_date = d
        codes = _all_a_share_codes(snapshot_date=snap_date)
        if not codes:
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning="code_list_unavailable", data_mode=context.mode)
        rows: list[dict[str, Any]] = []
        deadline = time.monotonic() + 18.0
        for i in range(0, len(codes), 250):
            if time.monotonic() > deadline:
                break
            batch = ",".join(_market_prefix(c) for c in codes[i:i + 250])
            url = f"https://hq.sinajs.cn/list={batch}"
            try:
                text = _urlopen_text(
                    url,
                    timeout=8.0,
                    headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"},
                )
            except Exception:
                continue
            for line in text.split(";"):
                if "=" not in line or '"' not in line:
                    continue
                key = line.split("=", 1)[0].split("_")[-1]
                code = key[-6:]
                values = line.split('"')[1].split(",")
                if len(values) < 31 or not values[0]:
                    continue
                open_price = _as_float(values[1])
                prev_close = _as_float(values[2])
                latest = _as_float(values[3])
                high = _as_float(values[4])
                low = _as_float(values[5])
                volume = _as_float(values[8])
                amount = _as_float(values[9])
                change_pct = None
                if latest is not None and prev_close not in (None, 0):
                    change_pct = round((latest - prev_close) / prev_close * 100, 4)
                rows.append({
                    "代码": code,
                    "名称": values[0],
                    "最新价": latest,
                    "涨跌幅": change_pct,
                    "成交量": volume,
                    "成交额": amount,
                    "今开": open_price,
                    "昨收": prev_close,
                    "最高": high,
                    "最低": low,
                    "日期": values[30] if len(values) > 30 else None,
                })
        if not rows:
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning="empty", data_mode=context.mode)
        return _quality(
            pd.DataFrame(rows),
            source=self.name,
            kind=self.kind,
            t0=t0,
            ctx=context,
            warnings=["sina_hq_missing_turnover_mcap_pe_pb_fund_flow"],
        )


class BaiduSpotProvider(SourceProvider):
    name = "baidu_gushitong_all_spot"
    kind = "all_spot"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        return failed_result(
            source=self.name,
            kind=self.kind,
            warning="baidu_all_spot_not_available_in_repo",
            data_mode=context.mode,
        )


class EfinanceSpotProvider(SourceProvider):
    name = "efinance_all_spot"
    kind = "all_spot"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        try:
            import efinance as ef  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning=f"import_failed:{exc}", data_mode=context.mode)
        try:
            df = ef.stock.get_realtime_quotes()
        except Exception as exc:  # noqa: BLE001
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, error=str(exc), data_mode=context.mode)
        if df is None or df.empty:
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning="empty", data_mode=context.mode)
        df = df.rename(columns={
            "股票代码": "代码",
            "股票名称": "名称",
            "最新价": "最新价",
            "涨跌幅": "涨跌幅",
            "成交量": "成交量",
            "成交额": "成交额",
            "换手率": "换手率",
            "市盈率-动态": "市盈率-动态",
            "市净率": "市净率",
            "总市值": "总市值",
            "流通市值": "流通市值",
        })
        return _quality(df, source=self.name, kind=self.kind, t0=t0, ctx=context)


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


class EastmoneyDailyKlineProvider(SourceProvider):
    name = "eastmoney_push2his_daily"
    kind = "daily_kline"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        symbol = str(context.symbol or "").zfill(6)
        if not symbol:
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="symbol_missing", data_mode=context.mode)
        fqt = {"qfq": "1", "hfq": "2", "": "0", None: "0"}.get(context.adjust, "1")
        params = {
            "secid": f"{_eastmoney_market_id(symbol)}.{symbol}",
            "klt": "101",
            "fqt": fqt,
            "lmt": str(max(1, int(context.days or 60))),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }
        try:
            data = _http_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", params, timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, error=str(exc), data_mode=context.mode)
        klines = ((data.get("data") or {}).get("klines") or [])
        rows: list[dict[str, Any]] = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 11:
                continue
            rows.append({
                "日期": parts[0],
                "股票代码": symbol,
                "开盘": _as_float(parts[1]),
                "收盘": _as_float(parts[2]),
                "最高": _as_float(parts[3]),
                "最低": _as_float(parts[4]),
                "成交量": _as_float(parts[5]),
                "成交额": _as_float(parts[6]),
                "振幅": _as_float(parts[7]),
                "涨跌幅": _as_float(parts[8]),
                "涨跌额": _as_float(parts[9]),
                "换手率": _as_float(parts[10]),
            })
        if not rows:
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning="empty", data_mode=context.mode)
        return _quality(pd.DataFrame(rows), source=self.name, kind=self.kind, t0=t0, ctx=context)


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
    name = "eastmoney_fflow"
    kind = "fund_flow"
    modes = ("live", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        t0 = time.monotonic()
        symbol = str(context.symbol or "").zfill(6) if context.symbol else ""
        if not symbol:
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="symbol_required_for_daykline", data_mode=context.mode)
        params = {
            "secid": f"{_eastmoney_market_id(symbol)}.{symbol}",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "lmt": str(max(1, int(context.days or 60))),
        }
        try:
            data = _http_json("https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get", params, timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, error=str(exc), data_mode=context.mode)
        klines = ((data.get("data") or {}).get("klines") or [])
        rows: list[dict[str, Any]] = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 6:
                continue
            rows.append({
                "日期": parts[0],
                "股票代码": symbol,
                "主力净流入-净额": _as_float(parts[1]),
                "小单净流入-净额": _as_float(parts[2]),
                "中单净流入-净额": _as_float(parts[3]),
                "大单净流入-净额": _as_float(parts[4]),
                "超大单净流入-净额": _as_float(parts[5]),
            })
        if not rows:
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning="empty", data_mode=context.mode)
        return _quality(pd.DataFrame(rows), source=self.name, kind=self.kind, t0=t0, ctx=context)


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
        t0 = time.monotonic()
        token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("PANGU_TUSHARE_TOKEN")
        if not token:
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="token_missing", data_mode=context.mode)
        symbol = str(context.symbol or "").zfill(6) if context.symbol else ""
        if not symbol:
            return failed_result(source=self.name, kind=self.kind, latency=0.0, warning="symbol_required", data_mode=context.mode)
        try:
            import tushare as ts  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning=f"import_failed:{exc}", data_mode=context.mode)
        suffix = "SH" if symbol.startswith(("6", "9")) else "SZ"
        ts_code = f"{symbol}.{suffix}"
        end_dt = datetime.strptime(context.effective_date, "%Y%m%d") if context.effective_date else datetime.now()
        start_dt = end_dt - timedelta(days=max(1, int(context.days or 60)) * 2)
        try:
            pro = ts.pro_api(token)
            df = pro.moneyflow(
                ts_code=ts_code,
                start_date=start_dt.strftime("%Y%m%d"),
                end_date=end_dt.strftime("%Y%m%d"),
            )
        except Exception as exc:  # noqa: BLE001
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, error=str(exc), data_mode=context.mode)
        if df is None or df.empty:
            return failed_result(source=self.name, kind=self.kind, latency=time.monotonic() - t0, warning="empty", data_mode=context.mode)
        out = df.rename(columns={
            "trade_date": "日期",
            "ts_code": "股票代码",
            "net_mf_amount": "主力净流入-净额",
            "buy_lg_amount": "大单买入额",
            "sell_lg_amount": "大单卖出额",
            "buy_elg_amount": "超大单买入额",
            "sell_elg_amount": "超大单卖出额",
        }).copy()
        out["股票代码"] = symbol
        if "主力净流入-净额" in out.columns:
            out["主力净流入-净额"] = pd.to_numeric(out["主力净流入-净额"], errors="coerce") * 10000
        return _quality(out, source=self.name, kind=self.kind, t0=t0, ctx=context)


class UnavailableProvider(SourceProvider):
    name = "unavailable"

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.modes = ("live", "snapshot", "diagnostic")

    def fetch(self, context: SourceContext) -> SourceResult:
        return failed_result(source=self.name, kind=self.kind, warning=f"{self.kind}_unavailable", data_mode=context.mode)
