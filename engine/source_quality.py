"""Source and field quality metadata for market data providers.

This module deliberately keeps the quality model small and serializable so it
can travel through DataFrame.attrs, pipeline source_status, reports, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd


FIELD_OK = "ok"
FIELD_MISSING = "missing"
FIELD_ESTIMATED = "estimated"
FIELD_STALE = "stale"
FIELD_INVALID = "invalid"


FIELD_ALIASES: dict[str, list[str]] = {
    "code": ["代码", "股票代码", "证券代码", "code", "stock_code", "symbol"],
    "name": ["名称", "股票简称", "简称", "name", "short_name"],
    "latest_price": ["最新价", "最新", "现价", "price", "close"],
    "change_pct": ["涨跌幅", "涨跌幅%", "change_pct", "pct_chg"],
    "volume": ["成交量", "volume", "vol"],
    "amount": ["成交额", "成交金额", "amount"],
    "turnover_rate": ["换手率", "turnover_rate", "turnover_ratio", "turnover_pct"],
    "float_mcap": ["流通市值", "float_mcap", "float_mv", "circ_mv"],
    "total_mcap": ["总市值", "市值", "total_mcap", "mcap", "total_mv"],
    "pe": ["PE", "市盈率", "市盈率-动态", "pe", "pe_ttm"],
    "pb": ["PB", "市净率", "pb"],
    "main_net_inflow": ["主力净流入", "主力净流入-净额", "净额", "main_net_inflow"],
    "date": ["日期", "交易日期", "trade_date", "date"],
    "open": ["开盘", "开盘价", "open"],
    "close": ["收盘", "收盘价", "close"],
    "high": ["最高", "最高价", "high"],
    "low": ["最低", "最低价", "low"],
}


REQUIRED_FIELDS: dict[str, list[str]] = {
    "all_spot": [
        "code",
        "name",
        "latest_price",
        "change_pct",
        "volume",
        "amount",
        "turnover_rate",
        "float_mcap",
        "total_mcap",
        "pe",
        "pb",
        "main_net_inflow",
    ],
    "daily_kline": ["date", "code", "open", "close", "high", "low", "volume", "amount", "turnover_rate"],
    "fund_flow": ["code", "date", "main_net_inflow"],
}


@dataclass
class SourceQuality:
    source: str
    ok: bool
    row_count: int = 0
    latency: float = 0.0
    warnings: list[str] = field(default_factory=list)
    field_quality: dict[str, str] = field(default_factory=dict)
    stale_or_not: bool = False
    date_matched_or_not: Optional[bool] = None
    data_mode: str = "live"
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.ok:
            if self.stale_or_not or any(v in {FIELD_MISSING, FIELD_INVALID, FIELD_STALE} for v in self.field_quality.values()):
                return "degraded"
            return "ok"
        return "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ok": self.ok,
            "status": self.status,
            "row_count": self.row_count,
            "latency": round(float(self.latency or 0.0), 4),
            "warnings": list(self.warnings),
            "field_quality": dict(self.field_quality),
            "stale_or_not": bool(self.stale_or_not),
            "date_matched_or_not": self.date_matched_or_not,
            "data_mode": self.data_mode,
            "errors": list(self.errors),
        }


@dataclass
class SourceResult:
    data: pd.DataFrame
    quality: SourceQuality
    chain: list[dict[str, Any]] = field(default_factory=list)

    def with_attrs(self) -> pd.DataFrame:
        df = self.data.copy()
        df.attrs["source_quality"] = self.quality.to_dict()
        df.attrs["source_chain"] = list(self.chain)
        return df


def find_field_column(df: pd.DataFrame, field: str) -> Optional[str]:
    aliases = FIELD_ALIASES.get(field, [field])
    cols = [str(c) for c in df.columns]
    lowered = {c.lower(): c for c in cols}
    for alias in aliases:
        if alias in cols:
            return alias
        key = str(alias).lower()
        if key in lowered:
            return lowered[key]
    for alias in aliases:
        alias_s = str(alias).lower()
        matches = [c for c in cols if alias_s in c.lower()]
        if matches:
            return min(matches, key=len)
    return None


def normalize_code_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d{6})", expand=False).fillna(series.astype(str)).str.zfill(6)


def normalize_turnover_fields(df: pd.DataFrame, source: str = "") -> tuple[pd.DataFrame, str]:
    """Add explicit turnover metadata without treating missing turnover as real 0."""

    out = df.copy()
    col = find_field_column(out, "turnover_rate")
    if col is None:
        out["turnover_rate"] = pd.NA
        out["turnover_missing"] = True
        out["turnover_source"] = "missing"
        out["turnover_status"] = FIELD_MISSING
        if "换手率" not in out.columns:
            out["换手率"] = pd.NA
        return out, FIELD_MISSING

    numeric = pd.to_numeric(out[col], errors="coerce")
    missing = numeric.isna()
    out["turnover_rate"] = numeric.where(~missing, pd.NA)
    out["turnover_missing"] = missing
    out["turnover_source"] = source or col
    out["turnover_status"] = missing.map(lambda m: FIELD_MISSING if bool(m) else FIELD_OK)
    if col != "换手率" and "换手率" not in out.columns:
        out["换手率"] = out["turnover_rate"]
    if len(out) == 0:
        return out, FIELD_MISSING
    if bool(missing.all()):
        return out, FIELD_MISSING
    if bool(missing.any()):
        return out, FIELD_INVALID
    return out, FIELD_OK


def assess_dataframe(
    df: pd.DataFrame | None,
    *,
    source: str,
    kind: str,
    latency: float = 0.0,
    warnings: Optional[list[str]] = None,
    stale: bool = False,
    expected_date: str | None = None,
    data_mode: str = "live",
    errors: Optional[list[str]] = None,
) -> SourceResult:
    warnings = list(warnings or [])
    errors = list(errors or [])
    if df is None:
        df = pd.DataFrame()
    clean = df.copy()
    field_quality: dict[str, str] = {}
    if not clean.empty:
        clean, turnover_status = normalize_turnover_fields(clean, source=source)
    else:
        turnover_status = FIELD_MISSING

    for field_name in REQUIRED_FIELDS.get(kind, []):
        col = find_field_column(clean, field_name)
        if field_name == "turnover_rate":
            field_quality[field_name] = turnover_status
            continue
        if col is None:
            field_quality[field_name] = FIELD_MISSING
            continue
        non_null = clean[col].notna()
        if len(clean) == 0 or not bool(non_null.any()):
            field_quality[field_name] = FIELD_MISSING
        else:
            field_quality[field_name] = FIELD_STALE if stale else FIELD_OK

    date_matched: Optional[bool] = None
    if expected_date:
        date_col = find_field_column(clean, "date")
        if date_col is not None and not clean.empty:
            target = _compact_date(expected_date)
            values = clean[date_col].map(_compact_date)
            date_matched = bool((values == target).any())
            if not date_matched:
                warnings.append(f"date_mismatch:{expected_date}")
        elif kind in {"daily_kline", "fund_flow"}:
            date_matched = False
            warnings.append(f"date_missing:{expected_date}")

    ok = len(clean) > 0 and not errors
    quality = SourceQuality(
        source=source,
        ok=ok,
        row_count=int(len(clean)),
        latency=latency,
        warnings=warnings,
        field_quality=field_quality,
        stale_or_not=stale,
        date_matched_or_not=date_matched,
        data_mode=data_mode,
        errors=errors,
    )
    return SourceResult(clean, quality)


def failed_result(
    *,
    source: str,
    kind: str,
    latency: float = 0.0,
    warning: str | None = None,
    error: str | None = None,
    data_mode: str = "live",
) -> SourceResult:
    warnings = [warning] if warning else []
    errors = [error] if error else []
    return assess_dataframe(
        pd.DataFrame(),
        source=source,
        kind=kind,
        latency=latency,
        warnings=warnings,
        errors=errors,
        data_mode=data_mode,
    )


def _compact_date(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        return pd.to_datetime(s).strftime("%Y%m%d")
    except Exception:
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) >= 8:
            return digits[:8]
        try:
            return datetime.strptime(s, "%Y%m%d").strftime("%Y%m%d")
        except Exception:
            return s
