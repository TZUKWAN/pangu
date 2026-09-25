"""Freshness 元数据与查询时刷新（Phase 2 / Task 2.3-2.4）。

每个数据源定义 SLA；Evidence 必须携带 source/fetched_at/published_at/
effective_at/asof/age_seconds/stale/quality/fallback_used——禁止只有 "status": "ok"。

MarketContextRefresher：
- 检查各源缓存 fetched_at 相对 asof 的新鲜度；
- 过期则调用注入的 refresh 函数做**增量**刷新（绝不重建全量档案）；
- 刷新失败 → fallback_used=True + stale=True，并给出 degraded 结论；
  关键源（realtime/daily_kline）失败时上层禁止 BUY（Phase 13 承接）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from engine.decision.asof import AsOfContext, TradingCalendar, _parse
from engine.decision.clock import Clock, now_shanghai

# 每源 SLA（秒）。日 K 的 SLA 语义 = 数据须覆盖最近已完成交易日。
SOURCE_SLA: Dict[str, float] = {
    "realtime_quote": 180.0,        # 实时行情：分钟级
    "news": 600.0,                  # 新闻快讯：分钟级
    "announcements": 86400.0,       # 公告：当天
    "daily_kline": 86400.0,         # 日 K：最近已完成交易日
    "industry_concept": 7 * 86400.0,
    "financials": 90 * 86400.0,     # 最新有效报告期
}
CRITICAL_SOURCES = ("realtime_quote", "daily_kline")


@dataclass
class SourceFreshness:
    source: str
    fetched_at: Optional[str]
    asof: str
    age_seconds: Optional[float]
    stale: bool
    quality: str = "ok"            # ok | degraded | failed
    fallback_used: bool = False
    published_at: Optional[str] = None
    effective_at: Optional[str] = None
    detail: str = ""
    latency_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_critical_failure(self) -> bool:
        return self.source in CRITICAL_SOURCES and (
            self.quality == "failed" or (self.stale and not self.fetched_at))


def _age(fetched_at: Optional[str], asof_dt) -> Optional[float]:
    if not fetched_at:
        return None
    dt = _parse(fetched_at)
    return max(0.0, (asof_dt - dt).total_seconds())


class MarketContextRefresher:
    """查询时上下文刷新器。refreshers: source -> fn(asof_iso) -> dict(meta)。

    fn 返回 {"fetched_at": iso, "published_at"?: iso, "effective_at"?: iso,
             "quality": "ok|degraded|failed", "detail"?: str}；
    未注册 refresher 的源只做新鲜度评估。
    """

    def __init__(self, refreshers: Optional[Dict[str, Callable[[str], dict]]] = None,
                 clock: Optional[Clock] = None,
                 sla_overrides: Optional[Dict[str, float]] = None,
                 last_fetch: Optional[Dict[str, str]] = None,
                 force_refresh: bool = False):
        self._refreshers = dict(refreshers or {})
        self._clock = clock
        self._sla = dict(SOURCE_SLA)
        if sla_overrides:
            self._sla.update(sla_overrides)
        self._last_fetch: Dict[str, str] = dict(last_fetch or {})
        self._force = force_refresh

    def refresh_context(self, ctx: AsOfContext) -> List[SourceFreshness]:
        asof_dt = _parse(ctx.asof_timestamp)
        asof_iso = ctx.asof_timestamp
        out: List[SourceFreshness] = []
        for source, sla in self._sla.items():
            t0 = time.monotonic()
            cached_at = self._last_fetch.get(source)
            age = _age(cached_at, asof_dt)
            need = self._force or age is None or age > sla
            fetched_at = cached_at
            quality, fallback, detail = "ok", False, ""
            published_at = effective_at = None
            if need:
                fn = self._refreshers.get(source)
                if fn is None:
                    # 无刷新器：如实标注不可评估
                    quality = "degraded" if age is None or age > sla else "ok"
                    fallback = True
                    detail = "no refresher registered; evaluated from cache only"
                else:
                    try:
                        meta = fn(asof_iso) or {}
                        fetched_at = meta.get("fetched_at") or _iso_now(self._clock)
                        self._last_fetch[source] = fetched_at
                        quality = meta.get("quality", "ok")
                        detail = meta.get("detail", "")
                        published_at = meta.get("published_at")
                        effective_at = meta.get("effective_at")
                        age = _age(fetched_at, asof_dt)
                    except Exception as e:  # noqa: BLE001 — 刷新失败必须显式降级
                        quality, fallback = "failed", True
                        detail = f"refresh error: {e!r}"[:200]
            latency = time.monotonic() - t0
            stale = bool(age is None or age > sla)
            out.append(SourceFreshness(
                source=source, fetched_at=fetched_at, asof=asof_iso,
                age_seconds=age, stale=stale, quality=quality,
                fallback_used=fallback, published_at=published_at,
                effective_at=effective_at, detail=detail,
                latency_seconds=round(latency, 3)))
        return out

    @staticmethod
    def summarize(items: List[SourceFreshness]) -> dict:
        """汇总结论：data_status + 关键源失败清单（Phase 13 禁 BUY 依据）。"""
        critical_failed = [i.source for i in items
                           if i.source in CRITICAL_SOURCES
                           and (i.quality == "failed" or i.stale)]
        any_failed = [i.source for i in items if i.quality == "failed"]
        status = "ok"
        if critical_failed:
            status = "failed"
        elif any_failed or any(i.stale for i in items):
            status = "degraded"
        return {
            "data_status": status,
            "critical_stale_or_failed": critical_failed,
            "failed_sources": any_failed,
            "sources": [i.to_dict() for i in items],
        }


def _iso_now(clock: Optional[Clock]) -> str:
    dt = clock.now() if clock else now_shanghai()
    return dt.isoformat(timespec="seconds")
