"""Source registry and default provider chains."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

import pandas as pd

from engine.source_quality import SourceResult, assess_dataframe, failed_result, find_field_column
from .base import SourceContext, SourceProvider

logger = logging.getLogger("pangu.sources.registry")


class SourceRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, list[SourceProvider]] = defaultdict(list)
        self._failure_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._failure_lock = threading.Lock()
        self._circuit_threshold = 3

    def register(self, provider: SourceProvider) -> None:
        self._providers[provider.kind].append(provider)

    def providers(self, kind: str) -> list[SourceProvider]:
        return list(self._providers.get(kind, []))

    def fetch(self, kind: str, context: SourceContext) -> SourceResult:
        chain: list[dict[str, Any]] = []
        best_degraded: SourceResult | None = None
        degraded_results: list[SourceResult] = []
        for provider in self.providers(kind):
            if not provider.supports(kind, context.mode):
                continue
            failure_key = (kind, provider.name)
            with self._failure_lock:
                circuit_open = self._failure_counts[failure_key] >= self._circuit_threshold
            if circuit_open:
                skipped = failed_result(
                    source=provider.name,
                    kind=kind,
                    warning=f"circuit_open_after_{self._circuit_threshold}_failures",
                    data_mode=context.mode,
                )
                chain.append(skipped.quality.to_dict())
                continue
            t0 = time.monotonic()
            try:
                result = provider.fetch(context)
            except Exception as exc:  # noqa: BLE001
                latency = time.monotonic() - t0
                logger.debug("%s provider %s failed: %s", kind, provider.name, exc)
                result = failed_result(
                    source=provider.name,
                    kind=kind,
                    latency=latency,
                    error=str(exc),
                    data_mode=context.mode,
                )
            result.quality.latency = result.quality.latency or (time.monotonic() - t0)
            chain.append(result.quality.to_dict())
            if result.quality.ok and not result.data.empty:
                with self._failure_lock:
                    self._failure_counts[failure_key] = 0
                if result.quality.status == "ok":
                    result.chain = chain
                    return result
                if best_degraded is None:
                    best_degraded = result
                degraded_results.append(result)
                if kind == "all_spot" and len(degraded_results) >= 2:
                    merged = self._merge_all_spot(degraded_results[:2], context)
                    merged.chain = chain
                    return merged
                continue
            with self._failure_lock:
                self._failure_counts[failure_key] += 1
        if best_degraded is not None:
            best_degraded.chain = chain
            return best_degraded
        final = assess_dataframe(
            pd.DataFrame(),
            source="unavailable",
            kind=kind,
            warnings=[f"all_{kind}_providers_failed"],
            data_mode=context.mode,
        )
        final.chain = chain
        return final

    @staticmethod
    def _merge_all_spot(results: list[SourceResult], context: SourceContext) -> SourceResult:
        """按股票代码合并两个互补快照，优先保留高优先级源的非空值。"""
        indexed: list[pd.DataFrame] = []
        sources: list[str] = []
        for result in results:
            frame = result.data.copy()
            code_col = find_field_column(frame, "code")
            if code_col is None or frame.empty:
                continue
            frame["_source_merge_code"] = frame[code_col].astype(str).str.strip().str.zfill(6)
            frame = frame.drop_duplicates("_source_merge_code").set_index("_source_merge_code")
            indexed.append(frame)
            sources.append(result.quality.source)
        if not indexed:
            return results[0]
        merged = indexed[0]
        for frame in indexed[1:]:
            merged = merged.combine_first(frame)
        merged = merged.reset_index(drop=True)
        return assess_dataframe(
            merged,
            source="merged:" + "+".join(sources),
            kind="all_spot",
            latency=sum(r.quality.latency for r in results),
            warnings=["merged_complementary_all_spot_sources"],
            data_mode=context.mode,
        )


def build_default_registry(loader: Any | None = None) -> SourceRegistry:
    from .providers.core import (
        AdataDailyKlineProvider,
        AdataFundFlowProvider,
        AdataSpotProvider,
        BaostockDailyKlineProvider,
        BaiduDailyKlineProvider,
        BaiduSpotProvider,
        ExactSnapshotProvider,
        LocalSnapshotProvider,
        MootdxDailyKlineProvider,
        SinaDailyKlineProvider,
        SinaSpotProvider,
        StaleCacheProvider,
        TencentDailyKlineProvider,
        TencentSpotProvider,
        ThsFundFlowProvider,
        ThsSpotProvider,
        UnavailableProvider,
    )

    registry = SourceRegistry()
    # all_spot: snapshot mode is strict; live does real providers first, then fresh local snapshot.
    registry.register(ExactSnapshotProvider("all_spot", modes=("snapshot",)))
    registry.register(ExactSnapshotProvider("all_spot", modes=("diagnostic",)))
    registry.register(StaleCacheProvider("all_spot", modes=("diagnostic",)))
    registry.register(ThsSpotProvider())
    registry.register(TencentSpotProvider())
    registry.register(SinaSpotProvider())
    registry.register(BaiduSpotProvider())
    registry.register(AdataSpotProvider())
    registry.register(LocalSnapshotProvider("all_spot", modes=("live",)))

    # daily_kline: mootdx (TCP, no IP block) + 百度日K带MA 作为优选源。
    registry.register(ExactSnapshotProvider("daily_kline", modes=("snapshot",)))
    registry.register(StaleCacheProvider("daily_kline", modes=("diagnostic",)))
    registry.register(SinaDailyKlineProvider())
    registry.register(TencentDailyKlineProvider())
    registry.register(MootdxDailyKlineProvider())
    registry.register(BaiduDailyKlineProvider())
    registry.register(AdataDailyKlineProvider())
    registry.register(BaostockDailyKlineProvider())
    registry.register(StaleCacheProvider("daily_kline", modes=("live",)))

    # fund_flow: unavailable is explicit final source, not an exception.
    registry.register(ExactSnapshotProvider("fund_flow", modes=("snapshot",)))
    registry.register(ThsFundFlowProvider())
    registry.register(AdataFundFlowProvider())
    registry.register(UnavailableProvider("fund_flow"))
    return registry
