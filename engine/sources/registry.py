"""Source registry and default provider chains."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import pandas as pd

from engine.source_quality import SourceResult, assess_dataframe, failed_result
from .base import SourceContext, SourceProvider

logger = logging.getLogger("pangu.sources.registry")


class SourceRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, list[SourceProvider]] = defaultdict(list)

    def register(self, provider: SourceProvider) -> None:
        self._providers[provider.kind].append(provider)

    def providers(self, kind: str) -> list[SourceProvider]:
        return list(self._providers.get(kind, []))

    def fetch(self, kind: str, context: SourceContext) -> SourceResult:
        chain: list[dict[str, Any]] = []
        for provider in self.providers(kind):
            if not provider.supports(kind, context.mode):
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
                result.chain = chain
                return result
        final = assess_dataframe(
            pd.DataFrame(),
            source="unavailable",
            kind=kind,
            warnings=[f"all_{kind}_providers_failed"],
            data_mode=context.mode,
        )
        final.chain = chain
        return final


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
