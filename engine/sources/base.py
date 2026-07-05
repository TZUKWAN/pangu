"""Base classes for pluggable data source providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from engine.source_quality import SourceResult


@dataclass
class SourceContext:
    loader: Any
    mode: str = "live"
    data_date: str | None = None
    symbol: str | None = None
    days: int = 60
    adjust: str = "qfq"
    date: str | None = None
    cache_key: str | None = None
    snapshot_dir: Path | None = None

    @property
    def effective_date(self) -> str | None:
        return self.date or self.data_date


class SourceProvider:
    name: str = "provider"
    kind: str = "generic"
    modes: tuple[str, ...] = ("live", "snapshot", "diagnostic")

    def supports(self, kind: str, mode: str) -> bool:
        return self.kind == kind and mode in self.modes

    def fetch(self, context: SourceContext) -> SourceResult:
        raise NotImplementedError


def mode_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(values or ("live", "snapshot", "diagnostic"))
