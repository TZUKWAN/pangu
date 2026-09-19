"""Quote provider bridge: PIT daily bars → PaperBroker-compatible quotes.

READ-only over the PIT archive (engine/data/pit_store.py).  Produces the
exact quote dict PaperBroker expects:
    {open, high, low, close, preclose, is_st}
or None when the symbol has no bar for that date (suspended / not listed /
non-trading day).  Dates are normalized: 'YYYY-MM-DD' and 'YYYYMMDD' both
accepted; the returned dict never leaks future data because PITStore's
daily_panel is strictly asof-guarded.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from engine.data.lookahead import to_compact


def quote_from_pit(pit_store, symbol: str, date) -> Optional[Dict]:
    """One day's bar for `symbol` as a PaperBroker quote dict, or None.

    Suspended / missing bar → None (never fabricate a quote).
    """
    d = to_compact(date)
    panel = pit_store.daily_panel(d, d, [symbol])
    if panel is None or panel.empty:
        return None
    row = panel.iloc[0]

    def _f(col: str) -> float:
        v = row.get(col)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return v if v == v else 0.0  # NaN → 0.0

    preclose = _f("preclose")
    close = _f("close")
    if preclose <= 0 and close <= 0:
        return None
    return {
        "open": _f("open"),
        "high": _f("high"),
        "low": _f("low"),
        "close": close,
        "preclose": preclose,
        "is_st": bool(row.get("is_st", False)),
    }


def quote_provider_factory(db_path: str) -> Callable[[str, str], Optional[Dict]]:
    """Return a PaperBroker-compatible quote_provider(symbol, date).

    The PITStore handle is created lazily on first call (so constructing the
    provider never touches the filesystem) and reused afterwards.
    """
    cache: Dict[str, object] = {}

    def provider(symbol: str, date: str) -> Optional[Dict]:
        store = cache.get("store")
        if store is None:
            from engine.data.pit_store import PITStore  # local import: heavy module

            store = PITStore(db_path)
            cache["store"] = store
        return quote_from_pit(store, symbol, date)

    return provider
