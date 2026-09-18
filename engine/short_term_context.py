"""Exact-date causal context archives for short-term recommendation replay."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .data_loader import safe_float


_EXACT_ARCHIVE_MODES = {"exact_date", "fresh_cache", "persisted_exact_date"}


class ShortTermContextArchive:
    """Persist and load only information observable on each report date."""

    def __init__(self, root: str | Path = "data/short_term_context") -> None:
        self.root = Path(root)

    def save_pipeline_result(self, data: Mapping[str, Any]) -> dict[str, Any]:
        date = str(data.get("date") or "").replace("-", "")
        if len(date) != 8 or not date.isdigit():
            raise ValueError("short-term context date must be YYYYMMDD")
        news = data.get("news") or {}
        archive_state = (news.get("source_state") or {}).get("archive") or {}
        exact_news = bool(
            str(news.get("date") or "").replace("-", "") == date
            and archive_state.get("status") == "ok"
            and archive_state.get("mode") in _EXACT_ARCHIVE_MODES
        )
        market_phase = ((data.get("market_modules") or {}).get("market_phase") or {})
        sentiment = data.get("sentiment") or {}
        board_state = {
            str(board.get("name") or ""): {
                "pct": safe_float(board.get("pct"), 0.0),
                "score": safe_float(board.get("score"), 0.0),
            }
            for board in (data.get("boards") or [])
            if board.get("name")
        }
        hot_themes = {
            str(item[0]) for item in (news.get("hot_themes") or [])
            if isinstance(item, (list, tuple)) and item
        }
        evidence_map = data.get("candidate_evidence") or {}
        items: dict[str, dict[str, Any]] = {}
        for bucket in ("candidates", "watchlist", "rejected", "final_recommendations"):
            for item in data.get(bucket) or []:
                code = str(item.get("code") or "").strip().zfill(6)
                if not code or not code.isdigit():
                    continue
                merged = dict(items.get(code) or {})
                merged.update(item)
                items[code] = merged

        stock_context: dict[str, dict[str, Any]] = {}
        for code, item in items.items():
            evidence = item.get("candidate_evidence") or evidence_map.get(code) or {}
            strategy = evidence.get("strategy") or {}
            theme = str(
                strategy.get("theme") or strategy.get("board")
                or item.get("theme") or item.get("board") or ""
            )
            news_evidence = item.get("news_evidence") or evidence.get("news_evidence") or {}
            board = board_state.get(theme)
            theme_known = bool(theme and (board is not None or theme in hot_themes))
            theme_invalidated = bool(
                theme_known and board is not None
                and (safe_float(board.get("pct"), 0.0) <= -2.0 or safe_float(board.get("score"), 0.0) < 0)
            )
            stock_context[code] = {
                "theme": theme,
                "news_scanned": bool(news_evidence),
                "news_evidence": news_evidence,
                "theme_status_known": theme_known,
                "theme_invalidated": theme_invalidated,
            }

        payload = {
            "schema_version": 1,
            "date": date,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_quality": str(data.get("data_quality") or "unknown"),
            "exact_news": exact_news,
            "news_archive": archive_state,
            "market_phase": str(market_phase.get("market_phase") or ""),
            "temperature": safe_float(sentiment.get("temperature"), 0.0),
            "board_state": board_state,
            "hot_themes": sorted(hot_themes),
            "stock_context": stock_context,
            "complete_global_context": bool(
                data.get("data_quality") == "ok"
                and exact_news
                and market_phase.get("market_phase")
                and safe_float(sentiment.get("temperature"), 0.0) > 0
            ),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{date}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
        return {"path": str(path), "date": date, "stock_count": len(stock_context), "complete": payload["complete_global_context"]}

    def load(self, date: str) -> dict[str, Any] | None:
        normalized = str(date).replace("-", "")
        path = self.root / f"{normalized}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or str(data.get("date") or "") != normalized:
                return None
            return data
        except Exception:
            return None

    def build_replay_context(
        self,
        signal_date: str,
        code: str,
        kline: pd.DataFrame,
        evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Build per-day context without substituting a nearby or current date."""
        dates = self._future_dates(kline, signal_date)
        entry_temperature = safe_float((evidence or {}).get("entry_temperature"), 0.0)
        contexts: dict[str, dict[str, Any]] = {}
        for date in dates:
            daily = self.load(date)
            if not daily or not daily.get("complete_global_context"):
                continue
            stock = (daily.get("stock_context") or {}).get(str(code).zfill(6)) or {}
            context: dict[str, Any] = {
                "market_phase": daily.get("market_phase"),
                "current_temperature": daily.get("temperature"),
                "entry_temperature": entry_temperature,
                "market_context_exact": True,
            }
            # Absence is not neutral: omit the field unless this stock was explicitly scanned.
            if stock.get("news_scanned"):
                context["news_evidence"] = stock.get("news_evidence") or {}
                context["news_context_exact"] = True
            if stock.get("theme_status_known"):
                context["theme_invalidated"] = bool(stock.get("theme_invalidated"))
                context["theme_status_known"] = True
            contexts[date] = context
        return contexts

    @staticmethod
    def _future_dates(kline: pd.DataFrame, signal_date: str) -> list[str]:
        if kline is None or len(kline) == 0:
            return []
        date_col = next(
            (name for name in ("日期", "date", "trade_date") if name in kline.columns),
            None,
        )
        if date_col is None:
            return []
        dates = (
            kline[date_col].astype(str).str.replace("-", "", regex=False).str[:8]
        )
        return sorted({date for date in dates if date > str(signal_date)})
