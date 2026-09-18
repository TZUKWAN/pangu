"""Exact-time archive for WallstreetCN's public A-share 7x24 live feed."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import requests


_SHANGHAI = ZoneInfo("Asia/Shanghai")


class WscnNewsArchive:
    ENDPOINT = "https://api-one-wscn.awtmt.com/apiv1/content/lives"

    def __init__(
        self,
        root: str | Path = "data/wscn_news_archive",
        *,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch(
        self,
        start: str,
        end: str,
        *,
        interval: float = 0.2,
        max_pages: int = 500,
        overwrite: bool = False,
        start_cursor: str = "",
    ) -> dict[str, Any]:
        start_dt = datetime.strptime(start, "%Y%m%d").replace(tzinfo=_SHANGHAI)
        end_dt = datetime.strptime(end, "%Y%m%d").replace(
            hour=23, minute=59, second=59, tzinfo=_SHANGHAI
        )
        if start_dt > end_dt:
            raise ValueError("start must be <= end")
        cursor = str(start_cursor or "")
        try:
            previous_cursor = int(cursor) if cursor else 2**63 - 1
        except ValueError as exc:
            raise ValueError("start_cursor must be numeric") from exc
        pages = 0
        seen_ids: set[str] = set()
        by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        reached_start = False
        failures: list[str] = []
        oldest_seen: datetime | None = None

        while pages < max_pages:
            params: dict[str, Any] = {
                "channel": "a-stock-channel",
                "limit": 100,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                response = self.session.get(
                    self.ENDPOINT,
                    params=params,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # noqa: BLE001
                failures.append(str(exc))
                break
            data = payload.get("data") or {}
            items = list(data.get("items") or [])
            if not items:
                failures.append("empty page before reaching start boundary")
                break
            pages += 1
            timestamps = [int(item.get("display_time") or 0) for item in items]
            if any(value <= 0 for value in timestamps):
                failures.append(f"page {pages} contains invalid display_time")
                break
            if timestamps != sorted(timestamps, reverse=True):
                failures.append(f"page {pages} is not reverse chronological")
                break
            for item in items:
                identifier = str(item.get("id") or "")
                if not identifier or identifier in seen_ids:
                    continue
                seen_ids.add(identifier)
                published = datetime.fromtimestamp(int(item["display_time"]), tz=_SHANGHAI)
                if published < start_dt:
                    reached_start = True
                    continue
                if published > end_dt:
                    continue
                date = published.strftime("%Y%m%d")
                by_date[date].append(self._normalize_item(item, published))
            next_cursor = str(data.get("next_cursor") or "")
            try:
                next_value = int(next_cursor)
            except (TypeError, ValueError):
                failures.append(f"page {pages} missing numeric next_cursor")
                break
            if next_value >= previous_cursor:
                failures.append(f"cursor did not move backward on page {pages}")
                break
            previous_cursor = next_value
            cursor = next_cursor
            oldest = datetime.fromtimestamp(min(timestamps), tz=_SHANGHAI)
            oldest_seen = oldest if oldest_seen is None else min(oldest_seen, oldest)
            print(
                f"wscn page={pages} oldest={oldest.isoformat()} retained={sum(map(len, by_date.values()))}",
                flush=True,
            )
            if oldest < start_dt:
                reached_start = True
                break
            if interval > 0:
                time.sleep(interval)

        complete = bool(reached_start and not failures)
        complete_from: datetime | None = None
        if (
            not complete
            and failures == ["empty page before reaching start boundary"]
            and oldest_seen is not None
        ):
            # The cursor chain is exact down to the oldest returned item.  The
            # calendar day containing that item may be truncated by provider
            # retention, but every later calendar day is still complete.
            complete_from = (oldest_seen + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        current = start_dt.date()
        complete_dates = 0
        while current <= end_dt.date():
            date = current.strftime("%Y%m%d")
            existing = self.root / f"{date}.json"
            date_complete = bool(
                complete
                or (complete_from is not None and current >= complete_from.date())
            )
            if date_complete:
                complete_dates += 1
            if overwrite or not existing.exists():
                items = sorted(
                    by_date.get(date, []), key=lambda item: item["display_timestamp"]
                )
                self._write_json(existing, {
                    "date": date,
                    "complete": date_complete,
                    "source": "wallstreetcn_a_stock_7x24",
                    "items": items,
                    "item_count": len(items),
                    "archive": {
                        "mode": "exact_published_time_cursor_archive",
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "pages": pages,
                        "range_start": start,
                        "range_end": end,
                        "start_cursor": str(start_cursor or ""),
                        "last_cursor": cursor,
                        "requested_range_complete": complete,
                        "complete_from": (
                            complete_from.strftime("%Y%m%d") if complete_from else ""
                        ),
                        "failures": failures,
                    },
                })
            current += timedelta(days=1)
        manifest = {
            "start": start,
            "end": end,
            "complete": complete,
            "pages": pages,
            "unique_items_seen": len(seen_ids),
            "retained_items": sum(map(len, by_date.values())),
            "dates_with_items": len(by_date),
            "failures": failures,
            "last_cursor": cursor,
            "start_cursor": str(start_cursor or ""),
            "complete_from": complete_from.strftime("%Y%m%d") if complete_from else "",
            "complete_dates": complete_dates,
        }
        self._write_json(self.root / f"manifest_{start}_{end}.json", manifest)
        return manifest

    @staticmethod
    def _normalize_item(item: Mapping[str, Any], published: datetime) -> dict[str, Any]:
        content = str(item.get("content_text") or item.get("title") or "").strip()
        themes = [
            {
                "id": str(theme.get("id") or ""),
                "title": str(theme.get("title") or ""),
                "key": str(theme.get("key") or ""),
            }
            for theme in (item.get("related_themes") or [])
            if isinstance(theme, Mapping)
        ]
        return {
            "id": str(item.get("id") or ""),
            "display_timestamp": int(item.get("display_time") or 0),
            "published_at": published.isoformat(),
            "title": str(item.get("title") or "")[:300],
            "content": content[:1200],
            "important": bool(item.get("score", 0) and float(item.get("score") or 0) >= 2),
            "themes": themes,
            "uri": str(item.get("uri") or ""),
            "channels": list(item.get("channels") or []),
        }

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="归档华尔街见闻A股7x24历史快讯")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--root", default="data/wscn_news_archive")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-cursor", default="")
    args = parser.parse_args()
    report = WscnNewsArchive(args.root).fetch(
        args.start,
        args.end,
        interval=args.interval,
        max_pages=args.max_pages,
        overwrite=args.overwrite,
        start_cursor=args.start_cursor,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
