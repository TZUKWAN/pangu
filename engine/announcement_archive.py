"""Official CNINFO announcement-event backfill with resumable exact-date archives."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import requests


CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
DEFAULT_EVENT_QUERIES = (
    "业绩预增",
    "扭亏为盈",
    "重大合同",
    "中标通知书",
    "项目中标",
    "回购公司股份",
    "增持计划",
    "业绩预亏",
    "预计亏损",
    "减持计划",
    "立案调查",
    "合同终止",
    "退市风险",
    "行政处罚",
    "重大资产重组",
    "发行股份购买资产",
    "资产注入",
    "控制权变更",
    "拟收购",
    "获得批准上市",
    "获批上市",
    "取得注册证",
    "技术突破",
    "实现量产",
)
_TAG_RE = re.compile(r"<[^>]+>")
_A_SHARE_CODE_RE = re.compile(r"^(?:00|30|60|68|83|87|92)\d{4}$")


@dataclass(frozen=True)
class AnnouncementEvent:
    code: str
    name: str
    title: str
    published_at: str
    announcement_id: str
    adjunct_url: str
    matched_query: str
    polarity: str
    event_type: str

    @property
    def publish_date(self) -> str:
        return self.published_at[:10].replace("-", "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "title": self.title,
            "published_at": self.published_at,
            "announcement_id": self.announcement_id,
            "adjunct_url": self.adjunct_url,
            "matched_query": self.matched_query,
            "polarity": self.polarity,
            "event_type": self.event_type,
            "source": "cninfo",
        }


class CninfoAnnouncementArchive:
    """Download a focused event corpus instead of scraping all disclosure noise."""

    def __init__(
        self,
        root: str | Path = "data/announcement_archive",
        *,
        session: requests.Session | None = None,
        page_size: int = 30,
        max_pages_per_query: int = 60,
        request_interval: float = 0.20,
        timeout: float = 20.0,
    ) -> None:
        self.root = Path(root)
        self.session = session or requests.Session()
        self.page_size = min(30, max(1, int(page_size)))
        self.max_pages_per_query = max(1, int(max_pages_per_query))
        self.request_interval = max(0.0, float(request_interval))
        self.timeout = max(1.0, float(timeout))

    def backfill(
        self,
        start: str,
        end: str,
        *,
        queries: Iterable[str] = DEFAULT_EVENT_QUERIES,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        start_date = self._parse_date(start)
        end_date = self._parse_date(end)
        if start_date > end_date:
            raise ValueError("start date must not be after end date")
        query_list = [str(query).strip() for query in queries if str(query).strip()]
        if not query_list:
            raise ValueError("at least one event query is required")

        events_by_date: dict[str, dict[str, AnnouncementEvent]] = {}
        query_status: dict[str, dict[str, Any]] = {}
        for query in query_list:
            events, status = self._fetch_query(query, start_date, end_date)
            query_status[query] = status
            for event in events:
                events_by_date.setdefault(event.publish_date, {})[event.announcement_id] = event

        complete = all(status.get("complete") for status in query_status.values())
        written = 0
        skipped = 0
        current = start_date
        while current <= end_date:
            date_key = current.strftime("%Y%m%d")
            path = self.root / f"{date_key}.json"
            if path.exists() and not overwrite:
                skipped += 1
            else:
                payload = {
                    "schema_version": 1,
                    "date": date_key,
                    "source": "cninfo",
                    "source_url": CNINFO_QUERY_URL,
                    "fetched_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                    "complete": complete,
                    "queries": query_status,
                    "events": [
                        event.to_dict()
                        for event in sorted(
                            events_by_date.get(date_key, {}).values(),
                            key=lambda item: (item.published_at, item.code, item.announcement_id),
                        )
                    ],
                }
                self._save(path, payload)
                written += 1
            current += timedelta(days=1)
        return {
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "queries": len(query_list),
            "events": sum(len(events) for events in events_by_date.values()),
            "written_dates": written,
            "skipped_dates": skipped,
            "complete": complete,
            "query_status": query_status,
        }

    def load(self, date: str) -> dict[str, Any] | None:
        path = self.root / f"{str(date).replace('-', '')}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _fetch_query(
        self,
        query: str,
        start: date_type,
        end: date_type,
    ) -> tuple[list[AnnouncementEvent], dict[str, Any]]:
        events: dict[str, AnnouncementEvent] = {}
        raw_items: dict[str, dict[str, Any]] = {}
        total = 0
        pages_needed = 1
        pages_fetched = 0
        error = ""
        for page in range(1, self.max_pages_per_query + 1):
            data: dict[str, Any] = {}
            rows: list[dict[str, Any]] = []
            page_error = ""
            best_page_one: dict[str, Any] = {}
            best_page_one_rank = (-1, -1)
            for attempt in range(1, 4):
                try:
                    candidate = (
                        self._request_page(query, start, end, page)
                        if attempt == 1
                        else self._request_page_isolated(query, start, end, page)
                    )
                    candidate_rows = list(candidate.get("announcements") or [])
                    candidate_total = int(candidate.get("totalAnnouncement") or 0)
                    if page == 1:
                        rank = (candidate_total, len(candidate_rows))
                        if rank > best_page_one_rank:
                            best_page_one = candidate
                            best_page_one_rank = rank
                        page_error = ""
                        if attempt < 3:
                            time.sleep(max(0.1, self.request_interval))
                        continue
                    if candidate_rows or candidate_total == 0:
                        data = candidate
                        rows = candidate_rows
                        page_error = ""
                        break
                    page_error = (
                        f"page {page} returned empty announcements while total={candidate_total}"
                    )
                except Exception as exc:  # noqa: BLE001
                    page_error = str(exc)
                if attempt < 3:
                    time.sleep(max(0.1, self.request_interval))
            if page == 1 and best_page_one:
                data = best_page_one
                rows = list(data.get("announcements") or [])
                best_total = int(data.get("totalAnnouncement") or 0)
                page_error = (
                    f"page 1 returned total={best_total} but no announcements"
                    if best_total > 0 and not rows else ""
                )
            if page_error:
                error = page_error
                break
            if page == 1:
                total = int(data.get("totalAnnouncement") or len(rows))
                pages_needed = max(1, math.ceil(total / self.page_size))
            pages_fetched += 1
            new_ids = 0
            for item in rows:
                raw_id = str(item.get("announcementId") or "").strip()
                if raw_id and raw_id not in raw_items:
                    new_ids += 1
                    raw_items[raw_id] = item
                event = self._parse_event(item, query)
                if event is None or not (start.strftime("%Y%m%d") <= event.publish_date <= end.strftime("%Y%m%d")):
                    continue
                events[event.announcement_id] = event
            if page >= pages_needed:
                break
            if new_ids == 0:
                error = f"page {page} repeated without new announcement ids"
                break
            if self.request_interval:
                time.sleep(self.request_interval)
        complete = bool(
            not error
            and pages_fetched >= pages_needed
            and pages_needed <= self.max_pages_per_query
            and (total == 0 or len(raw_items) >= total)
        )
        if pages_needed > self.max_pages_per_query and not error:
            error = f"required pages {pages_needed} exceed cap {self.max_pages_per_query}"
        elif not complete and not error and total > 0:
            error = f"raw unique ids {len(raw_items)} below reported total {total}"
        recovery_requests = 0
        if not complete and total > 0:
            raw_items, total, recovery_requests = self._recover_query_pages(
                query, start, end, raw_items, total
            )
            if len(raw_items) >= total:
                events = {}
                for item in raw_items.values():
                    event = self._parse_event(item, query)
                    if event is None or not (
                        start.strftime("%Y%m%d")
                        <= event.publish_date
                        <= end.strftime("%Y%m%d")
                    ):
                        continue
                    events[event.announcement_id] = event
                complete = True
                error = ""
        return list(events.values()), {
            "total_raw": total,
            "raw_unique_ids": len(raw_items),
            "pages_needed": pages_needed,
            "pages_fetched": pages_fetched,
            "recovery_requests": recovery_requests,
            "a_share_events": len(events),
            "complete": complete,
            "error": error,
        }

    def _recover_query_pages(
        self,
        query: str,
        start: date_type,
        end: date_type,
        raw_items: dict[str, dict[str, Any]],
        total: int,
    ) -> tuple[dict[str, dict[str, Any]], int, int]:
        requests_made = 0
        max_total = int(total)
        for _round in range(2):
            for page_size in (20, 15, 10):
                pages = min(
                    self.max_pages_per_query,
                    max(1, math.ceil(max_total / page_size)),
                )
                for page in range(1, pages + 1):
                    try:
                        data = self._request_page_isolated(
                            query, start, end, page, page_size=page_size
                        )
                    except Exception:  # noqa: BLE001
                        data = {}
                    requests_made += 1
                    max_total = max(
                        max_total, int(data.get("totalAnnouncement") or 0)
                    )
                    for item in data.get("announcements") or []:
                        identifier = str(item.get("announcementId") or "").strip()
                        if identifier:
                            raw_items[identifier] = item
                    if max_total > 0 and len(raw_items) >= max_total:
                        return raw_items, max_total, requests_made
                    time.sleep(max(0.15, self.request_interval))
        return raw_items, max_total, requests_made

    def _request_page(
        self,
        query: str,
        start: date_type,
        end: date_type,
        page: int,
        *,
        page_size: int | None = None,
        session: Any | None = None,
    ) -> dict[str, Any]:
        effective_page_size = min(30, max(1, int(page_size or self.page_size)))
        payload = {
            "pageNum": str(page),
            "pageSize": str(effective_page_size),
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": "",
            "searchkey": query,
            "secid": "",
            "category": "",
            "trade": "",
            "seDate": f"{start:%Y-%m-%d}~{end:%Y-%m-%d}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        response = (session or self.session).post(
            CNINFO_QUERY_URL,
            data=payload,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": "https://www.cninfo.com.cn/new/disclosure",
                "Origin": "https://www.cninfo.com.cn",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("CNINFO returned a non-object payload")
        return data

    def _request_page_isolated(
        self,
        query: str,
        start: date_type,
        end: date_type,
        page: int,
        *,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(self.session, requests.Session):
            with requests.Session() as session:
                return self._request_page(
                    query,
                    start,
                    end,
                    page,
                    page_size=page_size,
                    session=session,
                )
        return self._request_page(
            query, start, end, page, page_size=page_size
        )

    @staticmethod
    def _parse_event(item: dict[str, Any], query: str) -> AnnouncementEvent | None:
        code = str(item.get("secCode") or "").strip()
        if not _A_SHARE_CODE_RE.fullmatch(code):
            return None
        announcement_id = str(item.get("announcementId") or "").strip()
        if not announcement_id:
            return None
        title = html.unescape(_TAG_RE.sub("", str(item.get("announcementTitle") or ""))).strip()
        if not title:
            return None
        timestamp = item.get("announcementTime")
        try:
            published = datetime.fromtimestamp(
                float(timestamp) / 1000.0,
                tz=ZoneInfo("Asia/Shanghai"),
            ).isoformat(timespec="seconds")
        except (TypeError, ValueError, OSError):
            return None
        polarity, event_type = classify_announcement_title(title)
        adjunct = str(item.get("adjunctUrl") or "")
        if adjunct and not adjunct.startswith("http"):
            adjunct = f"https://static.cninfo.com.cn/{adjunct.lstrip('/')}"
        return AnnouncementEvent(
            code=code,
            name=str(item.get("secName") or code),
            title=title,
            published_at=published,
            announcement_id=announcement_id,
            adjunct_url=adjunct,
            matched_query=query,
            polarity=polarity,
            event_type=event_type,
        )

    def _save(self, path: Path, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    @staticmethod
    def _parse_date(value: str) -> date_type:
        normalized = str(value).replace("-", "")
        return datetime.strptime(normalized, "%Y%m%d").date()


def classify_announcement_title(title: str) -> tuple[str, str]:
    text = str(title)
    neutral_phrases = ("不减持", "终止减持计划", "减持计划期限届满", "回购注销限制性股票", "限制性股票回购注销", "不构成重大资产重组")
    if any(phrase in text for phrase in neutral_phrases):
        return "neutral", "excluded_context"
    negative_rules = (
        (("立案调查", "立案告知"), "investigation"),
        (("退市风险", "终止上市风险", "可能被终止上市"), "delisting_risk"),
        (("行政处罚", "处罚决定"), "regulatory_penalty"),
        (("业绩预亏", "预计亏损", "由盈转亏"), "performance_loss"),
        (("减持计划", "拟减持"), "shareholder_reduction"),
        (("合同终止", "终止合同", "项目终止"), "event_terminated"),
    )
    for keywords, event_type in negative_rules:
        if any(keyword in text for keyword in keywords):
            return "negative", event_type
    if "签订" in text and "合同" in text:
        return "positive", "major_contract"
    if any(keyword in text for keyword in (
        "发行股份购买资产", "资产注入", "控制权变更", "重大资产重组",
    )) or ("收购" in text and any(keyword in text for keyword in ("预案", "草案", "报告书", "方案"))):
        return "positive", "restructuring"
    if any(keyword in text for keyword in (
        "获得批准上市", "获批上市", "取得注册证", "获得注册证",
        "取得注册批件", "技术突破", "实现量产", "产品量产",
    )) or (
        "注册证" in text and any(marker in text for marker in ("取得", "获得"))
    ):
        return "positive", "product_breakthrough"
    positive_rules = (
        (("业绩预增", "扭亏为盈"), "performance_growth"),
        (("重大合同", "签订合同"), "major_contract"),
        (("中标通知书", "项目中标", "收到中标"), "project_win"),
        (("回购公司股份", "股份回购方案"), "share_buyback"),
        (("增持计划", "拟增持"), "shareholder_increase"),
    )
    for keywords, event_type in positive_rules:
        if any(keyword in text for keyword in keywords):
            return "positive", event_type
    return "neutral", "other"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill focused official CNINFO announcement events")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--output", default="data/announcement_archive")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--interval", type=float, default=0.20)
    args = parser.parse_args(argv)
    archive = CninfoAnnouncementArchive(args.output, request_interval=args.interval)
    result = archive.backfill(args.start, args.end, overwrite=args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
