"""WallstreetCN cursor archive tests."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from engine.wscn_news_archive import WscnNewsArchive


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _Session:
    def __init__(self, pages: list[dict]) -> None:
        self.pages = pages
        self.calls = 0
        self.requests: list[dict] = []

    def get(self, *args, **kwargs) -> _Response:
        self.requests.append({"args": args, "kwargs": kwargs})
        page = self.pages[self.calls]
        self.calls += 1
        return _Response(page)


def _ts(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())


def test_cursor_archive_is_exact_deduplicated_and_writes_empty_dates(tmp_path) -> None:
    pages = [
        {"data": {"items": [
            {"id": 3, "display_time": _ts("2026-07-03T10:00:00"), "content_text": "甲公司：中标。"},
            {"id": 2, "display_time": _ts("2026-07-02T09:00:00"), "content_text": "乙公司：签约。"},
        ], "next_cursor": str(_ts("2026-07-02T09:00:00"))}},
        {"data": {"items": [
            {"id": 2, "display_time": _ts("2026-07-02T09:00:00"), "content_text": "重复"},
            {"id": 1, "display_time": _ts("2026-06-30T23:00:00"), "content_text": "边界前"},
        ], "next_cursor": str(_ts("2026-06-30T23:00:00"))}},
    ]
    archive = WscnNewsArchive(tmp_path, session=_Session(pages))
    report = archive.fetch("20260701", "20260703", interval=0, max_pages=5)
    assert report["complete"] is True
    assert report["retained_items"] == 2
    july1 = json.loads((tmp_path / "20260701.json").read_text(encoding="utf-8"))
    july2 = json.loads((tmp_path / "20260702.json").read_text(encoding="utf-8"))
    assert july1["complete"] is True
    assert july1["items"] == []
    assert july2["item_count"] == 1
    assert july2["items"][0]["published_at"].startswith("2026-07-02T09:00:00")


def test_non_decreasing_cursor_marks_archive_incomplete(tmp_path) -> None:
    timestamp = _ts("2026-07-03T10:00:00")
    pages = [{"data": {
        "items": [{"id": 1, "display_time": timestamp, "content_text": "消息"}],
        "next_cursor": str(2**63 - 1),
    }}]
    report = WscnNewsArchive(tmp_path, session=_Session(pages)).fetch(
        "20260701", "20260703", interval=0, max_pages=1
    )
    assert report["complete"] is False
    assert any("cursor" in failure for failure in report["failures"])


def test_start_cursor_is_sent_on_first_request_and_audited(tmp_path) -> None:
    start_cursor = str(_ts("2026-07-02T12:00:00"))
    older_cursor = str(_ts("2026-06-30T23:00:00"))
    session = _Session([{"data": {
        "items": [{
            "id": 1,
            "display_time": _ts("2026-06-30T23:00:00"),
            "content_text": "boundary",
        }],
        "next_cursor": older_cursor,
    }}])
    archive = WscnNewsArchive(tmp_path, session=session)
    report = archive.fetch(
        "20260701",
        "20260701",
        interval=0,
        max_pages=1,
        start_cursor=start_cursor,
    )

    assert session.requests[0]["kwargs"]["params"]["cursor"] == start_cursor
    assert report["start_cursor"] == start_cursor
    day = json.loads((tmp_path / "20260701.json").read_text(encoding="utf-8"))
    assert day["archive"]["start_cursor"] == start_cursor
    assert day["archive"]["last_cursor"] == older_cursor


def test_invalid_start_cursor_fails_closed_before_network_call(tmp_path) -> None:
    session = _Session([])
    archive = WscnNewsArchive(tmp_path, session=session)

    with pytest.raises(ValueError, match="start_cursor must be numeric"):
        archive.fetch("20260701", "20260701", start_cursor="not-a-cursor")

    assert session.calls == 0


def test_retention_boundary_salvages_only_later_complete_dates(tmp_path) -> None:
    pages = [
        {"data": {
            "items": [
                {"id": 3, "display_time": _ts("2026-07-03T10:00:00"), "content_text": "new"},
                {"id": 1, "display_time": _ts("2026-07-01T09:00:00"), "content_text": "oldest"},
            ],
            "next_cursor": str(_ts("2026-07-01T09:00:00")),
        }},
        {"data": {"items": [], "next_cursor": ""}},
    ]
    archive = WscnNewsArchive(tmp_path, session=_Session(pages))
    report = archive.fetch("20260630", "20260703", interval=0, max_pages=5)

    assert report["complete"] is False
    assert report["complete_from"] == "20260702"
    assert report["complete_dates"] == 2
    june30 = json.loads((tmp_path / "20260630.json").read_text(encoding="utf-8"))
    july1 = json.loads((tmp_path / "20260701.json").read_text(encoding="utf-8"))
    july2 = json.loads((tmp_path / "20260702.json").read_text(encoding="utf-8"))
    july3 = json.loads((tmp_path / "20260703.json").read_text(encoding="utf-8"))
    assert june30["complete"] is False
    assert july1["complete"] is False
    assert july2["complete"] is True
    assert july3["complete"] is True
