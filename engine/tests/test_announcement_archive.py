"""CNINFO announcement backfill tests without live network."""

from __future__ import annotations

from engine.announcement_archive import CninfoAnnouncementArchive, classify_announcement_title


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def post(self, url, data, headers, timeout):
        self.calls.append(dict(data))
        return _Response(self.pages[int(data["pageNum"])])


def _item(identifier, code, title, timestamp=1783690000000):
    return {
        "announcementId": str(identifier),
        "secCode": code,
        "secName": "测试股",
        "announcementTitle": title,
        "announcementTime": timestamp,
        "adjunctUrl": f"finalpage/2026-07-10/{identifier}.PDF",
    }


def test_title_classification_handles_negative_precedence_and_negation() -> None:
    assert classify_announcement_title("关于收到重大项目中标通知书的公告") == ("positive", "project_win")
    assert classify_announcement_title("关于公司签订重大经营合同的公告") == ("positive", "major_contract")
    assert classify_announcement_title("发行股份购买资产暨重大资产重组报告书") == ("positive", "restructuring")
    assert classify_announcement_title("关于全资子公司药品获得批准上市的公告") == ("positive", "product_breakthrough")
    assert classify_announcement_title("关于取得医疗器械注册证的公告") == ("positive", "product_breakthrough")
    assert classify_announcement_title("关于不构成重大资产重组的说明") == ("neutral", "excluded_context")
    assert classify_announcement_title("关于公司收到立案调查告知书的公告") == ("negative", "investigation")
    assert classify_announcement_title("控股股东未来十二个月不减持承诺公告") == ("neutral", "excluded_context")
    assert classify_announcement_title("限制性股票回购注销完成公告") == ("neutral", "excluded_context")


def test_backfill_paginates_deduplicates_and_writes_empty_dates(tmp_path) -> None:
    pages = {
        1: {"totalAnnouncement": 3, "announcements": [
            _item(1, "000001", "关于收到<em>中标</em>通知书的公告"),
            _item(2, "00700", "港股公告"),
        ]},
        2: {"totalAnnouncement": 3, "announcements": [
            _item(1, "000001", "关于收到中标通知书的公告"),
            _item(3, "600001", "关于项目中标的公告"),
        ]},
    }
    archive = CninfoAnnouncementArchive(
        tmp_path, session=_Session(pages), page_size=2,
        request_interval=0, max_pages_per_query=5,
    )
    result = archive.backfill(
        "20260710", "20260711", queries=["中标通知书"], overwrite=True,
    )
    assert result["complete"] is True
    assert result["events"] == 2
    day = archive.load("20260710")
    assert len(day["events"]) == 2
    assert all("<em>" not in event["title"] for event in day["events"])
    assert archive.load("20260711")["events"] == []


def test_page_cap_marks_archive_incomplete(tmp_path) -> None:
    pages = {1: {"totalAnnouncement": 300, "announcements": [_item(1, "000001", "重大合同公告")]}}
    archive = CninfoAnnouncementArchive(
        tmp_path, session=_Session(pages), request_interval=0, max_pages_per_query=1,
    )
    result = archive.backfill("20260710", "20260710", queries=["重大合同"], overwrite=True)
    assert result["complete"] is False
    assert "exceed cap" in result["query_status"]["重大合同"]["error"]


def test_expected_empty_page_is_retried_before_marking_incomplete(tmp_path) -> None:
    class FlakySession:
        def __init__(self) -> None:
            self.page2_calls = 0

        def post(self, url, data, headers, timeout):
            page = int(data["pageNum"])
            if page == 1:
                return _Response({
                    "totalAnnouncement": 2,
                    "announcements": [_item(1, "000001", "重大合同公告")],
                })
            self.page2_calls += 1
            if self.page2_calls == 1:
                return _Response({"totalAnnouncement": 2, "announcements": []})
            return _Response({
                "totalAnnouncement": 2,
                "announcements": [_item(2, "600001", "重大合同公告")],
            })

    session = FlakySession()
    archive = CninfoAnnouncementArchive(
        tmp_path,
        session=session,
        page_size=1,
        request_interval=0,
        max_pages_per_query=5,
    )
    result = archive.backfill(
        "20260710", "20260710", queries=["重大合同"], overwrite=True
    )
    assert result["complete"] is True
    assert result["events"] == 2
    assert session.page2_calls == 2


def test_page_one_uses_highest_total_across_transient_zero_probes(tmp_path) -> None:
    class UnstablePageOneSession:
        def __init__(self) -> None:
            self.page1_calls = 0

        def post(self, url, data, headers, timeout):
            page = int(data["pageNum"])
            if page == 1:
                self.page1_calls += 1
                if self.page1_calls in (1, 3):
                    return _Response({"totalAnnouncement": 0, "announcements": []})
                return _Response({
                    "totalAnnouncement": 2,
                    "announcements": [_item(1, "000001", "重大合同公告")],
                })
            return _Response({
                "totalAnnouncement": 2,
                "announcements": [_item(2, "600001", "重大合同公告")],
            })

    session = UnstablePageOneSession()
    archive = CninfoAnnouncementArchive(
        tmp_path, session=session, page_size=1, request_interval=0, max_pages_per_query=5
    )
    result = archive.backfill(
        "20260710", "20260710", queries=["重大合同"], overwrite=True
    )
    assert result["complete"] is True
    assert result["events"] == 2
    assert result["query_status"]["重大合同"]["total_raw"] == 2
    assert session.page1_calls == 3


def test_redundant_page_sizes_recover_repeated_primary_page(tmp_path) -> None:
    items = [
        _item(index, "000001", "重大合同公告")
        for index in range(1, 32)
    ]

    class RecoverableSession:
        def post(self, url, data, headers, timeout):
            page = int(data["pageNum"])
            size = int(data["pageSize"])
            if size == 30:
                rows = items[:30]
            else:
                start = (page - 1) * size
                rows = items[start:start + size]
            return _Response({
                "totalAnnouncement": len(items),
                "announcements": rows,
            })

    archive = CninfoAnnouncementArchive(
        tmp_path,
        session=RecoverableSession(),
        request_interval=0,
        max_pages_per_query=10,
    )
    result = archive.backfill(
        "20260710", "20260710", queries=["重大合同"], overwrite=True
    )
    status = result["query_status"]["重大合同"]
    assert result["complete"] is True
    assert result["events"] == 31
    assert status["raw_unique_ids"] == 31
    assert status["recovery_requests"] > 0
