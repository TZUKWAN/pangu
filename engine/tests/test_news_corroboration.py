"""Official announcement corroboration tests."""

from __future__ import annotations

import json

from engine.news_corroboration import OfficialNewsCorroborator


def _write_day(tmp_path, date: str, events: list[dict], *, complete: bool = True) -> None:
    (tmp_path / f"{date}.json").write_text(json.dumps({
        "date": date,
        "complete": complete,
        "events": events,
    }, ensure_ascii=False), encoding="utf-8")


def _event(
    code: str,
    title: str,
    announcement_id: str,
    published_at: str,
) -> dict:
    return {
        "code": code,
        "name": "测试股",
        "title": title,
        "published_at": published_at,
        "announcement_id": announcement_id,
        "adjunct_url": f"https://static.cninfo.com.cn/{announcement_id}.PDF",
    }


def test_same_day_date_only_confirms_fact_but_not_intraday_order(tmp_path) -> None:
    _write_day(tmp_path, "20260710", [
        _event("600001", "2026年半年度业绩预增公告", "1", "2026-07-10T00:00:00+08:00")
    ])
    result = OfficialNewsCorroborator(tmp_path).corroborate(
        code="600001",
        signal_date="20260710",
        signal_event_type="earnings_positive",
        signal_published_at="20260710 19:30",
    )
    assert result.confirmed is True
    assert result.role_confirmed is True
    assert result.status == "confirmed_same_day_date_only"
    assert result.causal_order_known is False
    assert result.time_precision == "date_only"


def test_previous_day_contract_has_known_causal_order(tmp_path) -> None:
    _write_day(tmp_path, "20260709", [
        _event("600002", "关于公司签订重大经营合同的公告", "2", "2026-07-09T00:00:00+08:00")
    ])
    _write_day(tmp_path, "20260710", [])
    result = OfficialNewsCorroborator(tmp_path).corroborate(
        code="600002",
        signal_date="20260710",
        signal_event_type="major_contract",
    )
    assert result.confirmed is True
    assert result.official_event_type == "major_contract"
    assert result.status == "confirmed_previous_date"
    assert result.causal_order_known is True


def test_customer_or_other_stock_does_not_confirm_beneficiary(tmp_path) -> None:
    _write_day(tmp_path, "20260710", [
        _event("600941", "关于项目中标的公告", "3", "2026-07-10T00:00:00+08:00")
    ])
    result = OfficialNewsCorroborator(tmp_path).corroborate(
        code="603019",
        signal_date="20260710",
        signal_event_type="major_contract",
    )
    assert result.confirmed is False
    assert result.status == "no_compatible_official_announcement"


def test_unsupported_event_family_fails_closed(tmp_path) -> None:
    _write_day(tmp_path, "20260710", [])
    result = OfficialNewsCorroborator(tmp_path).corroborate(
        code="300001",
        signal_date="20260710",
        signal_event_type="policy_support",
    )
    assert result.confirmed is False
    assert result.status == "unsupported_event_family"


def test_incomplete_signal_day_archive_blocks_confirmation(tmp_path) -> None:
    _write_day(tmp_path, "20260710", [], complete=False)
    result = OfficialNewsCorroborator(tmp_path).corroborate(
        code="600001",
        signal_date="20260710",
        signal_event_type="earnings_positive",
    )
    assert result.confirmed is False
    assert result.status == "official_archive_missing_or_incomplete"


def test_three_day_old_same_stock_contract_is_stale_and_not_confirmation(tmp_path) -> None:
    _write_day(tmp_path, "20260707", [
        _event("600002", "关于公司签订重大经营合同的公告", "7", "2026-07-07T00:00:00+08:00")
    ])
    _write_day(tmp_path, "20260710", [])
    result = OfficialNewsCorroborator(tmp_path).corroborate(
        code="600002",
        signal_date="20260710",
        signal_event_type="major_contract",
    )
    assert result.confirmed is False


def test_contract_identity_requires_matching_news_and_pdf_amounts() -> None:
    mismatch = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="公司中标印尼项目，约人民币2.23亿元",
        signal_event_type="major_contract",
        announcement_detail={"amounts": ["3.36亿元"]},
    )
    matched = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="公司中标印尼项目，约人民币2.23亿元",
        signal_event_type="major_contract",
        announcement_detail={"amounts": ["22300万元"]},
    )
    assert mismatch.matched is False
    assert mismatch.status == "amount_mismatch"
    assert matched.matched is True
    assert matched.status == "amount_match"


def test_earnings_identity_matches_official_growth_range() -> None:
    matched = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="预计净利润同比增长208.71%-258.88%",
        signal_event_type="earnings_positive",
        announcement_detail={"growth_lower_pct": 208.71, "growth_upper_pct": 258.88},
    )
    mismatch = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="预计净利润同比增长30%-40%",
        signal_event_type="earnings_positive",
        announcement_detail={"growth_lower_pct": 208.71, "growth_upper_pct": 258.88},
    )
    assert matched.matched is True
    assert mismatch.matched is False


def test_earnings_identity_understands_fold_and_multiple_language() -> None:
    fold = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="上半年净利预增最高翻一倍，AI服务器营收同比增超230%",
        signal_event_type="earnings_positive",
        announcement_detail={"growth_lower_pct": 86.0, "growth_upper_pct": 101.0},
    )
    multiple = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="上半年净利预增10.99倍",
        signal_event_type="earnings_positive",
        announcement_detail={"growth_lower_pct": 1099.0, "growth_upper_pct": 1099.0},
    )
    assert fold.matched is True
    assert multiple.matched is True


def test_restructuring_identity_requires_same_subtype() -> None:
    matched = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="公司披露发行股份购买资产暨重大资产重组预案",
        signal_event_type="restructuring",
        announcement_detail={
            "title": "发行股份购买资产暨重大资产重组报告书（草案）",
            "evidence_snippets": ["本次交易构成重大资产重组"],
        },
    )
    mismatch = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="公司控制权发生变更",
        signal_event_type="restructuring",
        announcement_detail={
            "title": "发行股份购买资产暨重大资产重组报告书（草案）",
            "evidence_snippets": [],
        },
    )
    assert matched.matched is True
    assert mismatch.matched is False


def test_product_identity_requires_category_and_product_anchor() -> None:
    matched = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="公司创新药ABC-101获得批准上市",
        signal_event_type="product_breakthrough",
        announcement_detail={
            "title": "关于创新药ABC-101获得批准上市的公告",
            "evidence_snippets": ["ABC-101获得批准上市"],
        },
    )
    wrong_product = OfficialNewsCorroborator.audit_event_identity(
        news_evidence="公司创新药XYZ-9获得批准上市",
        signal_event_type="product_breakthrough",
        announcement_detail={
            "title": "关于创新药ABC-101获得批准上市的公告",
            "evidence_snippets": [],
        },
    )
    assert matched.matched is True
    assert wrong_product.matched is False
