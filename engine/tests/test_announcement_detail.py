"""Announcement PDF substance analysis tests."""

from __future__ import annotations

from engine.announcement_detail import AnnouncementDetailAnalyzer


def _event(event_type: str, name: str = "测试股份") -> dict:
    return {
        "announcement_id": "1", "code": "000001", "name": name,
        "event_type": event_type, "title": "测试公告",
    }


def test_purchase_contract_is_not_treated_as_sales_catalyst(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    detail = analyzer.analyze_text(
        _event("major_contract"),
        "公司与供应商签订采购合同，承诺采购金额18.6080亿美元。"
        "采购量占比较小，市场价格下跌可能令公司承受较大损失，敬请注意投资风险。",
    )
    assert detail.contract_direction == "purchase"
    assert detail.tradable_catalyst is False
    assert detail.catalyst_strength < 60


def test_sales_contract_with_amount_is_tradable_catalyst(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    detail = analyzer.analyze_text(
        _event("major_contract"),
        "公司与客户签订销售合同，向客户提供算力设备，合同金额为8.5亿元，"
        "占公司最近一个会计年度经审计营业收入的35%。",
    )
    assert detail.contract_direction == "sales_or_project_revenue"
    assert detail.tradable_catalyst is True
    assert detail.catalyst_strength >= 75


def test_performance_growth_requires_quantified_lower_bound(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    strong = analyzer.analyze_text(
        _event("performance_growth"),
        "预计归属于上市公司股东的净利润同比增加80%至110%。",
    )
    weak = analyzer.analyze_text(
        _event("performance_growth"),
        "公司预计本期业绩同比增长，具体数据以正式报告为准。",
    )
    assert strong.growth_lower_pct == 80.0
    assert strong.tradable_catalyst is True
    assert weak.tradable_catalyst is False


def test_st_stock_is_blocked_even_with_growth(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    detail = analyzer.analyze_text(
        _event("performance_growth", "*ST测试"),
        "预计归属于上市公司股东的净利润同比增加200%至250%。",
    )
    assert detail.tradable_catalyst is False
    assert any("ST" in risk for risk in detail.risk_flags)


def test_listing_rule_50_percent_is_not_mistaken_for_actual_growth(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    detail = analyzer.analyze_text(
        _event("performance_growth"),
        "本公告适用情形为实现盈利且净利润上升50%以上。"
        "公司预计归属于上市公司股东的净利润为1亿元至1.2亿元，同比增长20%至30%。",
    )
    assert detail.growth_lower_pct == 20.0
    assert detail.growth_upper_pct == 30.0
    assert detail.tradable_catalyst is False
    assert "归属于上市公司股东的净利润" in detail.evidence_snippets[0]


def test_attributable_profit_range_is_extracted_instead_of_deducted_profit(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    detail = analyzer.analyze_text(
        _event("performance_growth"),
        "预计归属于母公司所有者的净利润为12亿元到13亿元，同比增长69.11%到82.53%。"
        "预计归属于母公司所有者的扣除非经常性损益后的净利润同比增长625%到987%。",
    )
    assert detail.growth_lower_pct == 69.11
    assert detail.growth_upper_pct == 82.53
    assert detail.tradable_catalyst is True


def test_amount_parser_keeps_thousands_separator(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    detail = analyzer.analyze_text(
        _event("major_contract"),
        "公司收到中标通知书，项目金额为16,600.00万元。",
    )
    assert "16,600.00万元" in detail.amounts


def test_formal_buyback_plan_with_cancellation_and_amount_is_tradable(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    event = _event("share_buyback")
    event["title"] = "关于回购公司股份并用于注销的方案公告"
    detail = analyzer.analyze_text(
        event,
        "公司拟使用自有资金1.5亿元回购股份，回购股份将依法注销并减少注册资本。",
    )
    assert detail.event_stage == "formal_plan"
    assert detail.capital_purpose == "cancellation"
    assert detail.tradable_catalyst is True
    assert detail.catalyst_strength >= 80


def test_buyback_completion_and_progress_are_not_new_catalysts(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    completed_event = _event("share_buyback")
    completed_event["title"] = "关于回购公司股份完成暨股份变动的公告"
    completed = analyzer.analyze_text(
        completed_event,
        "公司回购实施完成，累计支付1亿元。",
    )
    progress_event = _event("share_buyback")
    progress_event["title"] = "关于首次回购公司股份的公告"
    progress = analyzer.analyze_text(
        progress_event,
        "公司根据既有回购方案首次回购股份，支付1000万元。",
    )
    assert completed.event_stage == "completed"
    assert completed.tradable_catalyst is False
    assert progress.event_stage == "execution_progress"
    assert progress.tradable_catalyst is False


def test_buyback_progress_title_wins_over_body_restatement_of_plan(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    event = _event("share_buyback")
    event["title"] = "关于回购公司股份的进展公告"
    detail = analyzer.analyze_text(
        event,
        "原回购股份方案拟使用1亿元并用于注销，本次公告披露回购进展情况。",
    )
    assert detail.event_stage == "execution_progress"
    assert detail.tradable_catalyst is False


def test_new_buyback_plan_title_wins_over_historical_completion_text(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    event = _event("share_buyback")
    event["title"] = "关于回购公司股份方案暨取得回购专项贷款承诺书的公告"
    detail = analyzer.analyze_text(
        event,
        "公司上一期回购实施完成，本次新回购股份方案拟投入2亿元。",
    )
    assert detail.event_stage == "formal_plan"
    assert detail.tradable_catalyst is True


def test_new_product_regulatory_approval_is_distinct_from_registration_change(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    approval_event = _event("product_breakthrough")
    approval_event["title"] = "关于全资子公司药品获得批准上市的公告"
    approval = analyzer.analyze_text(
        approval_event,
        "公司创新药获得批准上市，取得注册批件，可依法开展生产销售。",
    )
    change_event = _event("product_breakthrough")
    change_event["title"] = "关于医疗器械注册证变更的公告"
    changed = analyzer.analyze_text(
        change_event,
        "公司完成医疗器械注册证变更，不涉及新增产品。",
    )
    assert approval.event_stage == "regulatory_approval"
    assert approval.tradable_catalyst is True
    assert changed.event_stage == "renewal_or_change"
    assert changed.tradable_catalyst is False


def test_mass_production_requires_commercialization_evidence(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    event = _event("product_breakthrough")
    event["title"] = "关于新产品实现量产的公告"
    strong = analyzer.analyze_text(
        event,
        "新产品实现量产并已取得客户订单1.2亿元，将批量交付。",
    )
    weak = analyzer.analyze_text(
        event,
        "新产品实现量产，未来市场销售存在不确定性。",
    )
    assert strong.event_stage == "mass_production"
    assert strong.tradable_catalyst is True
    assert weak.tradable_catalyst is False


def test_restructuring_requires_formal_plan_not_planning_or_completion(tmp_path) -> None:
    analyzer = AnnouncementDetailAnalyzer(tmp_path)
    formal_event = _event("restructuring")
    formal_event["title"] = "发行股份购买资产暨重大资产重组报告书（草案）"
    formal = analyzer.analyze_text(
        formal_event,
        "本次交易构成重大资产重组，公司披露发行股份购买资产报告书。",
    )
    planning_event = _event("restructuring")
    planning_event["title"] = "关于筹划重大资产重组的提示性公告"
    planning = analyzer.analyze_text(
        planning_event,
        "公司正在筹划重大资产重组，交易方案尚未确定。",
    )
    completed_event = _event("restructuring")
    completed_event["title"] = "关于重大资产重组资产过户完成的公告"
    completed = analyzer.analyze_text(
        completed_event,
        "本次交易涉及资产已经过户完成。",
    )
    assert formal.event_stage == "formal_plan"
    assert formal.tradable_catalyst is True
    assert planning.event_stage == "planning"
    assert planning.tradable_catalyst is False
    assert completed.event_stage == "completed"
    assert completed.tradable_catalyst is False
