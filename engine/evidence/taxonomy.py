"""新闻事件分类法（Phase 3 / Task 3.2-3.3）。

25 类标准事件：方向、默认半衰期（天）、风险标记、所需印证、来源优先级。
分类基于标题关键词规则（有序、具体优先）；禁止再把新闻当成笼统的
正/负关键词计数。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from engine.evidence.model import EvidenceDirection, SourceTier


@dataclass(frozen=True)
class EventSpec:
    key: str
    direction: EvidenceDirection
    half_life_days: float
    is_risk: bool = False
    min_tier_for_buy: SourceTier = SourceTier.C   # 触发 BUY 所需最低来源层级
    corroboration_hint: int = 1
    keywords: Tuple[str, ...] = ()


EVENTS: Dict[str, EventSpec] = {s.key: s for s in [
    EventSpec("earnings_beat", EvidenceDirection.positive, 14,
              keywords=("业绩预增", "业绩超预期", "净利预增", "盈利预增", "业绩大增")),
    EventSpec("earnings_miss", EvidenceDirection.negative, 10, is_risk=True,
              keywords=("业绩预亏", "净利预亏", "业绩下滑", "盈利下滑", "业绩不达预期")),
    EventSpec("profit_warning", EvidenceDirection.negative, 10, is_risk=True,
              keywords=("业绩预告修正", "下修业绩", "商誉减值", "计提减值")),
    EventSpec("order_win", EvidenceDirection.positive, 14,
              keywords=("中标", "签订合同", "签署合同", "重大订单", "预中标")),
    EventSpec("contract", EvidenceDirection.positive, 14,
              keywords=("战略合作", "框架协议", "合作协议")),
    EventSpec("buyback", EvidenceDirection.positive, 20,
              keywords=("回购",)),
    EventSpec("insider_increase", EvidenceDirection.positive, 20,
              keywords=("增持",)),
    EventSpec("insider_reduce", EvidenceDirection.negative, 15, is_risk=True,
              keywords=("减持", "拟减持")),
    EventSpec("restructuring", EvidenceDirection.positive, 30,
              keywords=("重组", "资产注入", "借壳")),
    EventSpec("acquisition", EvidenceDirection.positive, 30,
              keywords=("收购", "并购")),
    EventSpec("regulatory_investigation", EvidenceDirection.negative, 20, is_risk=True,
              keywords=("立案调查", "立案", "处罚", "监管函", "问询函", "警示函")),
    EventSpec("litigation", EvidenceDirection.negative, 15, is_risk=True,
              keywords=("诉讼", "仲裁", "起诉")),
    EventSpec("product_release", EvidenceDirection.positive, 10,
              keywords=("发布", "推出新品", "发布新品", "上线")),
    EventSpec("policy_support", EvidenceDirection.positive, 20,
              keywords=("政策支持", "补贴", "税收优惠", "利好政策", "纳入试点")),
    EventSpec("policy_restriction", EvidenceDirection.negative, 20, is_risk=True,
              keywords=("限制", "整顿", "叫停", "暂停业务", "禁令")),
    EventSpec("supply_disruption", EvidenceDirection.negative, 10, is_risk=True,
              keywords=("停产", "断供", "供应中断", "事故")),
    EventSpec("price_increase", EvidenceDirection.positive, 7,
              keywords=("涨价", "提价", "上调价格")),
    EventSpec("capacity_expansion", EvidenceDirection.positive, 20,
              keywords=("扩产", "扩建", "新建产能", "投资建设")),
    EventSpec("capital_raise", EvidenceDirection.negative, 15,
              keywords=("定增", "增发", "募资", "配股", "发行可转债")),
    EventSpec("dividend", EvidenceDirection.positive, 10,
              keywords=("分红", "派息", "派现", "利润分配")),
    EventSpec("analyst_upgrade", EvidenceDirection.positive, 10,
              keywords=("上调评级", "买入评级", "上调目标价", "首次覆盖")),
    EventSpec("analyst_downgrade", EvidenceDirection.negative, 10,
              keywords=("下调评级", "卖出评级", "下调目标价")),
    EventSpec("industry_catalyst", EvidenceDirection.positive, 14,
              keywords=("行业景气", "需求爆发", "供不应求", "景气上行", "板块利好")),
    EventSpec("macro_event", EvidenceDirection.neutral, 20,
              keywords=("央行", "美联储", "LPR", "GDP", "PMI", "降准", "降息")),
    EventSpec("rumor", EvidenceDirection.neutral, 3, is_risk=True,
              min_tier_for_buy=SourceTier.A,   # 谣言只有 A 级印证才可能触发 BUY
              keywords=("传闻", "疑似", "网传", "据报道称", "市场消息称")),
    EventSpec("unknown", EvidenceDirection.neutral, 7, keywords=()),
]}

# 有序规则：具体优先（先匹配长词/具体事件，再落入 unknown）
_ORDERED = sorted(EVENTS.values(), key=lambda s: -max((len(k) for k in s.keywords), default=0))


def classify_event(title: str) -> Tuple[str, EvidenceDirection]:
    """标题 → (event_type, direction)。命中多个时取关键词最长者。"""
    if not title:
        return "unknown", EvidenceDirection.neutral
    for spec in _ORDERED:
        for kw in spec.keywords:
            if kw in title:
                return spec.key, spec.direction
    return "unknown", EvidenceDirection.neutral


def spec_of(event_type: str) -> EventSpec:
    return EVENTS.get(event_type, EVENTS["unknown"])


# 来源 → 层级映射（按来源名包含判定；官方公告优先）
TIER_RULES: Tuple[Tuple[str, SourceTier], ...] = (
    ("cninfo", SourceTier.A), ("巨潮", SourceTier.A), ("公告", SourceTier.A),
    ("交易所", SourceTier.A), ("监管", SourceTier.A),
    ("财联社", SourceTier.B), ("证券时报", SourceTier.B), ("上证报", SourceTier.B),
    ("中证报", SourceTier.B), ("新华", SourceTier.B), ("wscn", SourceTier.B),
    ("华尔街见闻", SourceTier.B),
    ("新浪", SourceTier.C), ("东财", SourceTier.C), ("同花顺", SourceTier.C),
    ("雪球", SourceTier.D), ("股吧", SourceTier.D), ("微博", SourceTier.D),
    ("网传", SourceTier.D),
)


def tier_of_source(source: str) -> SourceTier:
    s = (source or "").lower()
    for pat, tier in TIER_RULES:
        if pat.lower() in s:
            return tier
    return SourceTier.C
