"""Phase 3 Evidence Fabric 测试（Task 3.1-3.6）。"""
from __future__ import annotations

import pytest

from engine.evidence.engine import (EntityLinker, build_news_evidence,
                                    cluster_events, decay)
from engine.evidence.model import (Directness, EvidenceCategory,
                                   EvidenceDirection, EvidenceItem, SourceTier)
from engine.evidence.taxonomy import EVENTS, classify_event, tier_of_source

ASOF = "2026-09-25T15:05:00+08:00"


def _news(title, source="财联社", published="2026-09-25T10:00:00+08:00", **kw):
    d = {"title": title, "source": source, **kw}
    if published is not None:
        d["published_at"] = published
    return d


class TestClassification:
    def test_all_25_event_types_classified(self):
        samples = {
            "earnings_beat": "公司发布业绩预增公告",
            "earnings_miss": "三季报业绩预亏",
            "profit_warning": "公司下修业绩预告并计提减值",
            "order_win": "中标 5 亿元大单",
            "contract": "签署战略合作框架协议",
            "buyback": "公司拟回购 2% 股份",
            "insider_increase": "控股股东增持 1%",
            "insider_reduce": "股东拟减持不超过 3%",
            "restructuring": "公司筹划重大资产重组",
            "acquisition": "拟收购同行 51% 股权",
            "regulatory_investigation": "公司收到证监会立案调查通知书",
            "litigation": "公司涉及重大诉讼",
            "product_release": "公司发布新一代芯片产品",
            "policy_support": "工信部出台政策支持并提供补贴",
            "policy_restriction": "监管部门叫停相关业务并整顿",
            "supply_disruption": "工厂事故导致停产",
            "price_increase": "产品全面涨价 10%",
            "capacity_expansion": "公司投资建设新产能基地",
            "capital_raise": "公司披露定增募资预案",
            "dividend": "公司实施分红派息方案",
            "analyst_upgrade": "多家机构上调评级",
            "analyst_downgrade": "机构下调评级至卖出",
            "industry_catalyst": "行业景气上行板块利好频现",
            "macro_event": "央行宣布降准",
            "rumor": "网传公司将被收购",
        }
        assert len(samples) == 25
        for key, title in samples.items():
            got, direction = classify_event(title)
            assert got == key, f"{title!r} → {got}, want {key}"
            spec = EVENTS[key]
            assert spec.half_life_days > 0
        # 方向抽查
        assert classify_event("业绩预增")[1] == EvidenceDirection.positive
        assert classify_event("股东减持")[1] == EvidenceDirection.negative
        assert classify_event("网传消息")[1] == EvidenceDirection.neutral

    def test_tier_mapping(self):
        assert tier_of_source("cninfo公告") == SourceTier.A
        assert tier_of_source("财联社电报") == SourceTier.B
        assert tier_of_source("新浪财经") == SourceTier.C
        assert tier_of_source("雪球热帖") == SourceTier.D


class TestDedupAndConfidence:
    def _build(self, items):
        return build_news_evidence(items, ASOF,
                                   entity_links={str(i): ["600519"] for i in range(len(items))})

    def test_same_news_reposted_5_times_is_one_event(self):
        base = "公司中标城市轨道项目"
        items = [_news(base, source=s, published=f"2026-09-25T1{i}:00:00+08:00")
                 for i, s in enumerate(["财联社", "新浪财经", "证券时报", "同花顺", "雪球"])]
        ev = cluster_events(self._build(items))
        assert len(ev) == 1
        assert ev[0].corroboration_count == 5
        assert ev[0].source_tier == SourceTier.B          # 代表 = 最高层级
        assert ev[0].confidence > 0.5

    def test_official_announcement_beats_repost(self):
        items = [
            _news("公司签订重大合同", source="新浪财经"),
            _news("公司签订重大合同", source="巨潮公告", published="2026-09-25T09:00:00+08:00"),
        ]
        ev = cluster_events(self._build(items))
        assert ev[0].source_tier == SourceTier.A
        assert ev[0].corroboration_count == 2

    def test_different_events_not_clustered(self):
        items = [_news("公司中标地铁项目"), _news("公司回购股份计划")]
        ev = cluster_events(self._build(items))
        assert len(ev) == 2

    def test_single_tier_d_cannot_trigger_buy(self):
        ev = build_news_evidence([_news("网传公司将重组", source="雪球")], ASOF,
                                 entity_links={"0": ["600519"]})
        assert ev[0].source_tier == SourceTier.D
        assert ev[0].can_trigger_buy is False

    def test_rumor_needs_tier_a(self):
        from engine.evidence.taxonomy import spec_of
        assert spec_of("rumor").min_tier_for_buy == SourceTier.A


class TestDecay:
    def test_decay_by_half_life(self):
        from engine.evidence.taxonomy import spec_of
        spec = spec_of("price_increase")                # half-life 7d
        it = EvidenceItem(
            evidence_id="x", entity_type="stock", entity_id="600519",
            category=EvidenceCategory.news, event_type="price_increase",
            direction=EvidenceDirection.positive, magnitude=1.0,
            source="财联社", source_tier=SourceTier.B,
            published_at="2026-09-25T10:00:00+08:00", effective_at=None,
            fetched_at=ASOF, expiry_at=None, confidence=1.0,
            directness=Directness.direct,
            extra={"half_life_days": spec.half_life_days})
        assert decay(it, "2026-09-25T15:00:00+08:00") >= 0.97    # 当天几乎无衰减
        d7 = decay(it, "2026-10-02T10:00:00+08:00")             # 7 天 → 0.5
        assert abs(d7 - 0.5) < 0.01
        d14 = decay(it, "2026-10-09T10:00:00+08:00")            # 14 天 → 0.25
        assert abs(d14 - 0.25) < 0.01

    def test_different_half_lives(self):
        from engine.evidence.model import Directness as D
        def mk(etype, hl, published):
            return EvidenceItem(
                evidence_id=etype, entity_type="stock", entity_id="600519",
                category=EvidenceCategory.news, event_type=etype,
                direction=EvidenceDirection.positive, magnitude=1.0,
                source="财联社", source_tier=SourceTier.B,
                published_at=published, effective_at=None, fetched_at=ASOF,
                expiry_at=None, confidence=1.0, directness=D.direct,
                extra={"half_life_days": hl})
        # 十天前的普通利好(7d 半衰期) vs 五分钟前的重大公告(30d 半衰期)
        old = decay(mk("contract", 7, "2026-09-15T15:00:00+08:00"), ASOF)
        fresh = decay(mk("restructuring", 30, "2026-09-25T15:00:00+08:00"), ASOF)
        assert abs(old - 0.5 ** (10 / 7)) < 0.01 and fresh > 0.99 and fresh > old * 2


class TestEntityLinking:
    def test_direct_vs_sector(self):
        linker = EntityLinker({"600519": "贵州茅台", "000858": "五粮液",
                               "sz.300750": "宁德时代"})
        assert linker.link("贵州茅台发布公告") == ["600519"]
        assert linker.link("茅台批价下行") == ["600519"]      # 简称
        assert linker.link("300750 与车企合作") == ["300750"]
        assert linker.link("动力电池行业景气上行") == []        # 无个股 → 板块继承

    def test_st_prefix_name(self):
        linker = EntityLinker({"600070": "ST富润"})
        assert linker.link("ST富润收到警示函") == ["600070"]
        assert linker.link("富润寻求重整") == ["600070"]      # 去 ST 简称


class TestAsofDiscipline:
    def test_future_news_forbidden(self):
        items = [_news("未来新闻不能使用", published="2026-09-26T09:00:00+08:00")]
        ev = build_news_evidence(items, ASOF, entity_links={"0": ["600519"]})
        assert ev == []                                   # published > asof 丢弃

    def test_no_timestamp_forbidden(self):
        ev = build_news_evidence([_news("无时间戳新闻", published=None)], ASOF,
                                 entity_links={"0": ["600519"]})
        assert ev == []

    def test_sector_inherited_marked(self):
        ev = build_news_evidence(
            [_news("动力电池板块利好，行业景气上行")], ASOF, entity_links={"0": []})
        assert ev[0].directness == Directness.sector_inherited
        assert ev[0].confidence < 1.0
