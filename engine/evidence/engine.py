"""证据引擎（Phase 3 / Task 3.4-3.6）。

- 新闻 → EvidenceItem（事件分类 + 来源层级 + 时间元数据）
- 跨源去重聚类（fingerprint + 标题相似度）→ corroboration_count
- 时间衰减（事件半衰期）
- 实体链接（个股直指 vs 板块继承，严格区分 directness）
- asof 纪律：published_at > asof 的新闻一律禁用（未来新闻泄漏守卫）
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from engine.decision.clock import SHANGHAI
from engine.evidence.model import (Directness, EvidenceCategory,
                                   EvidenceDirection, EvidenceItem, SourceTier)
from engine.evidence.taxonomy import classify_event, spec_of, tier_of_source


def _parse_ts(s: Optional[str]) -> Optional[_dt.datetime]:
    if not s:
        return None
    dt = _dt.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=SHANGHAI)
    return dt.astimezone(SHANGHAI)


def build_news_evidence(items: Sequence[dict], asof: str,
                        entity_links: Optional[Dict[str, List[str]]] = None,
                        default_category: EvidenceCategory = EvidenceCategory.news,
                        ) -> List[EvidenceItem]:
    """新闻/公告条目 → 证据列表。

    items: [{title, published_at, source, url?, summary?}]
    entity_links: {item_index: [code,...]}（由调用方先做实体链接；缺失则空）
    asof 纪律：published_at > asof 的条目直接丢弃（未来新闻禁止使用）。
    """
    asof_dt = _parse_ts(asof)
    out: List[EvidenceItem] = []
    for i, it in enumerate(items):
        published = it.get("published_at")
        p_dt = _parse_ts(published)
        if p_dt is None or asof_dt is None or p_dt > asof_dt:
            continue                                   # 未来/无时间戳新闻禁用
        title = (it.get("title") or it.get("summary") or "").strip()
        if not title:
            continue
        etype, direction = classify_event(title)
        spec = spec_of(etype)
        source = it.get("source", "unknown")
        tier = tier_of_source(source)
        fetched = it.get("fetched_at") or asof
        expiry = (p_dt + _dt.timedelta(days=spec.half_life_days * 4)).isoformat(timespec="seconds")
        codes = (entity_links or {}).get(str(i)) or []
        links: List[Tuple[str, Directness]] = [(c, Directness.direct) for c in codes]
        if not links and spec.key in ("industry_catalyst", "policy_support",
                                      "policy_restriction", "macro_event"):
            links.append(("*sector*", Directness.sector_inherited))
        if not links:
            links.append(("*market*", Directness.market))
        for code, directness in links:
            magnitude = _magnitude(title, spec)
            out.append(EvidenceItem(
                evidence_id="", entity_type="stock" if not code.startswith("*")
                else ("sector" if code == "*sector*" else "market"),
                entity_id=code.lstrip("*"),
                category=default_category, event_type=etype, direction=direction,
                magnitude=magnitude, source=source, source_tier=tier,
                published_at=published, effective_at=published,
                fetched_at=fetched, expiry_at=expiry,
                confidence=0.0,                      # 聚类后统一计算
                directness=directness,
                content_hash=hashlib.sha256(
                    json_dumps([norm_title(title), etype]).encode()).hexdigest(),
                raw_reference=it.get("url", ""),
                summary=title[:160],
                extra={"half_life_days": spec.half_life_days}))
    return out


def cluster_events(items: List[EvidenceItem],
                   similarity_threshold: float = 0.45) -> List[EvidenceItem]:
    """同一事件跨媒体转载只算一个事件（Task 3.4）。

    聚类键：(entity_id, event_type, published 日期, 内容 hash 相似)。
    同簇代表 = 层级最高（其次最新）；corroboration_count = 簇内条数。
    """
    clusters: Dict[Tuple, List[EvidenceItem]] = {}
    for it in items:
        day = (it.published_at or "")[:10]
        key = (it.entity_id, it.event_type, day)
        placed = False
        for ckey, members in clusters.items():
            if ckey[0] != key[0] or ckey[1] != key[1]:
                continue
            rep = members[0]
            if _title_similarity(rep.summary, it.summary) >= similarity_threshold \
                    or rep.content_hash == it.content_hash:
                members.append(it)
                placed = True
                break
        if not placed:
            clusters[(key[0], key[1], key[2], len(clusters))] = [it]
    out: List[EvidenceItem] = []
    for members in clusters.values():
        rep = sorted(members, key=lambda m: (m.source_tier.weight,
                                             m.published_at or ""), reverse=True)[0]
        rep.corroboration_count = len(members)
        rep.confidence = _confidence(rep)
        out.append(rep)
    return out


def _confidence(it: EvidenceItem) -> float:
    """置信 = 层级权重 × 直接性 × min(1, 1+0.15×(印证数-1))，封顶 1。"""
    direct = {"direct": 1.0, "sector_inherited": 0.6, "market": 0.4}[it.directness.value]
    boost = min(1.0, 1.0 + 0.15 * (it.corroboration_count - 1))
    return round(min(1.0, it.source_tier.weight * direct * boost), 3)


def decay(item: EvidenceItem, asof: str) -> float:
    """半衰期衰减后的有效强度（Task 3.5）。exp(-age/half_life)。"""
    p = _parse_ts(item.published_at)
    a = _parse_ts(asof)
    if p is None or a is None:
        return 0.0
    age_days = max(0.0, (a - p).total_seconds() / 86400.0)
    hl = float(item.extra.get("half_life_days", 7)) or 7.0
    return round(item.magnitude * pow(0.5, age_days / hl), 4)


# --------------------------------------------------------------------------- #
# 实体链接（Task 3.6）
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    """规范化公司名：去 ST/星号前缀、去组织后缀。"""
    n = (name or "").strip().upper().replace(" ", "")
    n = re.sub(r"^(\*?ST)", "", n)
    n = re.sub(r"(股份|集团|科技|有限|公司)$", "", n)
    return n.strip("*")


class EntityLinker:
    """公司简称/代码 → 股票代码。

    处理：全名、代码、去 ST 简称、常用末二字昵称（如 贵州茅台→茅台），
    昵称冲突（歧义）时不注册，宁缺毋滥。
    """

    def __init__(self, universe_names: Dict[str, str]):
        self._by_code: Dict[str, str] = {}
        self._by_name: Dict[str, str] = {}
        self._by_norm: Dict[str, List[str]] = {}
        self._by_alias: Dict[str, List[str]] = {}
        for code, name in universe_names.items():
            c = str(code).split(".")[-1].zfill(6)
            self._by_code[c] = name
            self._by_name.setdefault(name, c)
            n = _norm_name(name)
            self._by_norm.setdefault(n, []).append(c)
        # 昵称：规范化名 ≥4 字时注册末二字（唯一才收）
        alias_votes: Dict[str, Dict[str, int]] = {}
        for code, name in universe_names.items():
            n = _norm_name(name)
            if len(n) >= 4:
                alias_votes.setdefault(n[-2:], {}).setdefault(n, 0)
                alias_votes[n[-2:]][n] += 1
        for alias, holders in alias_votes.items():
            if len(holders) == 1:
                n = next(iter(holders))
                self._by_alias.setdefault(alias, []).extend(
                    c for c in self._by_norm.get(n, []))

    def link(self, text: str) -> List[str]:
        found: List[str] = []
        for code in self._by_code:
            if code in text and code not in found:
                found.append(code)
        for name, code in self._by_name.items():
            if len(name) >= 2 and name in text and code not in found:
                found.append(code)
        for norm, codes in self._by_norm.items():
            if len(norm) >= 2 and norm in text:
                for c in codes:
                    if c not in found:
                        found.append(c)
        for alias, codes in self._by_alias.items():
            if alias in text:
                for c in codes:
                    if c not in found:
                        found.append(c)
        return found[:5]





def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def norm_title(title: str) -> str:
    t = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", title or "")
    return t


def _title_similarity(a: str, b: str) -> float:
    ta, tb = set(norm_title(a)), set(norm_title(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _magnitude(title: str, spec) -> float:
    """基础强度：命中词越长/数量词越具体 → 略强。0.5 基线，封顶 1。"""
    hits = sum(1 for kw in spec.keywords if kw in title)
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*%|(\d+)亿", title)
    m = 0.5 + 0.1 * (hits - 1)
    if nums:
        m += 0.15
    return round(min(1.0, m), 3)
