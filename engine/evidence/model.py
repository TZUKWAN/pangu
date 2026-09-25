"""统一证据模型（Phase 3 / Task 3.1）。

所有市场证据（行情/因子/新闻/公告/资金/流动性…）统一为 EvidenceItem，
带完整时间与来源元数据。禁止只有 "status":"ok" 的裸结果。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceCategory(str, Enum):
    price = "price"
    technical = "technical"
    factor = "factor"
    liquidity = "liquidity"
    capital_flow = "capital_flow"
    announcement = "announcement"
    news = "news"
    financial = "financial"
    sector = "sector"
    macro = "macro"
    risk = "risk"
    execution = "execution"


class EvidenceDirection(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"
    risk = "risk"


class SourceTier(str, Enum):
    """来源可信度（Task 3.3）。权重用于证据置信度加权。"""
    A = "A"      # 交易所/巨潮公告、监管机构
    B = "B"      # 高质量财经媒体、结构化行情数据
    C = "C"      # 新闻聚合、二手转载
    D = "D"      # 社交热度、未确认消息

    @property
    def weight(self) -> float:
        return {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}[self.value]

    @property
    def can_trigger_buy(self) -> bool:
        """Tier D 单一来源禁止直接触发 BUY。"""
        return self.value != "D"


class Directness(str, Enum):
    direct = "direct"                    # 直接指向该股票
    sector_inherited = "sector_inherited"  # 板块新闻继承（严格区分）
    market = "market"                    # 全市场


@dataclass
class EvidenceItem:
    evidence_id: str
    entity_type: str                     # stock | sector | market
    entity_id: str                       # 股票代码 / 板块名 / "market"
    category: EvidenceCategory
    event_type: str                      # taxonomy 键或 category 专属键
    direction: EvidenceDirection
    magnitude: float                     # 原始强度 0-1
    source: str
    source_tier: SourceTier
    published_at: Optional[str]          # ISO8601（发布时点）
    effective_at: Optional[str]          # 经济含义生效时点
    fetched_at: str
    expiry_at: Optional[str]             # 证据失效时点（half-life 之外）
    confidence: float                    # 最终置信 0-1
    directness: Directness
    corroboration_count: int = 1         # 跨源印证数（去重后）
    content_hash: str = ""
    raw_reference: str = ""              # 原始链接/出处
    summary: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("category", "direction", "source_tier", "directness"):
            v = getattr(self, name)
            if isinstance(v, str):
                setattr(self, name, {
                    "category": EvidenceCategory, "direction": EvidenceDirection,
                    "source_tier": SourceTier, "directness": Directness}[name](v))
        if not self.content_hash:
            self.content_hash = self.compute_hash()
        if not self.evidence_id:
            self.evidence_id = f"ev_{self.content_hash[:16]}"
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError(f"magnitude must be 0..1, got {self.magnitude}")

    def compute_hash(self) -> str:
        basis = json.dumps([self.entity_id, self.event_type, self.published_at,
                            self.summary], ensure_ascii=False)
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    @property
    def can_trigger_buy(self) -> bool:
        """Tier D 或风险方向不能单独触发 BUY（Phase 13 承接）。"""
        return self.source_tier.can_trigger_buy and \
            self.direction not in (EvidenceDirection.risk, EvidenceDirection.negative)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("category", "direction", "source_tier", "directness"):
            d[k] = getattr(self, k).value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceItem":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})
