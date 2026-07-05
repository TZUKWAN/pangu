"""Unified candidate evidence objects for recommendation decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateEvidence:
    code: str
    name: str = ""
    strategy: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    price_action: dict[str, Any] = field(default_factory=dict)
    volume_audit: dict[str, Any] = field(default_factory=dict)
    liquidity: dict[str, Any] = field(default_factory=dict)
    news_evidence: dict[str, Any] = field(default_factory=dict)
    anti_chase: dict[str, Any] = field(default_factory=dict)
    entry_plan: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "strategy": self.strategy,
            "data_quality": self.data_quality,
            "price_action": self.price_action,
            "volume_audit": self.volume_audit,
            "liquidity": self.liquidity,
            "news_evidence": self.news_evidence,
            "anti_chase": self.anti_chase,
            "entry_plan": self.entry_plan,
            "decision": self.decision,
            "raw": self.raw,
        }


def decision_from_item(item: dict[str, Any], default_status: str = "watch") -> dict[str, Any]:
    status = item.get("gate_status") or item.get("status") or default_status
    reasons = []
    for key in ("reject_reason", "watch_reason", "reason", "no_trade_reason"):
        value = item.get(key)
        if value:
            reasons.append(str(value))
    blockers = item.get("blockers") or item.get("block_reasons") or []
    if isinstance(blockers, str):
        blockers = [blockers]
    reasons.extend(str(x) for x in blockers if x)
    return {
        "status": status,
        "reasons": list(dict.fromkeys(reasons)),
        "confidence": item.get("confidence") or item.get("score"),
    }
