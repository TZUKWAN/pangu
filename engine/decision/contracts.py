"""Pangu 3.0 统一决策契约（Phase 1 / Task 1.1）。

本模块是 Pangu 作为"研究与下一交易日决策中间层"的对外领域模型。
宿主 Agent（Claude Code / Codex / OpenCode / Kimi / ZCode）只看到这里的对象。

设计规则：
- `decision` 输出层只允许 BUY / WATCH / AVOID / BLOCKED 四个枚举，
  内部的 final/candidate/watch/kept 等历史状态不得直接暴露给用户。
- `confidence_type` 明确区分 raw（未校准启发分）、evidence（证据强度）、
  calibrated（经过 OOS 校准的概率）。未校准分数永远不得表述为"上涨概率"。
- 所有对象可 JSON 序列化/反序列化（roundtrip），schema_version 向后兼容。
- 本模块不依赖任何 LLM、网络或数据库（纯领域模型）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "pangu3.decision.v1"

TOP20_DEFAULT_LIMIT = 20


class DecisionAction(str, Enum):
    BUY = "BUY"
    WATCH = "WATCH"
    AVOID = "AVOID"
    BLOCKED = "BLOCKED"


class ConfidenceType(str, Enum):
    RAW = "raw"                  # 未校准启发分（禁止表述为概率）
    EVIDENCE = "evidence"        # 证据加权置信（0-1，非概率语义）
    CALIBRATED = "calibrated"    # 经独立 OOS 校准的概率（附窗口/样本）


class MarketStatus(str, Enum):
    OPEN = "open"
    LUNCH_BREAK = "lunch_break"
    CLOSED_PRE = "closed_pre"          # 当日开盘前
    CLOSED_AFTER = "closed_after"      # 当日盘后（主决策窗口）
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------

@dataclass
class DecisionRequest:
    """一次决策请求。asof 是"以什么时点的信息分析"，缺省 = 当前时刻。"""
    query_timestamp: Optional[str] = None       # ISO8601 上海时区；None = now
    asof: Optional[str] = None                  # 信息截止时点（ISO8601）
    market: str = "CN"
    limit: int = TOP20_DEFAULT_LIMIT
    requested_execution_date: Optional[str] = None  # YYYY-MM-DD；None = 下一交易日
    risk_profile: str = "default"
    force_refresh: bool = False
    codes: Optional[List[str]] = None           # 限定股票池（单票分析用）
    include_watchlist: bool = True

    def __post_init__(self) -> None:
        if self.limit < 1 or self.limit > 100:
            raise ValueError(f"limit must be 1..100, got {self.limit}")
        if self.market != "CN":
            raise ValueError(f"only CN market is supported, got {self.market!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionRequest":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (d or {}).items() if k in known}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# 决策对象
# ---------------------------------------------------------------------------

@dataclass
class StockDecision:
    """单只股票的下一交易日决策 + 完整可审计证据引用。"""
    rank: int
    code: str
    name: str
    decision: DecisionAction
    score: float
    confidence: Optional[float]
    confidence_type: ConfidenceType
    execution_date: str                                   # 下一交易日 YYYY-MM-DD
    entry_condition: str
    entry_zone: Optional[List[float]] = None              # [low, high]
    invalid_condition: str = ""
    stop_loss: Optional[float] = None
    target_zone: Optional[List[float]] = None
    expected_holding_days: Optional[int] = None
    holding_range: Optional[List[int]] = None             # [min, max] 交易日
    exit_conditions: List[str] = field(default_factory=list)
    primary_strategy: str = ""
    factor_evidence: List[str] = field(default_factory=list)   # evidence_id 列表
    event_evidence: List[str] = field(default_factory=list)
    market_evidence: List[str] = field(default_factory=list)
    fundamental_evidence: List[str] = field(default_factory=list)
    liquidity_evidence: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    freshness: Dict[str, Any] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.decision, str):
            try:
                self.decision = DecisionAction(self.decision)
            except ValueError:
                raise ValueError(
                    f"decision must be one of {[a.value for a in DecisionAction]}, "
                    f"got {self.decision!r}") from None
        if isinstance(self.confidence_type, str):
            self.confidence_type = ConfidenceType(self.confidence_type)
        if self.rank < 1:
            raise ValueError(f"rank must be >=1, got {self.rank}")
        if not self.code:
            raise ValueError("code is required")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        d["confidence_type"] = self.confidence_type.value
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StockDecision":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (d or {}).items() if k in known}
        return cls(**kwargs)


@dataclass
class Top20DecisionSet:
    """稳定排序的决策候选集（≤ limit；不足不硬凑）。"""
    asof: str
    execution_date: str
    decisions: List[StockDecision]
    market_status: MarketStatus = MarketStatus.UNKNOWN
    market_conclusion: str = ""
    note: Optional[str] = None                    # 为何不足 N 只
    data_status: str = "ok"                       # ok | degraded | failed

    def __post_init__(self) -> None:
        if isinstance(self.market_status, str):
            self.market_status = MarketStatus(self.market_status)
        seen = set()
        for i, dec in enumerate(self.decisions, start=1):
            if dec.code in seen:
                raise ValueError(f"duplicate code in decision set: {dec.code}")
            seen.add(dec.code)
            if dec.rank != i:
                raise ValueError(
                    f"decisions must be consecutively ranked, got rank {dec.rank} at pos {i}")
        buys = [d for d in self.decisions if d.decision == DecisionAction.BUY]
        for d in buys:
            if self.data_status == "failed":
                raise ValueError("BUY decisions are forbidden when data_status=failed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "asof": self.asof,
            "execution_date": self.execution_date,
            "market_status": self.market_status.value,
            "market_conclusion": self.market_conclusion,
            "data_status": self.data_status,
            "note": self.note,
            "counts": self.counts(),
            "decisions": [d.to_dict() for d in self.decisions],
        }

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d in self.decisions:
            out[d.decision.value] = out.get(d.decision.value, 0) + 1
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Top20DecisionSet":
        return cls(
            asof=d["asof"],
            execution_date=d["execution_date"],
            decisions=[StockDecision.from_dict(x) for x in d.get("decisions", [])],
            market_status=MarketStatus(d.get("market_status", "unknown")),
            market_conclusion=d.get("market_conclusion", ""),
            note=d.get("note"),
            data_status=d.get("data_status", "ok"),
        )


# ---------------------------------------------------------------------------
# 运行记录
# ---------------------------------------------------------------------------

@dataclass
class DecisionRun:
    """一次 /pangu 决策的完整可审计记录。"""
    run_id: str
    query_timestamp: str
    decision_date: str
    execution_date: str
    asof_timestamp: str
    market_status: MarketStatus
    data_freshness: Dict[str, Any] = field(default_factory=dict)
    source_health: Dict[str, Any] = field(default_factory=dict)
    recommendations: Optional[Top20DecisionSet] = None
    warnings: List[str] = field(default_factory=list)
    blocked_reasons: List[str] = field(default_factory=list)
    model_version: str = ""
    evidence_version: str = ""
    latency: Dict[str, float] = field(default_factory=dict)
    request: Optional[DecisionRequest] = None

    def __post_init__(self) -> None:
        if isinstance(self.market_status, str):
            self.market_status = MarketStatus(self.market_status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "query_timestamp": self.query_timestamp,
            "decision_date": self.decision_date,
            "execution_date": self.execution_date,
            "asof_timestamp": self.asof_timestamp,
            "market_status": self.market_status.value,
            "data_freshness": self.data_freshness,
            "source_health": self.source_health,
            "recommendations": self.recommendations.to_dict() if self.recommendations else None,
            "warnings": list(self.warnings),
            "blocked_reasons": list(self.blocked_reasons),
            "model_version": self.model_version,
            "evidence_version": self.evidence_version,
            "latency": dict(self.latency),
            "request": self.request.to_dict() if self.request else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionRun":
        return cls(
            run_id=d["run_id"],
            query_timestamp=d["query_timestamp"],
            decision_date=d["decision_date"],
            execution_date=d["execution_date"],
            asof_timestamp=d["asof_timestamp"],
            market_status=MarketStatus(d.get("market_status", "unknown")),
            data_freshness=d.get("data_freshness", {}),
            source_health=d.get("source_health", {}),
            recommendations=Top20DecisionSet.from_dict(d["recommendations"])
            if d.get("recommendations") else None,
            warnings=list(d.get("warnings", [])),
            blocked_reasons=list(d.get("blocked_reasons", [])),
            model_version=d.get("model_version", ""),
            evidence_version=d.get("evidence_version", ""),
            latency=d.get("latency", {}),
            request=DecisionRequest.from_dict(d["request"]) if d.get("request") else None,
        )
