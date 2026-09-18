"""News-first opportunity discovery for the short-term decision agent.

This module does not decide that a stock is buyable.  It turns directly linked,
time-stamped market news into auditable *discovery signals*.  Trend, liquidity,
technical entry, risk and the final recommendation gate still have to approve a
discovered stock.

The split is deliberate:

1. news discovers what deserves analysis;
2. trend/technical evidence confirms whether price action agrees;
3. the final agent may abstain when evidence conflicts or is incomplete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from .frame_utils import find_col


_CODE_RE = re.compile(r"(?<!\d)(?:sh|sz|bj)?(\d{6})(?!\d)", re.IGNORECASE)
_GENERIC_SUBJECTS = {
    "A股公告速递", "环球市场情报", "TMT行业观察", "汽车大新闻", "焦点资讯",
    "公司新闻", "盘中快讯", "公告", "其他", "A股7×24快讯直播",
    "A股动态滚动播报", "股市重要公告提醒",
}

# Negative rules must be checked before positive rules.  For example, "终止回购"
# must never be interpreted as a positive buyback event.
_NEGATIVE_EVENTS: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("regulatory_risk", ("立案", "调查", "处罚", "监管措施", "纪律处分", "问询函"), 96.0),
    ("delisting_risk", ("退市风险", "终止上市", "可能被终止上市", "暂停上市"), 100.0),
    ("earnings_risk", ("预亏", "首亏", "续亏", "业绩变脸", "大幅下降", "净利润下降"), 92.0),
    ("shareholder_reduction", ("减持", "拟减持", "清仓式减持"), 88.0),
    ("event_terminated", ("终止重组", "终止收购", "终止回购", "合同终止", "项目终止"), 90.0),
    ("credit_risk", ("债务逾期", "无法偿还", "资金占用", "违规担保", "冻结", "破产"), 96.0),
    ("product_risk", ("召回", "停产", "安全事故", "重大诉讼"), 86.0),
)

_POSITIVE_EVENTS: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("earnings_positive", ("预增", "扭亏", "净利润增长", "业绩大增", "超预期", "业绩快报"), 90.0),
    ("buyback_or_increase", ("回购股份", "拟回购", "增持股份", "拟增持", "员工持股计划"), 87.0),
    ("major_contract", ("重大合同", "签订合同", "中标", "订单", "框架协议"), 84.0),
    ("product_breakthrough", ("获批", "批准上市", "取得注册证", "技术突破", "首发", "量产"), 84.0),
    ("restructuring", ("重大资产重组", "拟收购", "资产注入", "控制权变更"), 82.0),
    ("policy_support", ("政策支持", "专项资金", "纳入目录", "试点", "补贴"), 78.0),
    ("capacity_growth", ("扩产", "投产", "产能提升", "新建项目"), 72.0),
)

_THEME_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("人工智能", ("人工智能", "AI", "大模型")),
    ("算力", ("算力", "服务器", "数据中心", "智算")),
    ("机器人", ("机器人", "人形机器人", "工业机器人")),
    ("半导体", ("半导体", "芯片", "集成电路", "光刻")),
    ("低空经济", ("低空经济", "无人机", "eVTOL")),
    ("新能源车", ("新能源汽车", "新能源车", "智能驾驶", "固态电池")),
    ("光伏", ("光伏", "组件", "逆变器")),
    ("风电", ("风电", "风机", "海上风电")),
    ("医药", ("创新药", "医药", "药品", "医疗器械")),
    ("消费电子", ("消费电子", "手机", "MR", "VR")),
    ("军工", ("军工", "国防", "航天", "航空发动机")),
)

_NEGATED_EVENT_PHRASES = (
    "不构成重大资产重组", "不构成重组", "不构成关联交易", "不存在重大资产重组",
    "未构成重大资产重组", "不涉及重大资产重组",
)

_CAUTION_MARKERS: tuple[tuple[str, float], ...] = (
    ("尚需提交股东会", 12.0),
    ("尚需股东大会", 12.0),
    ("尚需监管", 12.0),
    ("存在不确定性", 12.0),
    ("存在市场风险", 10.0),
    ("存在风险", 8.0),
    ("业务尚处", 10.0),
    ("处于初期", 10.0),
    ("框架协议", 10.0),
    ("意向协议", 12.0),
    ("拟不超过", 8.0),
    ("拟不超", 8.0),
)


def normalize_a_share_code(value: Any) -> str:
    """Return a six-digit A-share code or an empty string."""
    match = _CODE_RE.search(str(value or "").strip())
    if not match:
        return ""
    code = match.group(1)
    if not code.startswith(("00", "30", "60", "68", "4", "8")):
        return ""
    return code


@dataclass
class NewsOpportunity:
    code: str
    name: str
    polarity: str
    event_type: str
    theme: str
    score: float
    confidence: float
    published_at: str
    sources: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    event_fingerprints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "polarity": self.polarity,
            "event_type": self.event_type,
            "theme": self.theme,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 2),
            "published_at": self.published_at,
            "sources": list(self.sources),
            "evidence": list(self.evidence),
            "event_fingerprints": list(self.event_fingerprints),
            "warnings": list(self.warnings),
        }

    def to_strategy_signal(self):
        """Convert discovery evidence to the existing strategy-pool contract."""
        from .strategy_pools import StrategySignal

        evidence_text = "；".join(self.evidence[:2])
        return StrategySignal(
            strategy_name="新闻事件驱动",
            code=self.code,
            name=self.name,
            board=self.theme or "事件驱动",
            trigger_reason=f"{self.event_type}: {evidence_text}"[:240],
            score=self.score,
            raw_features={
                "source": "news_discovery",
                "event_type": self.event_type,
                "polarity": self.polarity,
                "concept": self.theme or "事件驱动",
                "role": "事件核心",
                "confidence": round(self.confidence, 2),
                "published_at": self.published_at,
                "sources": list(self.sources),
                "evidence": list(self.evidence),
                "event_fingerprints": list(self.event_fingerprints),
            },
        )


@dataclass
class NewsOpportunityScanResult:
    date: str
    opportunities: list[NewsOpportunity] = field(default_factory=list)
    risk_alerts: list[NewsOpportunity] = field(default_factory=list)
    scanned_flashes: int = 0
    linked_flashes: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "opportunities": [item.to_dict() for item in self.opportunities],
            "risk_alerts": [item.to_dict() for item in self.risk_alerts],
            "scanned_flashes": self.scanned_flashes,
            "linked_flashes": self.linked_flashes,
            "warnings": list(self.warnings),
        }


class NewsOpportunityScanner:
    """Discover A-share candidates from direct, auditable news links."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        scfg = cfg.get("short_term_agent", {}) or {}
        self.min_opportunity_score = float(scfg.get("news_discovery_min_score", 76.0))
        self.max_opportunities = int(scfg.get("news_discovery_max_candidates", 20))

    def scan(
        self,
        news_result: Any,
        spot: pd.DataFrame | None = None,
    ) -> NewsOpportunityScanResult:
        date = str(getattr(news_result, "date", "") or "")
        flashes = list(getattr(news_result, "flashes", None) or [])
        result = NewsOpportunityScanResult(date=date, scanned_flashes=len(flashes))
        names_by_code, code_by_name = self._spot_names(spot)
        aggregated: dict[tuple[str, str], NewsOpportunity] = {}

        for flash in flashes:
            content = str(getattr(flash, "content", "") or "").strip()
            if not content:
                continue
            polarity, event_type, base_score = self._classify_event(content)
            if polarity == "neutral":
                continue
            linked = self._linked_stocks(flash, content, code_by_name)
            if not linked:
                continue
            result.linked_flashes += 1
            theme = self._theme(flash, content)
            source = str(getattr(flash, "source", "") or "unknown")
            important = bool(getattr(flash, "important", False))
            caution_penalty, caution_warnings = self._caution_penalty(content)
            fingerprint = str(getattr(flash, "content_hash", "") or "")
            time_text = str(getattr(flash, "time", "") or "")
            published_at = f"{date} {time_text}".strip()

            for code, linked_name, direct_structured in linked:
                name = linked_name or names_by_code.get(code, "") or code
                score = base_score + (5.0 if important else 0.0) + (4.0 if direct_structured else 0.0) - caution_penalty
                confidence = (
                    72.0 + (12.0 if direct_structured else 4.0) + (4.0 if important else 0.0)
                    - caution_penalty * 0.7
                )
                key = (code, polarity)
                item = aggregated.get(key)
                if item is None:
                    item = NewsOpportunity(
                        code=code,
                        name=name,
                        polarity=polarity,
                        event_type=event_type,
                        theme=theme,
                        score=min(100.0, score),
                        confidence=min(99.0, confidence),
                        published_at=published_at,
                    )
                    aggregated[key] = item
                else:
                    if score > item.score:
                        item.score = min(100.0, score)
                        item.event_type = event_type
                        item.theme = theme or item.theme
                        item.published_at = published_at or item.published_at
                    item.confidence = min(99.0, item.confidence + 3.0)
                if source not in item.sources:
                    item.sources.append(source)
                    if len(item.sources) > 1:
                        item.score = min(100.0, item.score + 3.0)
                        item.confidence = min(99.0, item.confidence + 4.0)
                snippet = content[:180]
                if snippet not in item.evidence:
                    item.evidence.append(snippet)
                if fingerprint and fingerprint not in item.event_fingerprints:
                    item.event_fingerprints.append(fingerprint)
                for warning in caution_warnings:
                    if warning not in item.warnings:
                        item.warnings.append(warning)

        positives = [
            item for item in aggregated.values()
            if item.polarity == "positive" and item.score >= self.min_opportunity_score
        ]
        risks = [item for item in aggregated.values() if item.polarity == "negative"]
        result.opportunities = sorted(positives, key=lambda item: (item.score, item.confidence), reverse=True)[
            : self.max_opportunities
        ]
        result.risk_alerts = sorted(risks, key=lambda item: (item.score, item.confidence), reverse=True)
        if not flashes:
            result.warnings.append("没有可供新闻发现扫描的快讯")
        elif not result.opportunities:
            result.warnings.append("没有直接关联 A 股且达到新闻发现阈值的正面事件")
        return result

    @staticmethod
    def _classify_event(text: str) -> tuple[str, str, float]:
        compact = re.sub(r"\s+", "", text)
        purchase_markers = ("采购合同", "采购服务器", "拟采购", "向供应商采购", "承诺采购")
        revenue_markers = ("销售合同", "向客户销售", "向客户提供", "收到中标", "中标通知书")
        if any(marker in compact for marker in purchase_markers) and not any(
            marker in compact for marker in revenue_markers
        ):
            return "neutral", "purchase_or_cost_commitment", 0.0
        for phrase in _NEGATED_EVENT_PHRASES:
            compact = compact.replace(phrase, "")
        for event_type, words, score in _NEGATIVE_EVENTS:
            if any(word in compact for word in words):
                return "negative", event_type, score
        for event_type, words, score in _POSITIVE_EVENTS:
            if any(word in compact for word in words):
                return "positive", event_type, score
        return "neutral", "unclassified", 0.0

    @staticmethod
    def _caution_penalty(text: str) -> tuple[float, list[str]]:
        compact = re.sub(r"\s+", "", text)
        penalty = 0.0
        warnings: list[str] = []
        for marker, value in _CAUTION_MARKERS:
            if marker in compact:
                penalty += value
                warnings.append(f"事件仍含条件/风险披露: {marker}")
        # Multiple caution phrases often describe the same uncertainty.  Cap
        # the penalty so that genuinely material but conditional events remain
        # visible as watch items rather than being assigned nonsensical scores.
        return min(35.0, penalty), warnings

    @staticmethod
    def _spot_names(spot: pd.DataFrame | None) -> tuple[dict[str, str], dict[str, str]]:
        if spot is None or spot.empty:
            return {}, {}
        code_col = find_col(spot, ["代码", "股票代码", "code"])
        name_col = find_col(spot, ["名称", "股票名称", "name"])
        if code_col is None or name_col is None:
            return {}, {}
        names_by_code: dict[str, str] = {}
        code_by_name: dict[str, str] = {}
        for code_raw, name_raw in zip(spot[code_col], spot[name_col]):
            code = normalize_a_share_code(code_raw)
            name = str(name_raw or "").strip()
            if not code or not name:
                continue
            names_by_code[code] = name
            if len(name) >= 3 and name not in code_by_name:
                code_by_name[name] = code
        return names_by_code, code_by_name

    @staticmethod
    def _linked_stocks(
        flash: Any,
        content: str,
        code_by_name: dict[str, str],
    ) -> list[tuple[str, str, bool]]:
        linked: dict[str, tuple[str, bool]] = {}
        for stock in list(getattr(flash, "stocks", None) or []):
            if not isinstance(stock, dict):
                continue
            code = normalize_a_share_code(stock.get("code"))
            if code:
                linked[code] = (str(stock.get("name") or "").strip(), True)
        for entity in list(getattr(flash, "entities", None) or []):
            if not isinstance(entity, dict) or entity.get("type") != "stock_code":
                continue
            code = normalize_a_share_code(entity.get("value"))
            if code and code not in linked:
                linked[code] = (str(entity.get("name") or "").strip(), False)
        for raw in _CODE_RE.findall(content):
            code = normalize_a_share_code(raw)
            if code and code not in linked:
                linked[code] = ("", False)
        # Exact company-name matching is only a fallback.  Requiring at least
        # three Chinese/ASCII characters avoids common one/two-character words.
        for name, code in code_by_name.items():
            if name in content and code not in linked:
                linked[code] = (name, False)
        return [(code, name, direct) for code, (name, direct) in linked.items()]

    @staticmethod
    def _theme(flash: Any, content: str) -> str:
        subjects = [
            str(value).strip()
            for value in list(getattr(flash, "subjects", None) or [])
            if str(value).strip() and str(value).strip() not in _GENERIC_SUBJECTS
        ]
        if subjects:
            return subjects[0]
        for canonical, aliases in _THEME_ALIASES:
            if any(alias in content for alias in aliases):
                return canonical
        return "事件驱动"
