"""板块驱动新闻证据层。

把泛财经快讯改造成「按热门板块/题材/个股」组织的证据审计：
- 对每条快讯/个股新闻做 bullish / bearish / risk 三分类
- 按题材聚合，得到每个板块的多空证据
- 给每个候选股生成新闻证据摘要，供 RecommendationGate 审计

输出字段与 pipeline、report、web API 对齐，不破坏现有接口。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .news_sentiment import _NEGATIVE_WORDS, _POSITIVE_WORDS, _RISK_WORDS

logger = logging.getLogger("pangu.news_evidence")


@dataclass
class NewsQueryContext:
    """新闻查询上下文：告诉证据层“现在市场关注什么”。"""

    date: str
    hot_themes: list[tuple[str, int]] = field(default_factory=list)
    boards: list[dict[str, Any]] = field(default_factory=list)
    strategy_signals: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    # 候选股已解析出的题材（来自 news_sentiment），code -> set(theme)
    candidate_themes: dict[str, set[str]] = field(default_factory=dict)
    # 显式题材/行业/个股清单（可与 hot_themes/boards/candidates 互补，便于查询层构造）
    themes: list[str] = field(default_factory=list)
    industries: list[str] = field(default_factory=list)
    stocks: list[str] = field(default_factory=list)
    # 显式关键词（可覆盖默认词表，便于策略层注入特定题材关键词）
    positive_keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    risk_keywords: list[str] = field(default_factory=list)

    @property
    def theme_names(self) -> set[str]:
        """所有应主动跟踪的题材名。"""
        names: set[str] = set()
        for theme, _ in self.hot_themes:
            names.add(theme)
        for b in self.boards:
            name = str(b.get("name") or "").strip()
            if name:
                names.add(name)
        for themes in self.candidate_themes.values():
            names.update(themes)
        for sigs in self.strategy_signals.values():
            for sig in sigs:
                for k in ("theme", "board", "concept"):
                    v = sig.get(k)
                    if v:
                        if isinstance(v, str):
                            names.add(v)
                        elif isinstance(v, (list, tuple)):
                            names.update(str(x) for x in v)
        # 显式题材/行业也纳入跟踪范围
        names.update(self.themes)
        names.update(self.industries)
        return {n for n in names if n}

    @property
    def stock_codes(self) -> set[str]:
        """所有应主动跟踪的个股代码。"""
        codes: set[str] = set()
        for c in self.candidates:
            v = str(c.get("code") or "").strip()
            if v:
                codes.add(v)
        codes.update(str(s).strip() for s in self.stocks if str(s).strip())
        return codes


@dataclass
class ThemeEvidence:
    """单个题材的新闻证据。"""

    theme: str
    bullish_snippets: list[str] = field(default_factory=list)
    bearish_snippets: list[str] = field(default_factory=list)
    risk_events: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    support_count: int = 0
    sentiment_label: str = "neutral"  # bullish / bearish / neutral / mixed
    score: float = 50.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "sentiment_label": self.sentiment_label,
            "score": round(self.score, 1),
            "support_count": self.support_count,
            "sources": sorted(self.sources),
            "bullish_snippets": self.bullish_snippets[:5],
            "bearish_snippets": self.bearish_snippets[:5],
            "risk_events": self.risk_events[:5],
        }


@dataclass
class CandidateEvidence:
    """单只个股的新闻证据。"""

    code: str
    name: str = ""
    themes: list[str] = field(default_factory=list)
    bullish_snippets: list[str] = field(default_factory=list)
    bearish_snippets: list[str] = field(default_factory=list)
    risk_events: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    sentiment_label: str = "neutral"
    support_count: int = 0
    verdict_reason: str = ""
    # 个股直接相关的新闻条数（含个股新闻、快讯直接提及代码/名称）
    direct_count: int = 0
    # 仅来自题材继承的新闻条数（无个股直接关联）
    inherited_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "themes": self.themes,
            "sentiment_label": self.sentiment_label,
            "support_count": self.support_count,
            "direct_count": self.direct_count,
            "inherited_count": self.inherited_count,
            "individual_backed": self.direct_count > 0,
            "sources": sorted(self.sources),
            "bullish_snippets": self.bullish_snippets[:5],
            "bearish_snippets": self.bearish_snippets[:5],
            "risk_events": self.risk_events[:5],
            "verdict_reason": self.verdict_reason,
        }


class NewsEvidenceCollector:
    """把原始新闻聚合结果转换为板块/个股证据。"""

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        self.cfg = cfg or {}
        ncfg = self.cfg.get("news_evidence", {})
        self.positive = set(ncfg.get("positive_words", _POSITIVE_WORDS))
        self.negative = set(ncfg.get("negative_words", _NEGATIVE_WORDS))
        self.risk = set(ncfg.get("risk_words", _RISK_WORDS))
        self.max_theme_evidence = int(ncfg.get("max_theme_evidence", 5))
        self.max_candidate_evidence = int(ncfg.get("max_candidate_evidence", 5))
        self.require_evidence = bool(ncfg.get("require_evidence", False))

    # ------------------------------------------------------------------ #
    def collect(
        self,
        ctx: NewsQueryContext,
        flashes: list[dict[str, Any]],
        stock_news: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """主入口。返回包含 theme_evidence / candidate_evidence / market_narrative 的字典。"""
        theme_evidence = self._collect_theme_evidence(ctx.theme_names, flashes, stock_news)
        candidate_evidence = self._collect_candidate_evidence(ctx, theme_evidence, flashes, stock_news)
        market_narrative = self._build_market_narrative(theme_evidence)

        return {
            "date": ctx.date,
            "hot_themes": ctx.hot_themes,
            "theme_evidence": {k: v.to_dict() for k, v in theme_evidence.items()},
            "candidate_evidence": {k: v.to_dict() for k, v in candidate_evidence.items()},
            "market_narrative": market_narrative,
            "top_bullish_themes": self._top_themes(theme_evidence, "bullish"),
            "top_bearish_themes": self._top_themes(theme_evidence, "bearish"),
            "risk_events": self._flatten_risks(theme_evidence),
            # 顶层多空聚合：把所有题材/快讯维度的证据按类别汇总
            "bullish_news": self._flatten_by_label(theme_evidence, "bullish"),
            "bearish_news": self._flatten_by_label(theme_evidence, "bearish"),
            "neutral_news": self._flatten_neutral(theme_evidence, candidate_evidence),
            "catalysts": self._flatten_catalysts(theme_evidence, candidate_evidence),
            "require_evidence": self.require_evidence,
        }

    # ------------------------------------------------------------------ #
    def _classify(self, text: str) -> dict[str, Any]:
        """对单条文本做 bullish / bearish / risk 分类。"""
        text = str(text or "")
        pos = sum(1 for w in self.positive if w in text)
        neg = sum(1 for w in self.negative if w in text)
        risk = sum(1 for w in self.risk if w in text)
        labels: set[str] = set()
        if pos > neg:
            labels.add("bullish")
        elif neg > pos:
            labels.add("bearish")
        if risk > 0:
            labels.add("risk")
        if not labels:
            labels.add("neutral")
        return {
            "labels": labels,
            "pos": pos,
            "neg": neg,
            "risk": risk,
        }

    def _snippet(self, text: str, max_len: int = 120) -> str:
        text = str(text or "").replace("\n", " ").strip()
        return text[:max_len] + ("..." if len(text) > max_len else "")

    def _source_from_item(self, item: dict[str, Any]) -> str:
        return str(item.get("source") or item.get("src") or "unknown")

    def _themes_for_item(
        self,
        item: dict[str, Any],
        theme_names: set[str],
    ) -> set[str]:
        """判断一条新闻关联哪些题材。"""
        text_parts = [str(item.get("content") or "")]
        subjects = item.get("subjects") or []
        if isinstance(subjects, str):
            subjects = [subjects]
        text_parts.extend(str(s) for s in subjects)
        title = item.get("title")
        if title:
            text_parts.append(str(title))
        text = " ".join(text_parts)
        matched: set[str] = set()
        # 1. 结构化题材标签
        for s in subjects:
            s = str(s).strip()
            if s and s in theme_names:
                matched.add(s)
        # 2. 文本关键词命中
        for theme in theme_names:
            if theme in text:
                matched.add(theme)
        return matched

    # ------------------------------------------------------------------ #
    def _collect_theme_evidence(
        self,
        theme_names: set[str],
        flashes: list[dict[str, Any]],
        stock_news: dict[str, list[dict[str, Any]]],
    ) -> dict[str, ThemeEvidence]:
        evidence: dict[str, ThemeEvidence] = {name: ThemeEvidence(theme=name) for name in theme_names}

        for flash in flashes:
            themes = self._themes_for_item(flash, theme_names)
            if not themes:
                continue
            text = str(flash.get("content") or flash.get("title") or "")
            src = self._source_from_item(flash)
            clf = self._classify(text)
            for theme in themes:
                ev = evidence[theme]
                ev.support_count += 1
                ev.sources.add(src)
                if "bullish" in clf["labels"]:
                    ev.bullish_snippets.append(self._snippet(text))
                if "bearish" in clf["labels"]:
                    ev.bearish_snippets.append(self._snippet(text))
                if "risk" in clf["labels"]:
                    ev.risk_events.append(self._snippet(text))

        for code, items in stock_news.items():
            for item in items:
                themes = self._themes_for_item(item, theme_names)
                if not themes:
                    continue
                text = str(item.get("title") or item.get("content") or "")
                src = self._source_from_item(item)
                clf = self._classify(text)
                for theme in themes:
                    ev = evidence[theme]
                    ev.support_count += 1
                    ev.sources.add(src)
                    if "bullish" in clf["labels"]:
                        ev.bullish_snippets.append(self._snippet(text))
                    if "bearish" in clf["labels"]:
                        ev.bearish_snippets.append(self._snippet(text))
                    if "risk" in clf["labels"]:
                        ev.risk_events.append(self._snippet(text))

        for ev in evidence.values():
            ev.bullish_snippets = self._dedup_snippets(ev.bullish_snippets)[: self.max_theme_evidence]
            ev.bearish_snippets = self._dedup_snippets(ev.bearish_snippets)[: self.max_theme_evidence]
            ev.risk_events = self._dedup_snippets(ev.risk_events)[: self.max_theme_evidence]
            ev.sentiment_label, ev.score = self._label_score(ev)
        return evidence

    def _collect_candidate_evidence(
        self,
        ctx: NewsQueryContext,
        theme_evidence: dict[str, ThemeEvidence],
        flashes: list[dict[str, Any]],
        stock_news: dict[str, list[dict[str, Any]]],
    ) -> dict[str, CandidateEvidence]:
        evidence: dict[str, CandidateEvidence] = {}

        for cand in ctx.candidates:
            code = str(cand.get("code") or "").strip()
            if not code:
                continue
            name = str(cand.get("name") or "").strip()
            ev = CandidateEvidence(code=code, name=name)
            direct_bullish = 0
            direct_bearish = 0
            direct_risk = 0

            # 候选股关联题材
            themes = set(ctx.candidate_themes.get(code, set()))
            board = str(cand.get("board") or "").strip()
            if board:
                themes.add(board)
            ev.themes = sorted(themes)

            # 1. 个股新闻（直接关联）
            for item in stock_news.get(code, []):
                text = str(item.get("title") or item.get("content") or "")
                src = self._source_from_item(item)
                clf = self._classify(text)
                ev.support_count += 1
                ev.direct_count += 1
                ev.sources.add(src)
                if "bullish" in clf["labels"]:
                    ev.bullish_snippets.append(self._snippet(text))
                    direct_bullish += 1
                if "bearish" in clf["labels"]:
                    ev.bearish_snippets.append(self._snippet(text))
                    direct_bearish += 1
                if "risk" in clf["labels"]:
                    ev.risk_events.append(self._snippet(text))
                    direct_risk += 1

            # 2. 快讯中直接提到该股票代码或名称（直接关联）
            for flash in flashes:
                text = str(flash.get("content") or flash.get("title") or "")
                if code not in text and name not in text:
                    stocks = flash.get("stocks") or []
                    if not any(str(s.get("code")) == code for s in stocks):
                        continue
                src = self._source_from_item(flash)
                clf = self._classify(text)
                ev.support_count += 1
                ev.direct_count += 1
                ev.sources.add(src)
                if "bullish" in clf["labels"]:
                    ev.bullish_snippets.append(self._snippet(text))
                    direct_bullish += 1
                if "bearish" in clf["labels"]:
                    ev.bearish_snippets.append(self._snippet(text))
                    direct_bearish += 1
                if "risk" in clf["labels"]:
                    ev.risk_events.append(self._snippet(text))
                    direct_risk += 1

            # 3. 继承题材证据（仅作上下文，不计入个股加分）
            inherited_bullish = 0
            for theme in themes:
                te = theme_evidence.get(theme)
                if not te:
                    continue
                ev.sources.update(te.sources)
                before = len(ev.bullish_snippets)
                ev.bullish_snippets.extend(te.bullish_snippets)
                ev.bearish_snippets.extend(te.bearish_snippets)
                ev.risk_events.extend(te.risk_events)
                inherited_bullish += len(te.bullish_snippets)
            ev.inherited_count = inherited_bullish + sum(
                len(theme_evidence.get(t).bearish_snippets) + len(theme_evidence.get(t).risk_events)
                for t in themes if theme_evidence.get(t)
            )

            ev.bullish_snippets = self._dedup_snippets(ev.bullish_snippets)[: self.max_candidate_evidence]
            ev.bearish_snippets = self._dedup_snippets(ev.bearish_snippets)[: self.max_candidate_evidence]
            ev.risk_events = self._dedup_snippets(ev.risk_events)[: self.max_candidate_evidence]
            ev.support_count = max(ev.support_count, len(ev.bullish_snippets) + len(ev.bearish_snippets) + len(ev.risk_events))

            # 定性：只有"直接关联"的新闻才能真正改变个股多空定性。
            # 只有题材继承、无个股直接关联时，保持中性，不因泛题材利好加分（防"伪利好"）。
            if ev.direct_count == 0:
                # 仅题材继承：不计入加分，定性中性（除非有重大风险事件必须降级）
                ev.sentiment_label = "neutral"
                ev.verdict_reason = self._verdict_reason(ev, individual_backed=False)
            else:
                # 用直接相关的片段重新打分
                direct_proxy = ThemeEvidence(
                    theme=code,
                    bullish_snippets=["d"] * direct_bullish,
                    bearish_snippets=["d"] * direct_bearish,
                    risk_events=["d"] * direct_risk,
                )
                ev.sentiment_label, _ = self._label_score(direct_proxy)
                ev.verdict_reason = self._verdict_reason(ev, individual_backed=True)
            evidence[code] = ev
        return evidence

    # ------------------------------------------------------------------ #
    def _label_score(self, ev: Any) -> tuple[str, float]:
        """根据多空风险片段数给证据定性和 0-100 分。"""
        b = len(getattr(ev, "bullish_snippets", []))
        br = len(getattr(ev, "bearish_snippets", []))
        r = len(getattr(ev, "risk_events", []))
        total = b + br + r
        if total == 0:
            return "neutral", 50.0
        # risk 事件权重更高
        bearish_weight = br + r * 1.5
        bullish_weight = b
        if bullish_weight > bearish_weight:
            label = "bullish"
        elif bearish_weight > bullish_weight:
            label = "bearish"
        else:
            label = "mixed"
        score = max(0.0, min(100.0, 50.0 + (b - br - r * 1.5) / max(total, 1) * 50.0))
        return label, score

    def _verdict_reason(self, ev: CandidateEvidence, individual_backed: bool = True) -> str:
        b = len(ev.bullish_snippets)
        br = len(ev.bearish_snippets)
        r = len(ev.risk_events)
        if not individual_backed:
            if r > 0:
                return f"仅泛题材证据，无个股直接关联（继承风险{r} 条），不加分"
            return f"仅泛题材证据，无个股直接关联（继承多{b} / 空{br}），不加分"
        if ev.sentiment_label == "bearish":
            return f"负面/风险新闻占主导（多{b} / 空{br} / 风险{r}）"
        if ev.sentiment_label == "mixed":
            return f"多空新闻交织（多{b} / 空{br} / 风险{r}）"
        if ev.sentiment_label == "bullish" and b >= 2:
            return f"新闻证据偏多（{b} 条）"
        if ev.support_count == 0:
            return "无新闻证据"
        return f"新闻证据偏弱（多{b} / 空{br} / 风险{r}）"

    def _dedup_snippets(self, snippets: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in snippets:
            key = s[:40]
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    # ------------------------------------------------------------------ #
    def _build_market_narrative(self, theme_evidence: dict[str, ThemeEvidence]) -> list[dict[str, Any]]:
        """生成市场级叙事摘要。"""
        narratives = []
        for ev in sorted(theme_evidence.values(), key=lambda x: x.score, reverse=True):
            if ev.support_count == 0:
                continue
            narratives.append({
                "theme": ev.theme,
                "label": ev.sentiment_label,
                "score": round(ev.score, 1),
                "summary": self._narrative_summary(ev),
            })
        return narratives[:8]

    def _narrative_summary(self, ev: ThemeEvidence) -> str:
        b = len(ev.bullish_snippets)
        br = len(ev.bearish_snippets)
        r = len(ev.risk_events)
        if ev.sentiment_label == "bullish":
            return f"{ev.theme} 受新闻正面驱动（{b} 条利多）"
        if ev.sentiment_label == "bearish":
            return f"{ev.theme} 出现负面/风险新闻（空{br} / 风险{r}）"
        return f"{ev.theme} 多空交织（多{b} / 空{br} / 风险{r}）"

    def _top_themes(self, theme_evidence: dict[str, ThemeEvidence], label: str, n: int = 5) -> list[dict[str, Any]]:
        items = [ev for ev in theme_evidence.values() if ev.sentiment_label == label and ev.support_count > 0]
        items.sort(key=lambda x: x.score, reverse=(label == "bullish"))
        if label == "bearish":
            items.sort(key=lambda x: x.score)
        return [ev.to_dict() for ev in items[:n]]

    def _flatten_risks(self, theme_evidence: dict[str, ThemeEvidence]) -> list[str]:
        risks: list[str] = []
        seen: set[str] = set()
        for ev in theme_evidence.values():
            for r in ev.risk_events:
                key = r[:40]
                if key not in seen:
                    seen.add(key)
                    risks.append(f"[{ev.theme}] {r}")
        return risks[:10]

    def _flatten_by_label(self, theme_evidence: dict[str, ThemeEvidence], label: str) -> list[str]:
        """按多/空类别扁平化所有题材的证据片段。"""
        out: list[str] = []
        seen: set[str] = set()
        src = "bullish_snippets" if label == "bullish" else "bearish_snippets"
        for ev in theme_evidence.values():
            if ev.sentiment_label != label and label != "bearish":
                continue
            snippets = getattr(ev, src, [])
            for s in snippets:
                key = s[:40]
                if key not in seen:
                    seen.add(key)
                    out.append(f"[{ev.theme}] {s}")
        return out[:15]

    def _flatten_neutral(
        self,
        theme_evidence: dict[str, ThemeEvidence],
        candidate_evidence: dict[str, CandidateEvidence],
    ) -> list[dict[str, Any]]:
        """聚合中性新闻：题材为中性且有个股证据为中性/无证据的记录。"""
        out: list[dict[str, Any]] = []
        for ev in theme_evidence.values():
            if ev.sentiment_label == "neutral" and ev.support_count == 0:
                continue
            if ev.sentiment_label == "neutral":
                out.append({"type": "theme", "theme": ev.theme, "support_count": ev.support_count})
        for code, cev in candidate_evidence.items():
            if cev.sentiment_label == "neutral":
                out.append({
                    "type": "candidate",
                    "code": code,
                    "name": cev.name,
                    "individual_backed": cev.direct_count > 0,
                    "verdict_reason": cev.verdict_reason,
                })
        return out[:20]

    def _flatten_catalysts(
        self,
        theme_evidence: dict[str, ThemeEvidence],
        candidate_evidence: dict[str, CandidateEvidence],
    ) -> list[dict[str, Any]]:
        """聚合催化事件：bullish 且个股直接关联的强催化。"""
        out: list[dict[str, Any]] = []
        for code, cev in candidate_evidence.items():
            if cev.sentiment_label == "bullish" and cev.direct_count > 0:
                out.append({
                    "code": code,
                    "name": cev.name,
                    "themes": cev.themes,
                    "support_count": cev.support_count,
                    "direct_count": cev.direct_count,
                    "top_snippet": cev.bullish_snippets[0] if cev.bullish_snippets else "",
                })
        # 题材级催化
        for ev in theme_evidence.values():
            if ev.sentiment_label == "bullish" and ev.support_count >= 2:
                out.append({
                    "theme": ev.theme,
                    "support_count": ev.support_count,
                    "score": round(ev.score, 1),
                })
        return out[:15]
