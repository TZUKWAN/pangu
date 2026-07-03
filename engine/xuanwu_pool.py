"""玄武池分层：把宽候选池收敛成可行动的严选池。

玄武池不是另一个排序名词，而是一组硬闸门：
市场状态 -> 板块主线 -> 量价走势 -> 新闻舆情 -> 交易计划 -> 多智能体裁决。
任一关键闸门失败时，股票只能进入观察池或排除池。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .data_loader import safe_float


_GENERIC_BOARDS = {"", "全市场强势", "观察池", "其他", "UNKNOWN", "None", "null"}


@dataclass
class XuanwuDecision:
    """单只股票的玄武池判定结果。"""

    code: str
    name: str
    status: str
    score: float
    decision: str
    gates: dict[str, str] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "status": self.status,
            "score": round(self.score, 1),
            "decision": self.decision,
            "gates": self.gates,
            "blockers": self.blockers,
            "evidence": self.evidence,
        }


class XuanwuPoolBuilder:
    """基于已生成的候选、板块、新闻和辩论结果构建玄武池。"""

    VERSION = "xuanwu_pool_v1"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.cfg = cfg.get("xuanwu_pool", cfg) or {}
        self.max_size = int(self.cfg.get("max_size", 8))
        self.watch_size = int(self.cfg.get("watch_size", 60))
        self.min_total_score = float(self.cfg.get("min_total_score", 75))
        self.rule_min_total_score = float(self.cfg.get("rule_min_total_score", 62))
        self.min_rps = float(self.cfg.get("min_rps", 70))
        self.min_debate_confidence = float(self.cfg.get("min_debate_confidence", 70))
        self.min_rr = float(self.cfg.get("min_risk_reward", 1.8))
        self.turnover_min = float(self.cfg.get("turnover_min", 3.0))
        self.turnover_max = float(self.cfg.get("turnover_max", 18.0))

    def build(
        self,
        *,
        sentiment: dict[str, Any],
        boards: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        news: dict[str, Any] | None = None,
        source_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """返回玄武池分层结果，并给每个候选生成判定。"""
        board_names = [str(b.get("name") or "") for b in boards[:5]]
        hot_themes = self._hot_theme_names(news or {})
        source_status = source_status or {}
        decisions: list[XuanwuDecision] = [
            self._decide(c, sentiment, board_names, hot_themes, source_status)
            for c in candidates
        ]
        decisions.sort(key=lambda d: d.score, reverse=True)

        xuanwu = [d for d in decisions if d.status == "xuanwu"][: self.max_size]
        watch = [d for d in decisions if d.status in {"watch", "pending_ai"}][: self.watch_size]
        rejected = [d for d in decisions if d.status == "rejected"][: self.watch_size]
        blocker_counts: dict[str, int] = {}
        for d in decisions:
            for blocker in d.blockers:
                blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1

        return {
            "version": self.VERSION,
            "policy": {
                "max_size": self.max_size,
                "min_total_score": self.min_total_score,
                "rule_min_total_score": self.rule_min_total_score,
                "min_rps": self.min_rps,
                "min_debate_confidence": self.min_debate_confidence,
                "min_risk_reward": self.min_rr,
                "turnover_range": [self.turnover_min, self.turnover_max],
            },
            "summary": {
                "candidate_count": len(candidates),
                "xuanwu_count": len(xuanwu),
                "watch_count": len(watch),
                "rejected_count": len([d for d in decisions if d.status == "rejected"]),
                "top_blockers": sorted(
                    blocker_counts.items(), key=lambda kv: kv[1], reverse=True
                )[:10],
            },
            "xuanwu": [d.to_dict() for d in xuanwu],
            "watch": [d.to_dict() for d in watch],
            "rejected": [d.to_dict() for d in rejected],
            "all_decisions": {d.code: d.to_dict() for d in decisions},
        }

    def _decide(
        self,
        c: dict[str, Any],
        sentiment: dict[str, Any],
        top_boards: list[str],
        hot_themes: set[str],
        source_status: dict[str, Any],
    ) -> XuanwuDecision:
        code = str(c.get("code") or "")
        name = str(c.get("name") or "")
        gates: dict[str, str] = {}
        blockers: list[str] = []
        score_parts: dict[str, float] = {}

        market_score = self._market_score(sentiment, source_status, gates, blockers)
        score_parts["market"] = market_score

        board_score = self._board_score(c, top_boards, hot_themes, gates, blockers)
        score_parts["board"] = board_score

        price_score = self._price_volume_score(c, gates, blockers)
        score_parts["price_volume"] = price_score

        news_score = self._news_score(c, hot_themes, gates, blockers)
        score_parts["news"] = news_score

        trade_score = self._trade_plan_score(c, gates, blockers)
        score_parts["trade_plan"] = trade_score

        debate_score = self._debate_score(c, gates, blockers)
        score_parts["debate"] = debate_score

        total = (
            market_score * 0.10
            + board_score * 0.20
            + price_score * 0.25
            + news_score * 0.15
            + trade_score * 0.15
            + debate_score * 0.15
        )
        total = max(0.0, min(100.0, total))

        hard_blockers = {
            "market_too_cold",
            "board_mapping_missing",
            "board_mapping_weak",
            "price_volume_fail",
            "negative_news_risk",
            "trade_plan_fail",
            "chasing_buy_point",
            "multi_agent_missing",
            "multi_agent_rejected",
            "multi_agent_low_confidence",
        }
        missing_debate = "multi_agent_missing" in blockers
        non_debate_hard = hard_blockers - {"multi_agent_missing", "multi_agent_low_confidence"}
        has_non_debate_hard = any(b in non_debate_hard for b in blockers)
        needs_debate = missing_debate and not has_non_debate_hard and total >= self.min_total_score - 15
        if missing_debate and not needs_debate:
            blockers = [b for b in blockers if b != "multi_agent_missing"]
            gates["debate"] = "not_scoped"

        rule_trial = self._is_rule_validated_trial(total, gates, blockers)

        if rule_trial:
            status = "xuanwu"
            decision = "规则验证玄武试选（未经过真实LLM辩论，轻仓观察）"
            validation_mode = "rule_trial"
        elif total >= self.min_total_score and not any(b in hard_blockers for b in blockers):
            status = "xuanwu"
            decision = "通过玄武池"
            validation_mode = "full"
        elif needs_debate:
            status = "pending_ai"
            decision = "待多智能体论证"
            validation_mode = "pending_ai"
        elif any(b in {"negative_news_risk", "trade_plan_fail", "chasing_buy_point", "multi_agent_rejected"} for b in blockers):
            status = "rejected"
            decision = "排除"
            validation_mode = "blocked"
        else:
            status = "watch"
            decision = "观察"
            validation_mode = "watch"

        evidence = {
            "score_parts": {k: round(v, 1) for k, v in score_parts.items()},
            "board": c.get("board"),
            "rps": safe_float(c.get("rps"), 0.0),
            "pct_change": safe_float(c.get("pct_change"), 0.0),
            "turnover_rate": safe_float(c.get("turnover_rate"), 0.0),
            "recommend_score": safe_float((c.get("recommend") or {}).get("recommend_score"), 0.0),
            "debate": c.get("debate") or None,
            "validation_mode": validation_mode,
        }
        return XuanwuDecision(
            code=code,
            name=name,
            status=status,
            score=total,
            decision=decision,
            gates=gates,
            blockers=blockers,
            evidence=evidence,
        )

    def _is_rule_validated_trial(
        self,
        total: float,
        gates: dict[str, str],
        blockers: list[str],
    ) -> bool:
        """Allow a tightly gated Xuanwu trial when LLM debate is unavailable.

        This is not a replacement for real multi-agent debate. It only prevents
        the pool from staying permanently empty when the candidate has passed
        board, price/volume and trade-plan gates, while still carrying an
        explicit rule_trial evidence flag for the UI and journal.
        """
        if gates.get("debate") != "rule_warn":
            return False
        if total < self.rule_min_total_score:
            return False
        required_pass = ("board", "price_volume", "trade_plan")
        if any(gates.get(k) != "pass" for k in required_pass):
            return False
        disqualifying = {
            "market_too_cold",
            "board_mapping_missing",
            "board_mapping_weak",
            "negative_news_risk",
            "trade_plan_fail",
            "chasing_buy_point",
            "multi_agent_missing",
            "multi_agent_rejected",
            "multi_agent_low_confidence",
        }
        return not any(b in disqualifying for b in blockers)

    def _market_score(
        self,
        sentiment: dict[str, Any],
        source_status: dict[str, Any],
        gates: dict[str, str],
        blockers: list[str],
    ) -> float:
        temp = safe_float(sentiment.get("temperature"), 50.0)
        if temp < 40:
            gates["market"] = "fail"
            blockers.append("market_too_cold")
            return 20.0
        score = min(100.0, max(40.0, temp))
        if source_status.get("market_data") == "degraded":
            score -= 8
        if source_status.get("structured_data") == "degraded":
            score -= 6
        gates["market"] = "pass" if score >= 50 else "weak"
        return max(0.0, score)

    def _board_score(
        self,
        c: dict[str, Any],
        top_boards: list[str],
        hot_themes: set[str],
        gates: dict[str, str],
        blockers: list[str],
    ) -> float:
        board = str(c.get("board") or "")
        if board in _GENERIC_BOARDS:
            gates["board"] = "missing"
            blockers.append("board_mapping_missing")
            return 35.0
        if board.startswith("行业:"):
            gates["board"] = "weak_mapping"
            blockers.append("board_mapping_weak")
            return 45.0
        score = 55.0
        if board in top_boards:
            score += 30.0
        if any(theme and (theme in board or board in theme) for theme in hot_themes):
            score += 15.0
        gates["board"] = "pass" if score >= 70 else "weak"
        return min(100.0, score)

    def _price_volume_score(
        self,
        c: dict[str, Any],
        gates: dict[str, str],
        blockers: list[str],
    ) -> float:
        rps = safe_float(c.get("rps"), 0.0)
        pct = safe_float(c.get("pct_change"), 0.0)
        turnover = safe_float(c.get("turnover_rate"), 0.0)
        technical = c.get("technical") or {}
        volume_ratio = safe_float((technical.get("volume") or {}).get("volume_ratio"), 0.0)
        score = 0.0
        if rps >= self.min_rps:
            score += 32
        if 0 <= pct <= 9.2:
            score += 20
        elif -2 <= pct < 0:
            score += 8
        if self.turnover_min <= turnover <= self.turnover_max:
            score += 20
        elif turnover == 0 and volume_ratio >= 1.5:
            score += 12
        if volume_ratio >= 1.5:
            score += 14
        if technical.get("hints"):
            score += 14
        if score < 55:
            gates["price_volume"] = "fail"
            blockers.append("price_volume_fail")
        else:
            gates["price_volume"] = "pass"
        return min(100.0, score)

    def _news_score(
        self,
        c: dict[str, Any],
        hot_themes: set[str],
        gates: dict[str, str],
        blockers: list[str],
    ) -> float:
        breakdown = (c.get("recommend") or {}).get("score_breakdown") or {}
        news_score = safe_float(breakdown.get("news_sentiment"), 50.0)
        themes = set(str(t) for t in (breakdown.get("news_themes") or []) if str(t).strip())
        text = str(breakdown.get("news_explain") or "")
        stock_news = c.get("stock_news") or []
        risk_words = ("减持", "问询", "处罚", "立案", "退市", "澄清", "风险提示", "异常波动")
        if any(w in text for w in risk_words) or any(
            any(w in str(n.get("title") or n.get("content") or "") for w in risk_words)
            for n in stock_news if isinstance(n, dict)
        ):
            gates["news"] = "fail"
            blockers.append("negative_news_risk")
            return 15.0
        score = news_score
        if themes & hot_themes:
            score += 18
        if stock_news:
            score += 8
        gates["news"] = "pass" if score >= 55 else "neutral"
        return min(100.0, max(35.0, score))

    def _trade_plan_score(
        self,
        c: dict[str, Any],
        gates: dict[str, str],
        blockers: list[str],
    ) -> float:
        ee = c.get("entry_exit") or {}
        close = safe_float(c.get("close"), 0.0)
        rr = safe_float(ee.get("risk_reward_ratio"), 0.0)
        buy_points = ee.get("buy_points") or []
        primary_bp = next((bp for bp in buy_points if bp.get("is_primary")), buy_points[0] if buy_points else {})
        buy = safe_float(primary_bp.get("price") if isinstance(primary_bp, dict) else 0.0, 0.0)
        buy_type = str(primary_bp.get("type") or "") if isinstance(primary_bp, dict) else ""
        stop_obj = ee.get("stop_loss") or {}
        stop = safe_float(stop_obj.get("price") if isinstance(stop_obj, dict) else 0.0, 0.0)
        targets = ee.get("take_profit") or []
        target = safe_float(targets[0].get("price") if targets else 0.0, 0.0)
        score = 0.0
        if buy and close and buy > close * 1.01:
            gates["trade_plan"] = "fail"
            blockers.append("chasing_buy_point")
            return 20.0
        if buy_type and "突破" in buy_type:
            gates["trade_plan"] = "fail"
            blockers.append("chasing_buy_point")
            return 25.0
        if rr >= self.min_rr:
            score += 35
        if buy > 0 and stop > 0 and stop < buy:
            score += 25
        if target > buy:
            score += 20
        if close and buy and buy <= close * 1.005:
            score += 10
        if buy_type and ("回踩" in buy_type or "支撑" in buy_type):
            score += 10
        if ee.get("position"):
            score += 10
        if score < 60:
            gates["trade_plan"] = "fail"
            blockers.append("trade_plan_fail")
        else:
            gates["trade_plan"] = "pass"
        return min(100.0, score)

    def _debate_score(
        self,
        c: dict[str, Any],
        gates: dict[str, str],
        blockers: list[str],
    ) -> float:
        debate = c.get("debate") or {}
        verdict = str(debate.get("verdict") or "")
        confidence = safe_float(debate.get("confidence"), 0.0)
        mode = str(debate.get("mode") or "")
        rule_degraded = bool(debate.get("rule_degraded")) or mode in {"rule", "rule_validation", "rule_skip"}
        if not verdict:
            gates["debate"] = "missing"
            blockers.append("multi_agent_missing")
            return 0.0
        if verdict == "回避":
            if rule_degraded:
                gates["debate"] = "rule_warn"
                return min(35.0, max(20.0, confidence))
            gates["debate"] = "fail"
            blockers.append("multi_agent_rejected")
            return 10.0
        if verdict != "推荐":
            gates["debate"] = "weak"
            return min(60.0, confidence)
        if confidence < self.min_debate_confidence:
            gates["debate"] = "weak"
            blockers.append("multi_agent_low_confidence")
            return confidence
        gates["debate"] = "pass"
        return min(100.0, 70.0 + (confidence - self.min_debate_confidence) * 0.75)

    @staticmethod
    def _hot_theme_names(news: dict[str, Any]) -> set[str]:
        themes: set[str] = set()
        for item in news.get("hot_themes") or []:
            if isinstance(item, (list, tuple)) and item:
                themes.add(str(item[0]))
            elif isinstance(item, dict):
                themes.add(str(item.get("name") or item.get("theme") or ""))
            elif item:
                themes.add(str(item))
        return {t for t in themes if t}
