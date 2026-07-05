"""最终推荐闸门。

把策略池产出、QuantGuard 结果、市场阶段统一为：
- final_recommendations（正式可推荐）
- watchlist（观察池）
- rejected（被明确拒绝）

硬闸门（6 道）：
1. 市场阶段适配（当前阶段禁止的策略直接降级观察池）
2. 数据真实性（RPS/资金必须为真实或已降级观察池）
3. QuantGuard 安全（通过 kept）
4. 资金流确认（观察池要求真实资金流）
5. 买卖点可执行（EntryExit 成功生成）
6. LLM/Agent 复核（已启用 LLM 时要求通过 review）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .data_loader import DataLoader, safe_float
from .entry_exit import EntryExitEngine
from .quant_guard import GuardResult
from .strategy_pools import StrategySignal
from .trend_scanner import StockCandidate

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    final_recommendations: list[dict[str, Any]] = field(default_factory=list)
    watchlist: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    gate_log: list[dict[str, Any]] = field(default_factory=list)
    block_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_recommendations": self.final_recommendations,
            "watchlist": self.watchlist,
            "rejected": self.rejected,
            "gate_log": self.gate_log,
            "block_reasons": self.block_reasons,
        }


class RecommendationGate:
    def __init__(
        self,
        dl: DataLoader,
        guard_result: GuardResult,
        market_phase: dict[str, Any],
        cfg: dict[str, Any] | None = None,
        recommendation_allowed: bool = True,
        date: Optional[str] = None,
        temperature: float = 50.0,
    ) -> None:
        self.dl = dl
        self.guard = guard_result
        self.phase = market_phase
        self.cfg = cfg or {}
        self.entry_engine = EntryExitEngine(dl, self.cfg.get("entry_exit", {}))
        self.date = date
        self.temperature = temperature

        self.allowed_strategies = set(self.phase.get("allowed_strategies", []))
        self.forbidden_strategies = set(self.phase.get("forbidden_strategies", []))
        self.recommendation_allowed = recommendation_allowed and self.phase.get("recommendation_allowed", True)

    def pass_gate(
        self,
        pooled_signals: dict[str, list[StrategySignal]],
        candidate_map: dict[str, StockCandidate],
        candidates: Optional[list[dict[str, Any]]] = None,
        llm_review_map: Optional[dict[str, dict[str, Any]]] = None,
    ) -> GateResult:
        result = GateResult()
        if not self.recommendation_allowed:
            result.gate_log.append({"gate": "global", "passed": False, "reason": "全局 recommendation_allowed=False"})

        kept_codes = {c.code for c in self.guard.kept}
        watch_codes = {c.code for c in self.guard.watch}
        rejected_codes = {item["code"] for item in self.guard.rejected}

        candidate_dict: dict[str, dict[str, Any]] = {}
        if candidates:
            candidate_dict = {str(c.get("code")): c for c in candidates if c.get("code")}

        # 同一 code 取最高分的策略信号
        best_signal: dict[str, tuple[str, StrategySignal]] = {}
        for strategy_name, signals in pooled_signals.items():
            for sig in signals:
                code = sig.code
                existing = best_signal.get(code)
                if existing is None or sig.score > existing[1].score:
                    best_signal[code] = (strategy_name, sig)

        all_codes = set(candidate_map.keys()) | set(candidate_dict.keys())

        for code in all_codes:
            cand = candidate_map.get(code)
            cand_dict = candidate_dict.get(code)
            signal_pair = best_signal.get(code)

            if signal_pair:
                strategy_name, sig = signal_pair
                item = self._build_item(sig, cand)
                if cand_dict:
                    for k in ("entry_exit", "technical", "debate", "xuanwu", "recommend",
                              "anti_chase", "entry_plan", "entry_style", "news_evidence",
                              "volume_audit"):
                        item.setdefault(k, cand_dict.get(k))
                self._judge_strategy_signal(code, strategy_name, sig, cand, item, result, watch_codes, rejected_codes, llm_review_map)
            else:
                # 旧 trend 扫描补充候选：无策略信号，只过 guard，不进入正式推荐
                item = dict(cand_dict) if cand_dict else (cand.to_dict() if cand else {})
                item.setdefault("gate_status", "pending")
                item.setdefault("strategy_name", "trend_supplement")
                item.setdefault("score", 0)
                self._judge_trend_only(code, cand, item, result, watch_codes, rejected_codes)

        # 去重：同一 code 多个策略取最高 score
        result.final_recommendations = self._dedup(result.final_recommendations)
        result.watchlist = self._dedup(result.watchlist)
        result.rejected = self._dedup(result.rejected)

        # 聚合 block_reasons：按 gate 维度统计被拦截原因（不去重，保留每条记录的统计意义）
        blocked = [g for g in result.gate_log if not g.get("passed", True) and g.get("gate") != "global"]
        seen: set[str] = set()
        block_reasons: list[str] = []
        for g in blocked:
            reason = str(g.get("reason") or "").strip()
            key = f"{g.get('gate')}|{reason}"
            if reason and key not in seen:
                seen.add(key)
                block_reasons.append(f"[{g.get('gate')}] {reason}")
        result.block_reasons = block_reasons
        return result

    def _judge_strategy_signal(
        self,
        code: str,
        strategy_name: str,
        sig: StrategySignal,
        cand: Optional[StockCandidate],
        item: dict[str, Any],
        result: GateResult,
        watch_codes: set[str],
        rejected_codes: set[str],
        llm_review_map: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        # 1. 市场阶段
        if not self._phase_allowed(strategy_name, sig):
            item["gate_status"] = "watch"
            item["watch_reason"] = f"当前阶段 {self.phase.get('market_phase')} 禁止策略 {strategy_name}"
            result.watchlist.append(item)
            result.gate_log.append({"code": code, "gate": "phase", "passed": False, "reason": item["watch_reason"]})
            return

        # 2. QuantGuard
        if code in rejected_codes:
            item["gate_status"] = "rejected"
            item["reject_reason"] = next((i["reason"] for i in self.guard.rejected if i["code"] == code), "QuantGuard 拒绝")
            result.rejected.append(item)
            result.gate_log.append({"code": code, "gate": "guard", "passed": False, "reason": item["reject_reason"]})
            return

        is_watch = code in watch_codes

        # 3. 数据真实性：正式推荐必须有真实 RPS
        if cand and cand.rps_mode != "real" and not is_watch:
            item["gate_status"] = "watch"
            item["watch_reason"] = f"RPS 模式为 {cand.rps_mode}"
            result.watchlist.append(item)
            result.gate_log.append({"code": code, "gate": "rps_real", "passed": False, "reason": item["watch_reason"]})
            return

        # 4. 资金流确认：不可用不直接阻断，除非策略强依赖资金流。
        valid_fund_status = {"available", "ok"}
        if cand and cand.fund_flow_status not in valid_fund_status and self._strategy_requires_fund_flow(strategy_name) and not is_watch:
            item["gate_status"] = "watch"
            item["watch_reason"] = f"策略强依赖资金流，但资金流状态 {cand.fund_flow_status}"
            result.watchlist.append(item)
            result.gate_log.append({"code": code, "gate": "fund_flow", "passed": False, "reason": item["watch_reason"]})
            return
        if cand and cand.fund_flow_status not in valid_fund_status:
            item["fund_flow_risk"] = f"资金流状态 {cand.fund_flow_status}，仅作中性偏弱证据"

        # 5. 量能审计：缺失/异常/无量突破不能进入 final
        va = item.get("volume_audit") or {}
        if va:
            va_status = va.get("status")
            pattern = va.get("price_volume_pattern")
            if va_status in {"missing", "abnormal"}:
                item["gate_status"] = "watch"
                item["watch_reason"] = f"量能审计不足：{va.get('reason', va_status)}"
                result.watchlist.append(item)
                result.gate_log.append({"code": code, "gate": "volume_audit", "passed": False, "reason": item["watch_reason"]})
                return
            if pattern in {"breakout_without_volume", "distribution_volume", "weak_rebound_no_volume", "panic_volume"}:
                item["gate_status"] = "watch"
                item["watch_reason"] = f"量价形态不支持正式推荐：{va.get('reason', pattern)}"
                result.watchlist.append(item)
                result.gate_log.append({"code": code, "gate": "volume_audit", "passed": False, "reason": item["watch_reason"]})
                return

        # 6. EntryExit 可执行（优先使用 Pipeline 已计算的买卖点）
        if item.get("entry_exit") and item["entry_exit"].get("buy_points"):
            pass
        else:
            try:
                ee = self.entry_engine.compute(cand, temperature=self.temperature, date=self.date) if cand else None
                if ee:
                    item["entry_exit"] = ee.to_dict()
                else:
                    item["watch_reason"] = "买卖点计算失败"
                    result.watchlist.append(item)
                    result.gate_log.append({"code": code, "gate": "entry_exit", "passed": False, "reason": item["watch_reason"]})
                    return
            except Exception as exc:  # noqa: BLE001
                item["watch_reason"] = f"买卖点异常: {exc}"
                result.watchlist.append(item)
                result.gate_log.append({"code": code, "gate": "entry_exit", "passed": False, "reason": item["watch_reason"]})
                return

        # 7. 反追涨闸门
        ac = item.get("anti_chase") or {}
        if ac.get("status") == "blocked":
            item["gate_status"] = "rejected"
            item["reject_reason"] = f"反追涨：{ac.get('reason', '')}"
            result.rejected.append(item)
            result.gate_log.append({"code": code, "gate": "anti_chase", "passed": False, "reason": item["reject_reason"]})
            return
        if ac.get("status") == "watch":
            item["gate_status"] = "watch"
            item["watch_reason"] = f"反追涨：{ac.get('reason', '')}"
            result.watchlist.append(item)
            result.gate_log.append({"code": code, "gate": "anti_chase", "passed": False, "reason": item["watch_reason"]})
            return

        # 8. 条件买点：追价型买点禁止进入 final
        entry_plan = item.get("entry_plan") or {}
        if entry_plan.get("is_chasing"):
            item["gate_status"] = "watch"
            item["watch_reason"] = f"追价型买点：{entry_plan.get('trigger_condition', '')}"
            result.watchlist.append(item)
            result.gate_log.append({"code": code, "gate": "entry_plan", "passed": False, "reason": item["watch_reason"]})
            return
        if item.get("entry_style") == "breakout_confirm":
            trigger_price = safe_float(entry_plan.get("trigger_price"), 0.0)
            current_price = safe_float(entry_plan.get("current_price"), 0.0)
            if trigger_price > 0 and current_price > trigger_price * 1.01:
                item["gate_status"] = "watch"
                item["watch_reason"] = "突破确认型买点已偏离触发价 >1%，不再追价"
                result.watchlist.append(item)
                result.gate_log.append({"code": code, "gate": "entry_plan", "passed": False, "reason": item["watch_reason"]})
                return

        # 9. 新闻多空证据审计
        ev = item.get("news_evidence") or {}
        ev_label = ev.get("sentiment_label", "")
        if ev and ev_label == "bearish":
            item["gate_status"] = "rejected"
            item["reject_reason"] = f"新闻利空：{ev.get('verdict_reason', '')}"
            result.rejected.append(item)
            result.gate_log.append({"code": code, "gate": "news_evidence", "passed": False, "reason": item["reject_reason"]})
            return
        if ev and ev_label == "mixed" and ev.get("risk_events"):
            item["gate_status"] = "watch"
            item["watch_reason"] = f"新闻多空交织且含风险：{ev.get('verdict_reason', '')}"
            result.watchlist.append(item)
            result.gate_log.append({"code": code, "gate": "news_evidence", "passed": False, "reason": item["watch_reason"]})
            return
        if ev and ev_label in ("neutral", "weak") and ev.get("support_count", 0) == 0:
            item["gate_status"] = "watch"
            item["watch_reason"] = f"新闻证据不足：{ev.get('verdict_reason', '')}"
            result.watchlist.append(item)
            result.gate_log.append({"code": code, "gate": "news_evidence", "passed": False, "reason": item["watch_reason"]})
            return

        # 10. LLM 复核（若启用）
        review = llm_review_map.get(code) if llm_review_map else None
        if self.cfg.get("llm", {}).get("enable_review", False):
            if not review or not review.get("passed"):
                item["watch_reason"] = "LLM 复核未通过"
                result.watchlist.append(item)
                result.gate_log.append({"code": code, "gate": "llm_review", "passed": False, "reason": item["watch_reason"]})
                return
            item["llm_review"] = review

        item["gate_status"] = "final"
        result.final_recommendations.append(item)
        result.gate_log.append({"code": code, "gate": "final", "passed": True})

    def _judge_trend_only(
        self,
        code: str,
        cand: Optional[StockCandidate],
        item: dict[str, Any],
        result: GateResult,
        watch_codes: set[str],
        rejected_codes: set[str],
    ) -> None:
        if code in rejected_codes:
            item["gate_status"] = "rejected"
            item["reject_reason"] = next((i["reason"] for i in self.guard.rejected if i["code"] == code), "QuantGuard 拒绝")
            result.rejected.append(item)
            result.gate_log.append({"code": code, "gate": "guard", "passed": False, "reason": item["reject_reason"]})
            return

        if code in watch_codes:
            item["gate_status"] = "watch"
            item["watch_reason"] = "QuantGuard 护栏观察"
        else:
            item["gate_status"] = "watch"
            item["watch_reason"] = "旧趋势扫描补充候选，无策略池信号"
        result.watchlist.append(item)
        result.gate_log.append({"code": code, "gate": "strategy_signal", "passed": False, "reason": item["watch_reason"]})

    def _phase_allowed(self, strategy_name: str, sig: StrategySignal) -> bool:
        if not sig.respect_market_phase and sig.allow_when_phase_forbids:
            return True
        # 策略名与阶段 forbidden 做模糊匹配
        for forbidden in self.forbidden_strategies:
            if forbidden in strategy_name or strategy_name in forbidden:
                return False
        return True

    def _strategy_requires_fund_flow(self, strategy_name: str) -> bool:
        s = str(strategy_name or "").lower()
        return any(token in s for token in ("fund", "flow", "资金", "主力", "北向"))

    def _build_item(self, sig: StrategySignal, cand: Optional[StockCandidate]) -> dict[str, Any]:
        item = sig.to_dict()
        item["gate_status"] = "pending"
        if cand:
            item["close"] = cand.close
            item["pct_change"] = cand.pct_change
            item["turnover_rate"] = cand.turnover_rate
            item["rps"] = cand.rps
            item["rps_mode"] = cand.rps_mode
            item["fund_inflow_days"] = cand.fund_inflow_days
            item["fund_flow_status"] = cand.fund_flow_status
            item["is_watchlist"] = cand.is_watchlist
        return item

    def _dedup(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for item in items:
            code = item["code"]
            if code not in best or item.get("score", 0) > best[code].get("score", 0):
                best[code] = item
        return sorted(best.values(), key=lambda x: x.get("score", 0), reverse=True)
