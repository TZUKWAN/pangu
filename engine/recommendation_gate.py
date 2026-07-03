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

from .data_loader import DataLoader
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_recommendations": self.final_recommendations,
            "watchlist": self.watchlist,
            "rejected": self.rejected,
            "gate_log": self.gate_log,
        }


class RecommendationGate:
    def __init__(
        self,
        dl: DataLoader,
        guard_result: GuardResult,
        market_phase: dict[str, Any],
        cfg: dict[str, Any] | None = None,
        recommendation_allowed: bool = True,
    ) -> None:
        self.dl = dl
        self.guard = guard_result
        self.phase = market_phase
        self.cfg = cfg or {}
        self.entry_engine = EntryExitEngine(dl, self.cfg.get("entry_exit", {}))

        self.allowed_strategies = set(self.phase.get("allowed_strategies", []))
        self.forbidden_strategies = set(self.phase.get("forbidden_strategies", []))
        self.recommendation_allowed = recommendation_allowed and self.phase.get("recommendation_allowed", True)

    def pass_gate(
        self,
        pooled_signals: dict[str, list[StrategySignal]],
        candidate_map: dict[str, StockCandidate],
        llm_review_map: Optional[dict[str, dict[str, Any]]] = None,
    ) -> GateResult:
        result = GateResult()
        if not self.recommendation_allowed:
            result.gate_log.append({"gate": "global", "passed": False, "reason": "全局 recommendation_allowed=False"})

        kept_codes = {c.code for c in self.guard.kept}
        watch_codes = {c.code for c in self.guard.watch}
        rejected_codes = {item["code"] for item in self.guard.rejected}

        for strategy_name, signals in pooled_signals.items():
            for sig in signals:
                code = sig.code
                cand = candidate_map.get(code)
                item = self._build_item(sig, cand)

                # 1. 市场阶段
                if not self._phase_allowed(strategy_name, sig):
                    item["gate_status"] = "watch"
                    item["watch_reason"] = f"当前阶段 {self.phase.get('market_phase')} 禁止策略 {strategy_name}"
                    result.watchlist.append(item)
                    result.gate_log.append({"code": code, "gate": "phase", "passed": False, "reason": item["watch_reason"]})
                    continue

                # 2. QuantGuard
                if code in rejected_codes:
                    item["gate_status"] = "rejected"
                    item["reject_reason"] = next((i["reason"] for i in self.guard.rejected if i["code"] == code), "QuantGuard 拒绝")
                    result.rejected.append(item)
                    result.gate_log.append({"code": code, "gate": "guard", "passed": False, "reason": item["reject_reason"]})
                    continue

                is_watch = code in watch_codes

                # 3. 数据真实性：正式推荐必须有真实 RPS
                if cand and cand.rps_mode != "real" and not is_watch:
                    item["gate_status"] = "watch"
                    item["watch_reason"] = f"RPS 模式为 {cand.rps_mode}"
                    result.watchlist.append(item)
                    result.gate_log.append({"code": code, "gate": "rps_real", "passed": False, "reason": item["watch_reason"]})
                    continue

                # 4. 资金流确认
                if cand and cand.fund_flow_status != "ok" and not is_watch:
                    item["gate_status"] = "watch"
                    item["watch_reason"] = f"资金流状态 {cand.fund_flow_status}"
                    result.watchlist.append(item)
                    result.gate_log.append({"code": code, "gate": "fund_flow", "passed": False, "reason": item["watch_reason"]})
                    continue

                # 5. EntryExit 可执行
                try:
                    ee = self.entry_engine.compute(cand) if cand else None
                    if ee:
                        item["entry_exit"] = ee
                    else:
                        item["watch_reason"] = "买卖点计算失败"
                        result.watchlist.append(item)
                        result.gate_log.append({"code": code, "gate": "entry_exit", "passed": False, "reason": item["watch_reason"]})
                        continue
                except Exception as exc:  # noqa: BLE001
                    item["watch_reason"] = f"买卖点异常: {exc}"
                    result.watchlist.append(item)
                    result.gate_log.append({"code": code, "gate": "entry_exit", "passed": False, "reason": item["watch_reason"]})
                    continue

                # 6. LLM 复核（若启用）
                review = llm_review_map.get(code) if llm_review_map else None
                if self.cfg.get("llm", {}).get("enable_review", False):
                    if not review or not review.get("passed"):
                        item["watch_reason"] = "LLM 复核未通过"
                        result.watchlist.append(item)
                        result.gate_log.append({"code": code, "gate": "llm_review", "passed": False, "reason": item["watch_reason"]})
                        continue
                    item["llm_review"] = review

                item["gate_status"] = "final"
                result.final_recommendations.append(item)
                result.gate_log.append({"code": code, "gate": "final", "passed": True})

        # 去重：同一 code 多个策略取最高 score
        result.final_recommendations = self._dedup(result.final_recommendations)
        result.watchlist = self._dedup(result.watchlist)
        result.rejected = self._dedup(result.rejected)
        return result

    def _phase_allowed(self, strategy_name: str, sig: StrategySignal) -> bool:
        if not sig.respect_market_phase and sig.allow_when_phase_forbids:
            return True
        # 策略名与阶段 forbidden 做模糊匹配
        for forbidden in self.forbidden_strategies:
            if forbidden in strategy_name or strategy_name in forbidden:
                return False
        return True

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
