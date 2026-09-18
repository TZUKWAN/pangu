"""Short-term multi-layer exit planning and deterministic sell decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExitRule:
    rule_type: str
    action: str
    condition: str
    trigger_price: float | None = None
    priority: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_type": self.rule_type,
            "action": self.action,
            "condition": self.condition,
            "trigger_price": round(self.trigger_price, 2) if self.trigger_price is not None else None,
            "priority": self.priority,
        }


@dataclass
class ExitPlan:
    entry_price: float
    initial_stop: float
    first_target: float | None
    final_target: float | None
    trailing_reference: float | None
    max_holding_days: int
    sentiment_exit_drop: float
    rules: list[ExitRule] = field(default_factory=list)
    conservative_same_day_order: str = "stop_first"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_price": round(self.entry_price, 2),
            "initial_stop": round(self.initial_stop, 2),
            "first_target": round(self.first_target, 2) if self.first_target is not None else None,
            "final_target": round(self.final_target, 2) if self.final_target is not None else None,
            "trailing_reference": round(self.trailing_reference, 2) if self.trailing_reference is not None else None,
            "max_holding_days": self.max_holding_days,
            "sentiment_exit_drop": round(self.sentiment_exit_drop, 1),
            "conservative_same_day_order": self.conservative_same_day_order,
            "rules": [rule.to_dict() for rule in sorted(self.rules, key=lambda item: item.priority)],
        }


@dataclass
class ExitDecision:
    action: str = "hold"  # hold / reduce_half / exit_all
    reason: str = "尚未触发卖出条件"
    rule_type: str = "none"
    execution_price: float | None = None
    new_stop: float | None = None
    priority: int = 999

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "rule_type": self.rule_type,
            "execution_price": round(self.execution_price, 2) if self.execution_price is not None else None,
            "new_stop": round(self.new_stop, 2) if self.new_stop is not None else None,
            "priority": self.priority,
        }


class ShortTermExitEngine:
    """Build and evaluate price, time, sentiment, news and trend exits."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        scfg = cfg.get("short_term_agent", cfg) or {}
        self.max_holding_days = max(1, int(scfg.get("horizon_days", 3)))
        self.sentiment_exit_drop = float(scfg.get("sentiment_exit_drop", 15.0))
        # 趋势破位缓冲：收盘需低于 MA20×(1-buffer) 才判破位（0=原行为）。
        # 1-3 日短线持仓对均线天然敏感，适度缓冲可减少噪音止损。
        self.trend_break_buffer = max(0.0, float(scfg.get("trend_break_buffer", 0.0)))

    def build_plan(
        self,
        *,
        entry_price: float,
        stop_price: float,
        take_profit_prices: list[float] | None = None,
        trailing_reference: float | None = None,
    ) -> ExitPlan:
        targets = sorted(price for price in (take_profit_prices or []) if price > entry_price)
        first_target = targets[0] if targets else None
        final_target = targets[-1] if len(targets) > 1 else first_target
        plan = ExitPlan(
            entry_price=entry_price,
            initial_stop=stop_price,
            first_target=first_target,
            final_target=final_target,
            trailing_reference=trailing_reference if trailing_reference and trailing_reference > 0 else None,
            max_holding_days=self.max_holding_days,
            sentiment_exit_drop=self.sentiment_exit_drop,
        )
        plan.rules = [
            ExitRule("hard_stop", "exit_all", f"开盘或盘中触及硬止损 {stop_price:.2f}", stop_price, 10),
            ExitRule("news_invalidation", "exit_all", "出现立案、减持、业绩变脸、合同终止等直接负面事件", None, 15),
            ExitRule("market_retreat", "exit_all", f"市场进入冰点/退潮，或情绪温度较入场下降≥{self.sentiment_exit_drop:.0f}点", None, 20),
            ExitRule("theme_invalidation", "exit_all", "题材催化被证伪、核心龙头转弱或板块退出强势序列", None, 25),
            ExitRule("trend_break", "exit_all", "收盘跌破MA10且放量，或有效跌破MA20", None, 30),
        ]
        if first_target is not None:
            plan.rules.append(ExitRule(
                "first_target", "reduce_half",
                f"触及第一目标 {first_target:.2f}，减半并把止损抬到成本",
                first_target, 40,
            ))
        if final_target is not None:
            plan.rules.append(ExitRule(
                "final_target", "exit_all", f"触及最终目标 {final_target:.2f}", final_target, 45,
            ))
        if trailing_reference and trailing_reference > 0:
            plan.rules.append(ExitRule(
                "trailing_stop", "exit_all",
                f"盈利后收盘跌破跟踪参考 {trailing_reference:.2f}", trailing_reference, 50,
            ))
        plan.rules.append(ExitRule(
            "time_stop", "exit_all",
            f"持有满 {self.max_holding_days} 个交易日仍未兑现，收盘退出", None, 60,
        ))
        return plan

    def evaluate(self, plan: ExitPlan | dict[str, Any], context: dict[str, Any]) -> ExitDecision:
        if isinstance(plan, dict):
            plan = self._plan_from_dict(plan)
        open_price = self._num(context.get("open"))
        high = self._num(context.get("high"))
        low = self._num(context.get("low"))
        close = self._num(context.get("close"))
        entry = plan.entry_price
        stop = self._num(context.get("active_stop")) or plan.initial_stop

        # Conservative backtest ordering when one daily bar touches both sides.
        if open_price > 0 and open_price <= stop:
            return ExitDecision("exit_all", "跳空低于硬止损，按开盘价退出", "hard_stop", open_price, priority=10)
        if low > 0 and low <= stop:
            return ExitDecision("exit_all", "盘中触及硬止损", "hard_stop", stop, priority=10)

        news = context.get("news_evidence") or {}
        if news.get("risk_events") or str(news.get("sentiment_label") or "") == "bearish":
            return ExitDecision("exit_all", "直接负面新闻/风险事件使原催化失效", "news_invalidation", close or None, priority=15)

        phase = str(context.get("market_phase") or "")
        entry_temp = self._num(context.get("entry_temperature"))
        current_temp = self._num(context.get("current_temperature"))
        if phase in {"冰点期", "退潮期"} or (
            entry_temp > 0 and current_temp > 0
            and entry_temp - current_temp >= plan.sentiment_exit_drop
        ):
            return ExitDecision("exit_all", "市场情绪进入冰点/退潮或较入场显著转弱", "market_retreat", close or None, priority=20)

        if bool(context.get("theme_invalidated")):
            return ExitDecision("exit_all", "题材催化证伪或板块核心转弱", "theme_invalidation", close or None, priority=25)

        ma10 = self._num(context.get("ma10"))
        ma20 = self._num(context.get("ma20"))
        volume_ratio = self._num(context.get("volume_ratio"))
        ma20_stop_level = ma20 * (1 - self.trend_break_buffer) if ma20 > 0 else 0.0
        if close > 0 and (
            (ma20_stop_level > 0 and close < ma20_stop_level)
            or (ma10 > 0 and close < ma10 and volume_ratio >= 1.2)
        ):
            return ExitDecision("exit_all", "趋势破位：跌破MA20或放量跌破MA10", "trend_break", close, priority=30)

        if plan.final_target is not None and high >= plan.final_target:
            return ExitDecision("exit_all", "触及最终止盈目标", "final_target", plan.final_target, priority=45)
        if plan.first_target is not None and high >= plan.first_target and not bool(context.get("first_target_taken")):
            return ExitDecision(
                "reduce_half", "触及第一止盈目标，减半并把剩余仓位止损抬到成本",
                "first_target", plan.first_target, new_stop=entry, priority=40,
            )

        trailing = self._num(context.get("trailing_reference")) or (plan.trailing_reference or 0.0)
        unrealized_return = (close / entry - 1.0) if close > 0 and entry > 0 else 0.0
        if trailing > 0 and close > 0 and close < trailing and unrealized_return > 0:
            return ExitDecision("exit_all", "盈利后跌破移动止盈参考", "trailing_stop", close, priority=50)

        days_held = int(context.get("days_held") or 0)
        if days_held >= plan.max_holding_days:
            return ExitDecision("exit_all", "达到短期最大持有天数，按收盘退出", "time_stop", close or None, priority=60)

        return ExitDecision(new_stop=stop)

    @staticmethod
    def _num(value: Any) -> float:
        try:
            number = float(value)
            return number if number == number else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _plan_from_dict(self, data: dict[str, Any]) -> ExitPlan:
        return ExitPlan(
            entry_price=self._num(data.get("entry_price")),
            initial_stop=self._num(data.get("initial_stop")),
            first_target=self._num(data.get("first_target")) or None,
            final_target=self._num(data.get("final_target")) or None,
            trailing_reference=self._num(data.get("trailing_reference")) or None,
            max_holding_days=int(data.get("max_holding_days") or self.max_holding_days),
            sentiment_exit_drop=self._num(data.get("sentiment_exit_drop")) or self.sentiment_exit_drop,
            conservative_same_day_order=str(data.get("conservative_same_day_order") or "stop_first"),
        )
