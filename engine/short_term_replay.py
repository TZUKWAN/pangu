"""Causal 1-3 day replay and acceptance contract for formal recommendations.

The replay deliberately differs from the legacy close-to-close evaluator:

* a post-close signal can only enter from the next trading day;
* an entry must be executable inside the declared entry zone;
* stop-loss wins when a daily bar touches both stop and profit targets;
* partial exits, slippage, commission and sell-side stamp duty are charged;
* missing exact-date news/sentiment/theme context is reported, never imputed;
* an 85% claim requires enough trades and dates, not an empty or tiny sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from .frame_utils import safe_float
from .exit_engine import ShortTermExitEngine


@dataclass(frozen=True)
class ShortTermReplayConfig:
    entry_window_days: int = 1
    capital_per_trade: float = 100_000.0
    lot_size: int = 100
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    slippage_bps: float = 10.0
    target_win_rate: float = 0.85
    min_trades: int = 100
    min_trade_dates: int = 30
    min_execution_rate: float = 0.25
    min_profit_factor: float = 1.20
    max_drawdown_limit: float = 0.15
    require_causal_context: bool = True

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None = None) -> "ShortTermReplayConfig":
        data = dict(cfg or {})
        section = data.get("short_term_replay", data) or {}
        return cls(
            entry_window_days=max(1, min(3, int(section.get("entry_window_days", 1)))),
            capital_per_trade=max(1_000.0, float(section.get("capital_per_trade", 100_000.0))),
            lot_size=max(1, int(section.get("lot_size", 100))),
            commission_rate=max(0.0, float(section.get("commission_rate", 0.0003))),
            min_commission=max(0.0, float(section.get("min_commission", 5.0))),
            stamp_duty_rate=max(0.0, float(section.get("stamp_duty_rate", 0.0005))),
            slippage_bps=max(0.0, float(section.get("slippage_bps", 10.0))),
            target_win_rate=min(1.0, max(0.0, float(section.get("target_win_rate", 0.85)))),
            min_trades=max(1, int(section.get("min_trades", 100))),
            min_trade_dates=max(1, int(section.get("min_trade_dates", 30))),
            min_execution_rate=min(1.0, max(0.0, float(section.get("min_execution_rate", 0.25)))),
            min_profit_factor=max(0.0, float(section.get("min_profit_factor", 1.20))),
            max_drawdown_limit=min(1.0, max(0.0, float(section.get("max_drawdown_limit", 0.15)))),
            require_causal_context=bool(section.get("require_causal_context", True)),
        )


@dataclass
class TradeFill:
    date: str
    action: str
    shares: int
    raw_price: float
    execution_price: float
    fee: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "action": self.action,
            "shares": self.shares,
            "raw_price": round(self.raw_price, 4),
            "execution_price": round(self.execution_price, 4),
            "fee": round(self.fee, 4),
            "reason": self.reason,
        }


@dataclass
class ReplayOutcome:
    signal_date: str
    code: str
    status: str
    reason: str
    entry_date: str = ""
    exit_date: str = ""
    shares: int = 0
    holding_days: int = 0
    first_target_taken: bool = False
    gross_return: float | None = None
    net_return: float | None = None
    net_pnl: float | None = None
    max_favorable_excursion: float | None = None
    max_adverse_excursion: float | None = None
    causal_context_complete: bool = False
    fills: list[TradeFill] = field(default_factory=list)

    @property
    def win(self) -> bool | None:
        if self.status != "closed" or self.net_pnl is None:
            return None
        return self.net_pnl > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_date": self.signal_date,
            "code": self.code,
            "status": self.status,
            "reason": self.reason,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "shares": self.shares,
            "holding_days": self.holding_days,
            "first_target_taken": self.first_target_taken,
            "gross_return": round(self.gross_return, 6) if self.gross_return is not None else None,
            "net_return": round(self.net_return, 6) if self.net_return is not None else None,
            "net_pnl": round(self.net_pnl, 2) if self.net_pnl is not None else None,
            "max_favorable_excursion": (
                round(self.max_favorable_excursion, 6)
                if self.max_favorable_excursion is not None else None
            ),
            "max_adverse_excursion": (
                round(self.max_adverse_excursion, 6)
                if self.max_adverse_excursion is not None else None
            ),
            "causal_context_complete": self.causal_context_complete,
            "win": self.win,
            "fills": [fill.to_dict() for fill in self.fills],
        }


class ShortTermReplayEngine:
    """Replay one formal recommendation using only data available after its signal."""

    REQUIRED_CONTEXT_FIELDS = {
        "news_evidence", "market_phase", "current_temperature", "theme_invalidated",
        "news_context_exact", "market_context_exact", "theme_status_known",
    }

    def __init__(self, cfg: ShortTermReplayConfig | Mapping[str, Any] | None = None) -> None:
        raw = cfg if isinstance(cfg, Mapping) else {}
        self.cfg = cfg if isinstance(cfg, ShortTermReplayConfig) else ShortTermReplayConfig.from_dict(cfg)
        # 透传原始配置给退出引擎（short_term_agent.horizon_days / trend_break_buffer 等）
        self.exit_engine = ShortTermExitEngine(self._extract_exit_cfg(raw))

    @staticmethod
    def _extract_exit_cfg(raw: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            return {}
        merged = dict(raw)
        # ShortTermReplayConfig.from_dict 支持把 short_term_replay 段直接当顶层；
        # 退出引擎的 horizon_days 属 short_term_agent 段，避免被前者覆盖。
        replay_section = raw.get("short_term_replay")
        if isinstance(replay_section, Mapping):
            merged.update({k: v for k, v in replay_section.items() if k not in merged})
        return merged

    def replay(
        self,
        recommendation: Mapping[str, Any],
        kline: pd.DataFrame,
        *,
        signal_date: str,
        daily_context: Mapping[str, Mapping[str, Any]] | None = None,
        causal_signal_evidence: bool = False,
    ) -> ReplayOutcome:
        code = str(recommendation.get("code") or "").zfill(6)
        base = ReplayOutcome(signal_date=str(signal_date), code=code, status="invalid", reason="")
        frame = self._normalize_kline(kline)
        if frame is None:
            base.reason = "K线缺少日期或OHLC字段"
            return base

        ee = recommendation.get("entry_exit") or {}
        plan = recommendation.get("exit_plan") or ee.get("exit_plan") or {}
        entry_plan = recommendation.get("entry_plan") or ee.get("entry_plan") or {}
        plan_error = self._validate_plan(plan)
        if plan_error:
            base.reason = plan_error
            return base

        future = frame[frame["_date"] > str(signal_date)].reset_index(drop=True)
        if future.empty:
            base.status = "pending"
            base.reason = "信号日后尚无交易日K线"
            return base

        entry_idx = -1
        entry_raw = 0.0
        for idx, row in future.head(self.cfg.entry_window_days).iterrows():
            candidate_price = self._entry_price(entry_plan, plan, row)
            if candidate_price > 0:
                entry_idx = int(idx)
                entry_raw = candidate_price
                break
        if entry_idx < 0:
            if len(future) < self.cfg.entry_window_days:
                base.status = "pending"
                base.reason = "入场观察窗口尚未走完"
            else:
                base.status = "no_entry"
                base.reason = "下一交易日未触发条件买点或价格超出入场区间"
            return base

        slip = self.cfg.slippage_bps / 10_000.0
        entry_exec = entry_raw * (1.0 + slip)
        shares = int(self.cfg.capital_per_trade / entry_exec / self.cfg.lot_size) * self.cfg.lot_size
        if shares < self.cfg.lot_size:
            base.reason = "资金规模不足以买入一个交易单位"
            return base
        entry_row = future.iloc[entry_idx]
        entry_date = str(entry_row["_date"])
        buy_fee = self._commission(shares * entry_exec)
        fills = [TradeFill(entry_date, "buy", shares, entry_raw, entry_exec, buy_fee, "条件买点触发")]
        remaining = shares
        active_stop = safe_float(plan.get("initial_stop"), 0.0)
        first_taken = False
        gross_sale_value = 0.0
        net_sale_value = 0.0
        raw_buy_value = shares * entry_raw
        net_buy_cost = shares * entry_exec + buy_fee
        held_highs: list[float] = []
        held_lows: list[float] = []
        contexts = daily_context or {}
        context_complete = bool(causal_signal_evidence)
        exit_reason = ""
        exit_date = ""
        holding_days_count = 0

        max_days = int(plan.get("max_holding_days") or 3)
        held_rows = future.iloc[entry_idx: entry_idx + max_days]
        for held_index, (_, row) in enumerate(held_rows.iterrows(), start=1):
            holding_days_count = held_index
            date = str(row["_date"])
            held_highs.append(float(row["_high"]))
            held_lows.append(float(row["_low"]))
            external = dict(contexts.get(date) or {})
            if (
                not self.REQUIRED_CONTEXT_FIELDS.issubset(external)
                or not bool(external.get("news_context_exact"))
                or not bool(external.get("market_context_exact"))
                or not bool(external.get("theme_status_known"))
            ):
                context_complete = False
            technical = self._technical_context(frame, date)
            context = {
                **technical,
                **external,
                "open": float(row["_open"]),
                "high": float(row["_high"]),
                "low": float(row["_low"]),
                "close": float(row["_close"]),
                "active_stop": active_stop,
                "first_target_taken": first_taken,
                "days_held": held_index,
            }
            decision = self.exit_engine.evaluate(plan, context)

            # If both targets are reached after stop checks, replay both declared fills.
            if decision.rule_type == "final_target" and not first_taken:
                first_target = safe_float(plan.get("first_target"), 0.0)
                half = self._half_lot(remaining)
                if half > 0 and first_target > 0:
                    sale = self._sell_fill(date, half, first_target, "第一目标减半")
                    fills.append(sale)
                    remaining -= half
                    gross_sale_value += half * first_target
                    net_sale_value += half * sale.execution_price - sale.fee
                    first_taken = True
                    active_stop = entry_exec
                if remaining > 0:
                    final_target = safe_float(plan.get("final_target"), decision.execution_price or 0.0)
                    sale = self._sell_fill(date, remaining, final_target, "最终目标清仓")
                    fills.append(sale)
                    gross_sale_value += remaining * final_target
                    net_sale_value += remaining * sale.execution_price - sale.fee
                    remaining = 0
                    exit_reason = "final_target"
                    exit_date = date
                    break

            if decision.action == "reduce_half" and not first_taken:
                half = self._half_lot(remaining)
                if half > 0:
                    raw_price = safe_float(decision.execution_price, 0.0)
                    sale = self._sell_fill(date, half, raw_price, decision.rule_type)
                    fills.append(sale)
                    remaining -= half
                    gross_sale_value += half * raw_price
                    net_sale_value += half * sale.execution_price - sale.fee
                    first_taken = True
                    active_stop = entry_exec
                continue

            if decision.action == "exit_all" and remaining > 0:
                raw_price = safe_float(decision.execution_price, float(row["_close"]))
                sale = self._sell_fill(date, remaining, raw_price, decision.rule_type)
                fills.append(sale)
                gross_sale_value += remaining * raw_price
                net_sale_value += remaining * sale.execution_price - sale.fee
                remaining = 0
                exit_reason = decision.rule_type
                exit_date = date
                break

        if remaining > 0:
            base.status = "pending"
            base.reason = "尚未达到退出条件或最大持有交易日"
            base.entry_date = entry_date
            base.shares = shares
            base.holding_days = len(held_rows)
            base.first_target_taken = first_taken
            base.causal_context_complete = context_complete
            base.fills = fills
            return base

        gross_return = (gross_sale_value - raw_buy_value) / raw_buy_value
        net_pnl = net_sale_value - net_buy_cost
        net_return = net_pnl / net_buy_cost
        base.status = "closed"
        base.reason = exit_reason
        base.entry_date = entry_date
        base.exit_date = exit_date
        base.shares = shares
        base.holding_days = holding_days_count
        base.first_target_taken = first_taken
        base.gross_return = gross_return
        base.net_return = net_return
        base.net_pnl = net_pnl
        base.max_favorable_excursion = max(held_highs) / entry_exec - 1.0 if held_highs else None
        base.max_adverse_excursion = min(held_lows) / entry_exec - 1.0 if held_lows else None
        base.causal_context_complete = context_complete
        base.fills = fills
        return base

    def acceptance(self, outcomes: list[ReplayOutcome]) -> dict[str, Any]:
        closed = [item for item in outcomes if item.status == "closed" and item.net_return is not None]
        wins = [item for item in closed if item.win]
        signal_count = len(outcomes)
        executed_count = len(closed)
        win_rate = len(wins) / executed_count if executed_count else 0.0
        lower = self._wilson_lower(len(wins), executed_count)
        returns = [float(item.net_return or 0.0) for item in closed]
        gains = sum(value for value in returns if value > 0)
        losses = abs(sum(value for value in returns if value < 0))
        profit_factor = gains / losses if losses > 0 else (math.inf if gains > 0 else 0.0)
        max_drawdown = self._max_drawdown(closed)
        trade_dates = {item.signal_date for item in closed}
        execution_rate = executed_count / signal_count if signal_count else 0.0
        causal_coverage = (
            sum(1 for item in closed if item.causal_context_complete) / executed_count
            if executed_count else 0.0
        )
        blockers: list[str] = []
        if executed_count < self.cfg.min_trades:
            blockers.append(f"成交样本 {executed_count} < {self.cfg.min_trades}")
        if len(trade_dates) < self.cfg.min_trade_dates:
            blockers.append(f"独立交易日 {len(trade_dates)} < {self.cfg.min_trade_dates}")
        if execution_rate < self.cfg.min_execution_rate:
            blockers.append(
                f"条件买点成交覆盖率 {execution_rate:.1%} < {self.cfg.min_execution_rate:.1%}"
            )
        if self.cfg.require_causal_context and causal_coverage < 1.0:
            blockers.append(f"精确日期新闻/情绪/题材退出上下文覆盖率仅 {causal_coverage:.1%}")
        if win_rate < self.cfg.target_win_rate:
            blockers.append(f"实测成功率 {win_rate:.1%} < {self.cfg.target_win_rate:.1%}")
        if lower < self.cfg.target_win_rate:
            blockers.append(f"95% Wilson 下界 {lower:.1%} < {self.cfg.target_win_rate:.1%}")
        avg_return = sum(returns) / len(returns) if returns else 0.0
        if avg_return <= 0:
            blockers.append(f"平均净收益 {avg_return:.2%} 不为正")
        if profit_factor < self.cfg.min_profit_factor:
            blockers.append(f"盈亏比因子 {profit_factor:.2f} < {self.cfg.min_profit_factor:.2f}")
        if max_drawdown > self.cfg.max_drawdown_limit:
            blockers.append(f"最大回撤 {max_drawdown:.1%} > {self.cfg.max_drawdown_limit:.1%}")

        sample_blocked = executed_count < self.cfg.min_trades or len(trade_dates) < self.cfg.min_trade_dates
        context_blocked = self.cfg.require_causal_context and causal_coverage < 1.0
        status = "verified" if not blockers else (
            "insufficient_sample" if sample_blocked else (
                "incomplete_causal_context" if context_blocked else "target_not_met"
            )
        )
        return {
            "verification_status": status,
            "verified_success_rate_85": status == "verified",
            "target_win_rate": self.cfg.target_win_rate,
            "signal_count": signal_count,
            "executed_count": executed_count,
            "no_entry_count": sum(1 for item in outcomes if item.status == "no_entry"),
            "pending_count": sum(1 for item in outcomes if item.status == "pending"),
            "invalid_count": sum(1 for item in outcomes if item.status == "invalid"),
            "independent_trade_dates": len(trade_dates),
            "execution_rate": round(execution_rate, 6),
            "observed_win_rate": round(win_rate, 6),
            "wilson_95_lower": round(lower, 6),
            "average_net_return": round(avg_return, 6),
            "profit_factor": round(profit_factor, 6) if math.isfinite(profit_factor) else None,
            "max_drawdown": round(max_drawdown, 6),
            "causal_context_coverage": round(causal_coverage, 6),
            "blockers": blockers,
        }

    def _entry_price(self, entry_plan: Mapping[str, Any], plan: Mapping[str, Any], row: pd.Series) -> float:
        trigger = safe_float(entry_plan.get("trigger_price"), safe_float(plan.get("entry_price"), 0.0))
        if trigger <= 0:
            return 0.0
        zone = entry_plan.get("ideal_entry_zone") or [trigger * 0.99, trigger * 1.01]
        try:
            lower, upper = sorted((float(zone[0]), float(zone[1])))
        except (TypeError, ValueError, IndexError):
            lower, upper = trigger * 0.99, trigger * 1.01
        open_price = float(row["_open"])
        high = float(row["_high"])
        low = float(row["_low"])
        style = str(entry_plan.get("entry_style") or recommendation_style(entry_plan) or "")
        if style == "breakout_confirm":
            if open_price > upper:
                return 0.0
            if trigger <= open_price <= upper:
                return open_price
            return trigger if high >= trigger else 0.0
        if lower <= open_price <= upper:
            return open_price
        if open_price < lower or open_price > upper:
            return 0.0
        return trigger if low <= trigger <= high else 0.0

    @staticmethod
    def _normalize_kline(kline: pd.DataFrame) -> pd.DataFrame | None:
        if kline is None or len(kline) == 0:
            return None
        names = {
            "date": ["日期", "date", "trade_date"],
            "open": ["开盘", "open"],
            "high": ["最高", "high"],
            "low": ["最低", "low"],
            "close": ["收盘", "close"],
            "volume": ["成交量", "volume", "vol"],
        }
        columns: dict[str, str | None] = {}
        lower = {str(col).lower(): str(col) for col in kline.columns}
        for key, candidates in names.items():
            columns[key] = next(
                (candidate for candidate in candidates if candidate in kline.columns),
                next((lower[candidate.lower()] for candidate in candidates if candidate.lower() in lower), None),
            )
        if any(columns[key] is None for key in ("date", "open", "high", "low", "close")):
            return None
        frame = kline.copy()
        frame["_date"] = frame[columns["date"]].astype(str).str.replace("-", "", regex=False).str[:8]
        for key in ("open", "high", "low", "close"):
            frame[f"_{key}"] = pd.to_numeric(frame[columns[key]], errors="coerce")
        if columns["volume"]:
            frame["_volume"] = pd.to_numeric(frame[columns["volume"]], errors="coerce")
        else:
            frame["_volume"] = float("nan")
        frame = frame.dropna(subset=["_open", "_high", "_low", "_close"])
        frame = frame[(frame["_open"] > 0) & (frame["_high"] > 0) & (frame["_low"] > 0) & (frame["_close"] > 0)]
        return frame.sort_values("_date").drop_duplicates("_date", keep="last").reset_index(drop=True)

    @staticmethod
    def _validate_plan(plan: Mapping[str, Any]) -> str:
        if not isinstance(plan, Mapping) or not plan:
            return "缺少结构化退出计划"
        entry = safe_float(plan.get("entry_price"), 0.0)
        stop = safe_float(plan.get("initial_stop"), 0.0)
        first = safe_float(plan.get("first_target"), 0.0)
        final = safe_float(plan.get("final_target"), 0.0)
        max_days = int(plan.get("max_holding_days") or 0)
        if entry <= 0 or stop <= 0 or stop >= entry:
            return "退出计划止损无效"
        if first <= entry or final < first:
            return "退出计划止盈目标无效"
        if not 1 <= max_days <= 3:
            return "退出计划持有期必须为1-3个交易日"
        if plan.get("conservative_same_day_order") != "stop_first":
            return "退出计划缺少同日止损优先约定"
        return ""

    def _technical_context(self, frame: pd.DataFrame, date: str) -> dict[str, Any]:
        history = frame[frame["_date"] <= date]
        closes = history["_close"]
        volumes = history["_volume"]
        ma10 = float(closes.tail(10).mean()) if len(closes) >= 10 else 0.0
        ma20 = float(closes.tail(20).mean()) if len(closes) >= 20 else 0.0
        volume_avg = float(volumes.tail(5).mean()) if len(volumes.dropna()) >= 5 else 0.0
        volume_now = safe_float(volumes.iloc[-1], 0.0) if len(volumes) else 0.0
        return {
            "ma10": ma10,
            "ma20": ma20,
            "volume_ratio": volume_now / volume_avg if volume_avg > 0 else 0.0,
        }

    def _commission(self, amount: float) -> float:
        return max(self.cfg.min_commission, amount * self.cfg.commission_rate)

    def _sell_fill(self, date: str, shares: int, raw_price: float, reason: str) -> TradeFill:
        slip = self.cfg.slippage_bps / 10_000.0
        execution = raw_price * (1.0 - slip)
        amount = shares * execution
        fee = self._commission(amount) + amount * self.cfg.stamp_duty_rate
        return TradeFill(date, "sell", shares, raw_price, execution, fee, reason)

    def _half_lot(self, shares: int) -> int:
        return int(shares / 2 / self.cfg.lot_size) * self.cfg.lot_size

    @staticmethod
    def _wilson_lower(wins: int, total: int, z: float = 1.959963984540054) -> float:
        if total <= 0:
            return 0.0
        p = wins / total
        denominator = 1.0 + z * z / total
        centre = p + z * z / (2.0 * total)
        margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        return max(0.0, (centre - margin) / denominator)

    @staticmethod
    def _max_drawdown(outcomes: list[ReplayOutcome]) -> float:
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for item in sorted(outcomes, key=lambda value: (value.exit_date, value.signal_date, value.code)):
            equity *= 1.0 + float(item.net_return or 0.0)
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        return max_drawdown


def recommendation_style(entry_plan: Mapping[str, Any]) -> str:
    """Compatibility helper for snapshots that nest style under a legacy key."""
    return str(entry_plan.get("style") or "")
