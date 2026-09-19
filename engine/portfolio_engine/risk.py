"""Pre-trade risk engine: every target must pass every applicable check.

Units: daily_pnl and drawdown are fractions of equity (e.g. -0.025 = -2.5%);
adv (average daily turnover) and notionals are in currency units.

Side policy:
  daily_loss_hard  blocks ALL orders (including sells);
  daily_loss_soft  blocks NEW BUY orders but always allows SELL (exits must
                   stay possible to de-risk);
  drawdown cap     blocks BUY orders, allows SELL;
  market veto / DISABLED mode block everything.
ExecutionMode LIVE needs contract checks elsewhere (compliance, kill switch,
broker); this engine only treats DISABLED as a veto.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

from ..contracts import ExecutionMode, PortfolioTarget, RiskDecision

_EPS = 1e-9


@dataclass
class RiskEngineConfig:
    max_order_notional: float = 100_000.0
    max_weight_per_stock: float = 0.10
    min_liquidity_amount: float = 5_000_000.0   # day-amount floor
    participation_cap: float = 0.05             # order value <= cap * adv
    daily_loss_soft: float = -0.02              # fraction of equity
    daily_loss_hard: float = -0.04              # fraction of equity
    max_drawdown: float = 0.15                  # fraction from high water
    allow_new_positions: bool = True
    tradable_market: bool = True


@dataclass
class RiskContext:
    daily_pnl: float = 0.0          # fraction of equity (negative = losing)
    equity: float = 1_000_000.0
    drawdown: float = 0.0           # 0..1
    adv: Dict[str, float] = None    # symbol -> day amount
    tradable: bool = True
    mode: ExecutionMode = ExecutionMode.PAPER

    def __post_init__(self) -> None:
        if self.adv is None:
            self.adv = {}


def _norm_target(target_like: Union[Dict[str, Any], PortfolioTarget], equity: float) -> Dict[str, Any]:
    if isinstance(target_like, PortfolioTarget):
        weight = float(target_like.target_weight)
        side = str(target_like.meta.get("side", "BUY")).upper()
        raw_notional = target_like.meta.get("notional")
        notional = float(raw_notional) if raw_notional is not None else weight * equity
        return {
            "symbol": target_like.symbol,
            "strategy_id": target_like.strategy_id,
            "side": side,
            "weight": weight,
            "notional": notional,
        }
    weight = float(target_like.get("weight", target_like.get("target_weight", 0.0)))
    side = str(target_like.get("side", "BUY")).upper()
    raw_notional = target_like.get("notional")
    return {
        "symbol": str(target_like.get("symbol", "")),
        "strategy_id": str(target_like.get("strategy_id", "")),
        "side": side,
        "weight": weight,
        "notional": float(raw_notional) if raw_notional is not None else weight * equity,
    }


class RiskEngine:
    def __init__(self, config: Optional[RiskEngineConfig] = None) -> None:
        self.config = config or RiskEngineConfig()

    # ------------------------------------------------------------------ #
    def assess_target(
        self, target_like: Union[Dict[str, Any], PortfolioTarget], ctx: RiskContext
    ) -> RiskDecision:
        cfg = self.config
        t = _norm_target(target_like, ctx.equity)
        side = t["side"]
        is_buy = side == "BUY"
        checks: Dict[str, Any] = {}
        failed: list = []

        def record(name: str, passed: bool, detail: str) -> None:
            checks[name] = {"passed": bool(passed), "detail": detail}
            if not passed:
                failed.append(name)

        # ---- hard vetoes -------------------------------------------------
        record("mode_enabled", ctx.mode != ExecutionMode.DISABLED,
               f"execution_mode={ctx.mode.value}")
        record("market_tradable", ctx.tradable, f"tradable={ctx.tradable}")
        record("daily_loss_hard", not (ctx.daily_pnl <= cfg.daily_loss_hard),
               f"daily_pnl={ctx.daily_pnl:.4f} hard={cfg.daily_loss_hard}")
        record("new_positions_allowed",
               (not is_buy) or cfg.allow_new_positions,
               f"side={side} allow_new={cfg.allow_new_positions}")

        # ---- drawdown cap (buys only; exits must stay possible) ----------
        record("drawdown_cap", (not is_buy) or ctx.drawdown < cfg.max_drawdown,
               f"drawdown={ctx.drawdown:.4f} cap={cfg.max_drawdown}")

        # ---- daily loss soft (sells pass, buys blocked) -------------------
        record("daily_loss_soft",
               (not is_buy) or ctx.daily_pnl > cfg.daily_loss_soft,
               f"daily_pnl={ctx.daily_pnl:.4f} soft={cfg.daily_loss_soft}")

        # ---- sizing / liquidity ------------------------------------------
        record("notional_cap", t["notional"] <= cfg.max_order_notional + _EPS,
               f"notional={t['notional']:.0f} cap={cfg.max_order_notional:.0f}")
        record("weight_cap", t["weight"] <= cfg.max_weight_per_stock + _EPS,
               f"weight={t['weight']:.4f} cap={cfg.max_weight_per_stock}")
        adv = float(ctx.adv.get(t["symbol"], 0.0))
        record("liquidity_floor", adv >= cfg.min_liquidity_amount,
               f"adv={adv:.0f} floor={cfg.min_liquidity_amount:.0f}")
        record("participation_cap", t["notional"] <= cfg.participation_cap * adv + _EPS,
               f"order_value={t['notional']:.0f} limit={cfg.participation_cap * adv:.0f}")

        approved = not failed
        return RiskDecision(
            approved=approved,
            reason="" if approved else f"risk_blocked:{'+'.join(failed)}",
            checks=checks,
        )
