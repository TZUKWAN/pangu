"""Promotion gates between strategy lifecycle statuses.

All thresholds below are ENGINEERING BARS, not return promises: they exist
to force a strategy to clear a minimum evidence bar (P5-011 paper-candidate
bar) before it may touch paper / shadow / real money.  Passing them says
nothing about future returns.

Note on paper -> shadow_live: shadow_live is a READ-ONLY observation stage
(no orders, no market impact, no client money), so compliance_ok is NOT
required there; it becomes mandatory from shadow_live -> limited_live, when
real (small) capital is at risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..contracts import StrategyStatus

# P5-011 paper-candidate engineering bar (aligned with the replay
# acceptance contract in engine/short_term_replay.py).
MIN_OOS_PROFIT_FACTOR = 1.2
MIN_OOS_SHARPE = 1.0
MAX_OOS_DRAWDOWN = 0.15
MIN_TRADE_COUNT = 200
MIN_TRADE_DATE_COUNT = 120


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class PromotionContext:
    """Evidence bundle handed to the registry together with a promotion."""

    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    paper_report_ref: str = ""
    shadow_report_ref: str = ""
    compliance_ok: bool = False
    broker_ok: bool = False
    reconcile_ok: bool = False
    kill_switch_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_metrics": dict(self.validation_metrics),
            "paper_report_ref": self.paper_report_ref,
            "shadow_report_ref": self.shadow_report_ref,
            "compliance_ok": bool(self.compliance_ok),
            "broker_ok": bool(self.broker_ok),
            "reconcile_ok": bool(self.reconcile_ok),
            "kill_switch_ready": bool(self.kill_switch_ready),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PromotionContext":
        return cls(
            validation_metrics=dict(d.get("validation_metrics", {})),
            paper_report_ref=str(d.get("paper_report_ref", "")),
            shadow_report_ref=str(d.get("shadow_report_ref", "")),
            compliance_ok=bool(d.get("compliance_ok", False)),
            broker_ok=bool(d.get("broker_ok", False)),
            reconcile_ok=bool(d.get("reconcile_ok", False)),
            kill_switch_ready=bool(d.get("kill_switch_ready", False)),
        )


@dataclass
class GateResult:
    approved: bool
    failed_checks: List[str] = field(default_factory=list)
    checked_at: str = field(default_factory=_now_iso)


def _num(metrics: Dict[str, Any], key: str, failed: List[str]) -> Optional[float]:
    val = metrics.get(key)
    if val is None:
        failed.append(f"{key}_missing")
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        failed.append(f"{key}_not_numeric")
        return None


def _flag(metrics: Dict[str, Any], key: str, failed: List[str]) -> None:
    if not metrics.get(key):
        failed.append(key if metrics.get(key) is not None else f"{key}_missing")


def _paper_bar(ctx: PromotionContext, failed: List[str]) -> None:
    """P5-011 bar for validated -> paper (reused by limited_live -> live)."""
    m = ctx.validation_metrics
    pf = _num(m, "oos_profit_factor", failed)
    if pf is not None and pf < MIN_OOS_PROFIT_FACTOR:
        failed.append("oos_profit_factor_below_1p2")
    if pf is not None and pf <= 1.0:
        failed.append("oos_profit_factor_not_above_1")
    sharpe = _num(m, "oos_sharpe", failed)
    if sharpe is not None and sharpe < MIN_OOS_SHARPE:
        failed.append("oos_sharpe_below_1p0")
    mdd = _num(m, "oos_max_drawdown", failed)
    if mdd is not None and mdd > MAX_OOS_DRAWDOWN:
        failed.append("oos_max_drawdown_above_15pct")
    tc = _num(m, "trade_count", failed)
    if tc is not None and tc < MIN_TRADE_COUNT:
        failed.append("trade_count_below_200")
    tdc = _num(m, "trade_date_count", failed)
    if tdc is not None and tdc < MIN_TRADE_DATE_COUNT:
        failed.append("trade_date_count_below_120")
    _flag(m, "cost2x_positive", failed)
    _flag(m, "parameter_stable", failed)
    _flag(m, "leakage_audit_passed", failed)
    _flag(m, "independent_backtest_consistent", failed)
    ne = _num(m, "oos_net_expectancy", failed)
    if ne is not None and ne <= 0:
        failed.append("oos_net_expectancy_not_positive")


def _infra_flags(ctx: PromotionContext, failed: List[str]) -> None:
    if not ctx.paper_report_ref.strip():
        failed.append("paper_report_ref_missing")
    if not ctx.shadow_report_ref.strip():
        failed.append("shadow_report_ref_missing")
    if not ctx.broker_ok:
        failed.append("broker_ok_false")
    if not ctx.reconcile_ok:
        failed.append("reconcile_ok_false")
    if not ctx.compliance_ok:
        failed.append("compliance_ok_false")
    if not ctx.kill_switch_ready:
        failed.append("kill_switch_not_ready")


def evaluate_promotion(ctx: PromotionContext, from_status, to_status) -> GateResult:
    """Evaluate the gate for one lifecycle edge.  Returns GateResult.

    Supported edges:
      validated    -> paper        : full P5-011 OOS bar (also applies to
                                      suspended -> paper resumption).
      paper        -> shadow_live  : paper_report_ref + broker_ok + reconcile_ok.
                                      compliance_ok NOT required (shadow is
                                      read-only, no orders, no money at risk).
      shadow_live  -> limited_live : shadow_report_ref + compliance_ok +
                                      kill_switch_ready.
      limited_live -> live         : full paper bar (oos metrics still present)
                                      plus all report refs / infra flags.
    Any other edge has no gate defined and fails closed.
    """
    failed: List[str] = []
    from_s, to_s = StrategyStatus(from_status), StrategyStatus(to_status)

    if to_s == StrategyStatus.PAPER and from_s in (StrategyStatus.VALIDATED,
                                                   StrategyStatus.SUSPENDED):
        # Full P5-011 bar.  A strategy resuming from suspended into the
        # executable track (suspended -> paper) must re-clear the same bar.
        _paper_bar(ctx, failed)
    elif from_s == StrategyStatus.PAPER and to_s == StrategyStatus.SHADOW_LIVE:
        if not ctx.paper_report_ref.strip():
            failed.append("paper_report_ref_missing")
        if not ctx.broker_ok:
            failed.append("broker_ok_false")
        if not ctx.reconcile_ok:
            failed.append("reconcile_ok_false")
        # compliance_ok intentionally NOT required: shadow_live is read-only.
    elif from_s == StrategyStatus.SHADOW_LIVE and to_s == StrategyStatus.LIMITED_LIVE:
        if not ctx.shadow_report_ref.strip():
            failed.append("shadow_report_ref_missing")
        if not ctx.compliance_ok:
            failed.append("compliance_ok_false")
        if not ctx.kill_switch_ready:
            failed.append("kill_switch_not_ready")
    elif from_s == StrategyStatus.LIMITED_LIVE and to_s == StrategyStatus.LIVE:
        _paper_bar(ctx, failed)  # oos metrics must still be present and passing
        _infra_flags(ctx, failed)
    else:
        failed.append(f"no_promotion_gate_defined:{from_s.value}->{to_s.value}")

    return GateResult(approved=not failed, failed_checks=failed, checked_at=_now_iso())
