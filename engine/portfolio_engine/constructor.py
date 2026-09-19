"""Portfolio construction: turn strategy targets into a capped, scaled plan.

Rules encoded here are risk-budgeting engineering (per-stock / industry /
theme caps, drawdown and volatility scaling), NOT alpha: uncalibrated
targets are never penalized, only ordered after calibrated ones at equal
score, and cash is always an allowed allocation (100% cash is a valid plan).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..contracts import PortfolioTarget

_EPS = 1e-9


@dataclass
class PortfolioConfig:
    max_weight_per_stock: float = 0.10
    max_positions: int = 20
    max_industry_weight: float = 0.30
    max_theme_weight: float = 0.30
    base_exposure: float = 0.8
    drawdown_scalar: bool = True
    vol_scalar: bool = True
    min_confidence: float = 0.0
    # symbol -> industry/theme bucket; None disables industry/theme caps.
    industry_of: Optional[Callable[[str], Optional[str]]] = None

    def to_rule_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "industry_of"}
        d["industry_of"] = "provided" if self.industry_of is not None else None
        return d

    def rule_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_rule_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


@dataclass
class PortfolioPlanContext:
    decision_date: str
    equity: float
    current_positions: Dict[str, float] = field(default_factory=dict)  # symbol -> weight
    current_drawdown: float = 0.0          # 0..1, fraction below high water
    market_vol_pctile: Optional[float] = None  # 0..1 or None when unknown


@dataclass
class PortfolioPlan:
    decision_date: str
    allocations: List[Dict[str, Any]] = field(default_factory=list)
    cash_weight: float = 1.0
    notes: List[str] = field(default_factory=list)
    rule_hash: str = ""
    no_trade_reason: str = ""

    @property
    def gross_weight(self) -> float:
        return sum(float(a["weight"]) for a in self.allocations)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PortfolioConstructor:
    def __init__(self, config: Optional[PortfolioConfig] = None) -> None:
        self.config = config or PortfolioConfig()

    # ------------------------------------------------------------------ #
    def build(self, targets: List[PortfolioTarget], ctx: PortfolioPlanContext) -> PortfolioPlan:
        cfg = self.config
        notes: List[str] = []
        plan = PortfolioPlan(
            decision_date=ctx.decision_date, rule_hash=cfg.rule_hash()
        )

        if not targets:
            plan.cash_weight = 1.0
            plan.no_trade_reason = "no_targets"
            plan.notes = ["cash_only: fewer than 1 target"]
            return plan

        # ---- filtering --------------------------------------------------
        pool: List[PortfolioTarget] = []
        dropped_er = 0
        dropped_conf = 0
        for t in targets:
            if t.expected_return is not None and t.expected_return <= 0:
                dropped_er += 1
                continue
            if t.confidence is not None and t.confidence < cfg.min_confidence:
                dropped_conf += 1
                continue
            pool.append(t)
        if dropped_er:
            notes.append(f"dropped_zero_or_negative_expected_return={dropped_er}")
        if dropped_conf:
            notes.append(f"dropped_below_min_confidence={dropped_conf}")

        if not pool:
            plan.cash_weight = 1.0
            plan.no_trade_reason = "all_targets_filtered"
            plan.notes = notes + ["cash_only: every target filtered out"]
            return plan

        # ---- ordering: score desc, calibrated before uncalibrated -------
        def score_of(t: PortfolioTarget) -> float:
            if t.calibrated and t.confidence is not None:
                return float(t.confidence)
            return float(t.raw_score) if t.raw_score is not None else 0.0

        pool.sort(key=lambda t: (-score_of(t), not t.calibrated, t.symbol))
        n_uncal = sum(1 for t in pool if not t.calibrated)
        if n_uncal:
            notes.append(
                f"uncalibrated_targets_ranked_after_calibrated_at_equal_score={n_uncal}"
            )

        # ---- exposure scaling -------------------------------------------
        dd_scalar = 1.0
        if cfg.drawdown_scalar:
            dd_scalar = _clip(1.0 - _clip(float(ctx.current_drawdown), 0.0, 1.0), 0.2, 1.0)
        gross = cfg.base_exposure * dd_scalar
        if ctx.market_vol_pctile is not None:
            vol_brake = 1.0 - 0.5 * _clip(float(ctx.market_vol_pctile), 0.0, 1.0)
            gross *= vol_brake
            notes.append(f"market_vol_brake={vol_brake:.3f}")
        notes.append(f"exposure_scalar={gross / max(cfg.base_exposure, _EPS):.3f}")

        # ---- inverse-vol reweighting (budget preserving) ----------------
        vols = [float(t.meta["vol"]) for t in pool
                if t.meta.get("vol") is not None and float(t.meta["vol"]) > 0]
        ref_vol = sorted(vols)[len(vols) // 2] if vols else None
        if cfg.vol_scalar and ref_vol:
            notes.append(f"inverse_vol_reweight_ref_vol={ref_vol:.5f}")

        def adjusted_weight(t: PortfolioTarget) -> float:
            w = float(t.target_weight)
            if cfg.vol_scalar and ref_vol:
                v = t.meta.get("vol")
                if v is not None and float(v) > 0:
                    w *= _clip(ref_vol / float(v), 0.5, 2.0)
            return w

        # ---- cap enforcement ---------------------------------------------
        cap_by_bucket: Optional[float] = None
        if cfg.industry_of is None:
            notes.append("industry_cap_skipped: no industry_of mapping")
        else:
            cap_by_bucket = min(cfg.max_industry_weight, cfg.max_theme_weight)

        bucket_used: Dict[str, float] = {}
        used = 0.0
        skipped_industry = 0
        skipped_gross = 0
        for t in pool:
            if len(plan.allocations) >= cfg.max_positions:
                notes.append(f"max_positions_cap_hit={cfg.max_positions}")
                break
            remaining = gross - used
            if remaining <= _EPS:
                skipped_gross += 1
                continue

            w = min(adjusted_weight(t), cfg.max_weight_per_stock, remaining)
            if w <= _EPS:
                continue

            if cap_by_bucket is not None:
                bucket = cfg.industry_of(t.symbol)
                if bucket is not None:
                    room = cap_by_bucket - bucket_used.get(bucket, 0.0)
                    if room <= _EPS:
                        skipped_industry += 1
                        continue
                    w = min(w, room)
                    bucket_used[bucket] = bucket_used.get(bucket, 0.0) + w
                else:
                    notes.append(f"industry_unknown_no_cap={t.symbol}")

            used += w
            plan.allocations.append({
                "symbol": t.symbol,
                "weight": round(w, 10),
                "strategy_id": t.strategy_id,
                "calibrated": bool(t.calibrated),
            })

        if skipped_industry:
            notes.append(f"industry_cap_excluded_targets={skipped_industry}")
        if skipped_gross:
            notes.append(f"exhausted_budget_skipped_targets={skipped_gross}")

        plan.cash_weight = round(max(0.0, 1.0 - plan.gross_weight), 10)
        plan.notes = notes
        return plan
