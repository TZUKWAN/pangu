"""Strategy registry package: manifests, lifecycle state machine, gates.

The registry is the ONLY way a strategy may legally change lifecycle status
(idea -> research -> validated -> paper -> shadow_live -> limited_live ->
live, with suspend/retire side edges).  Promotions into anything executable
run the P5-011 engineering gates in gates.py first.
"""
from .gates import (
    MAX_OOS_DRAWDOWN,
    MIN_OOS_PROFIT_FACTOR,
    MIN_OOS_SHARPE,
    MIN_TRADE_COUNT,
    MIN_TRADE_DATE_COUNT,
    GateResult,
    PromotionContext,
    evaluate_promotion,
)
from .manifest import (
    ALLOWED_INITIAL_STATUS,
    StrategyManifest,
    code_sha_for_module,
)
from .registry import (
    AUTO_SUSPEND_TRIGGERS,
    GateFailed,
    IllegalStrategyTransition,
    StrategyRegistry,
    seed_registry,
)

__all__ = [
    "AUTO_SUSPEND_TRIGGERS",
    "ALLOWED_INITIAL_STATUS",
    "GateFailed",
    "GateResult",
    "IllegalStrategyTransition",
    "MAX_OOS_DRAWDOWN",
    "MIN_OOS_PROFIT_FACTOR",
    "MIN_OOS_SHARPE",
    "MIN_TRADE_COUNT",
    "MIN_TRADE_DATE_COUNT",
    "PromotionContext",
    "StrategyManifest",
    "StrategyRegistry",
    "code_sha_for_module",
    "evaluate_promotion",
    "seed_registry",
]
