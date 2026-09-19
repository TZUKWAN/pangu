"""Pangu 2.0 验证层：无未来函数回测 v2 + 统计验证 + 实验登记。

模块一览：
- data_interface: ResearchData 协议复用 + SyntheticValidationData（场景注入）
- exec_model:     纯函数执行模型（涨跌停/费用/容量/同 bar 约定）
- backtest_v2:    BacktestV2 引擎 + HistoryView（LookaheadError 防线）
- walk_forward:   WalkForwardSplitter + HoldoutPolicy（一次解锁 + 审计）
- statistics:     deflated_sharpe / bootstrap_ci / pbo_cscv / BH 校正
- robustness:     参数邻域 / 成本压力 / 延迟压力 / 子期表现
- cross_check:    独立重定价交叉核对（P4-011）
- leakage:        静态泄漏扫描 + assert_label_alignment
- experiment_registry: append-only JSONL 实验登记簿
"""
from __future__ import annotations

from .data_interface import LookaheadError, ResearchData, SyntheticValidationData
from .exec_model import (
    FeeSchedule,
    Fill,
    fees,
    fill_buy,
    fill_sell,
    limit_prices,
    order_capacity,
    same_bar_order,
)
from .backtest_v2 import (
    BacktestConfig,
    BacktestResult,
    BacktestV2,
    HistoryView,
    Strategy,
    TargetOrder,
)
from .walk_forward import (
    HoldoutPolicy,
    HoldoutViolation,
    Split,
    WalkForwardSplitter,
)
from .statistics import (
    bootstrap_ci,
    daily_return_stats,
    deflated_sharpe,
    multiple_hypothesis_report,
    pbo_cscv,
)
from .robustness import (
    DelayedStrategy,
    cost_stress,
    delay_stress,
    param_neighborhood,
    subperiod_metrics,
)
from .cross_check import cross_check
from .leakage import assert_label_alignment, audit_module
from .experiment_registry import (
    ExperimentRegistry,
    MissingExperimentFields,
    REQUIRED_FIELDS,
)

__all__ = [
    "LookaheadError", "ResearchData", "SyntheticValidationData",
    "FeeSchedule", "Fill", "fees", "fill_buy", "fill_sell",
    "limit_prices", "order_capacity", "same_bar_order",
    "BacktestConfig", "BacktestResult", "BacktestV2", "HistoryView",
    "Strategy", "TargetOrder",
    "HoldoutPolicy", "HoldoutViolation", "Split", "WalkForwardSplitter",
    "bootstrap_ci", "daily_return_stats", "deflated_sharpe",
    "multiple_hypothesis_report", "pbo_cscv",
    "DelayedStrategy", "cost_stress", "delay_stress", "param_neighborhood",
    "subperiod_metrics",
    "cross_check", "assert_label_alignment", "audit_module",
    "ExperimentRegistry", "MissingExperimentFields", "REQUIRED_FIELDS",
]
