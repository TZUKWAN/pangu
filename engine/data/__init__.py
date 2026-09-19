"""PIT 数据底座包（Pangu 2.0 Phase 1）。

- PITStore：全市场日线/宇宙/行业/指数/公告/新闻的因果读取层。
- UniverseBuilder：asof 可投资宇宙（带剔除原因）。
- DataQualityReport：数据完整度/新鲜度/PIT 安全判定。
- lookahead：前视偏差断言与源码扫描。
- corporate_actions：除权候选推断（仅诊断用；收益一律用 pct_change）。
"""
from .corporate_actions import detect_ex_right_candidates, persist_corporate_actions
from .lookahead import (
    LOOKAHEAD_PATTERNS,
    assert_no_future_rows,
    scan_module_for_lookahead,
    scan_source,
    to_compact,
    to_iso,
)
from .pit_store import PITStore
from .quality import DataQualityReport
from .universe import UniverseBuilder

__all__ = [
    "PITStore",
    "UniverseBuilder",
    "DataQualityReport",
    "assert_no_future_rows",
    "scan_module_for_lookahead",
    "scan_source",
    "LOOKAHEAD_PATTERNS",
    "to_compact",
    "to_iso",
    "detect_ex_right_candidates",
    "persist_corporate_actions",
]
